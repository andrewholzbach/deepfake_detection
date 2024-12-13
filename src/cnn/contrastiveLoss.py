import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import random
import numpy as np
import time
import pickle

#device = cpu if run locally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# image preprocessing + transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Paths to datasets
real_dir = "src/cnn/train_realImages"
fake_dir = "src/cnn/train_fakeImages"
test_real = "src/cnn/test_realImages"
test_fake = "src/cnn/test_fakeImages"

#ImageDataset represents single image inputs
class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        image = Image.open(self.image_paths[idx])

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image

#Represents paired inputs for the contrastive loss
class PairedDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_images = [os.path.join(real_dir, fname) for fname in os.listdir(real_dir)]
        self.fake_images = [os.path.join(fake_dir, fname) for fname in os.listdir(fake_dir)]
        self.transform = transform

    def __len__(self):
        return max(len(self.real_images), len(self.fake_images))

    def __getitem__(self, idx):
        # Randomly decide the type of pair (real-real, fake-fake, or real-fake)
        pair_type = random.choice(['real-real', 'fake-fake', 'real-fake'])

        if pair_type == 'real-real':
            img1 = Image.open(self.real_images[idx % len(self.real_images)]).convert("RGB")
            img2 = Image.open(self.real_images[(idx + 1) % len(self.real_images)]).convert("RGB")
            label = 0  # Similar pair (real-real)
        elif pair_type == 'fake-fake':
            img1 = Image.open(self.fake_images[idx % len(self.fake_images)]).convert("RGB")
            img2 = Image.open(self.fake_images[(idx + 1) % len(self.fake_images)]).convert("RGB")
            label = 0  # Similar pair (fake-fake)
        else:
            img1 = Image.open(self.real_images[idx % len(self.real_images)]).convert("RGB")
            img2 = Image.open(self.fake_images[idx % len(self.fake_images)]).convert("RGB")
            label = 1  # Dissimilar pair (real-fake)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label)

# Define the CNN model
class ContrastiveCNN(nn.Module):
    def __init__(self):
        super(ContrastiveCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Contrastive loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = (1 - label) * torch.pow(euclidean_distance, 2) + \
               label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return loss.mean()

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=8):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for img1, img2, labels in train_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            optimizer.zero_grad()

            output1 = model(img1)
            output2 = model(img2)
            loss = criterion(output1, output2, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

                output1 = model(img1)
                output2 = model(img2)
                loss = criterion(output1, output2, labels)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")




def evaluate(model, test_real_dataloader, real_dataloader, fake_dataloader, device='cpu', threshold=0.5):

    total_time = 0.0  # To accumulate processing time

    model.eval()

    start_time = time.time() 
    real_embeddings = []
    for real_images in real_dataloader:
        real_images = real_images.to(device)
        with torch.no_grad():
            real_embeddings.append(model(real_images))
    real_embeddings = torch.cat(real_embeddings)

    fake_embeddings = []
    for fake_images in fake_dataloader:
        fake_images = fake_images.to(device)
        with torch.no_grad():
            fake_embeddings.append(model(fake_images))
    fake_embeddings = torch.cat(fake_embeddings)

    correct_count = 0
    total_count = 0

    end_time = time.time() 
    batch_time = (end_time - start_time) * 1000  # to get time in milliseconds
    total_time += batch_time
    

    with torch.no_grad():
        for test_real_images in test_real_dataloader:
            start_time = time.time() 
            test_real_images = test_real_images.to(device)

        
            test_real_embeddings = model(test_real_images)

            real_distances = F.pairwise_distance(
                test_real_embeddings.unsqueeze(1),
                real_embeddings.unsqueeze(0),
                keepdim=False
            ).mean(dim=1)

            fake_distances = F.pairwise_distance(
                test_real_embeddings.unsqueeze(1),
                fake_embeddings.unsqueeze(0),
                keepdim=False
            ).mean(dim=1)

            preds = (real_distances < fake_distances).int()
            
            correct_count += preds.sum().item()
            total_count += test_real_images.size(0)
            end_time = time.time() 
            batch_time = (end_time - start_time) * 1000  # to get time in milliseconds
            total_time += batch_time

    accuracy = correct_count / total_count
    average_time_per_image = total_time / total_count 

    return accuracy, average_time_per_image


#Train the model
# train_model(model, train_loader, val_loader, criterion, optimizer, epochs=8)

# #save the model to pickle
# with open('model.pkl', 'wb') as file:
#     pickle.dump(model, file)

# with open('model.pkl', 'rb') as file:
#     model = pickle.load(file)

# #Get accuracy metrics
# accuracy, avgTime1 = evaluate(model, testReal_loader, real_loader, fake_loader, threshold=0.5)
# print()
# accuracy2, avgTime2 = evaluate(model, testFake_loader, real_loader, fake_loader, threshold=0.5)
# accuracy2 = 1- accuracy2
# print(f"Accuracy of labeling real test images as real: {accuracy}")
# print(f"Accuracy of labeling fake test images as fake: {accuracy2}")
# print()
# avgTime = (avgTime1 + avgTime2)/2
# print(f"Average time taken per prediction (in milliseconds): {avgTime}")
# print()

def save_embeddings(model, dataloader, file_path, device='cpu'):
    model.eval()  
    all_embeddings = []

    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu().numpy())  

    all_embeddings = np.vstack(all_embeddings)

    with open(file_path, 'w') as f:
        for embedding in all_embeddings:
            embedding_line = ' '.join(map(str, embedding))
            f.write(embedding_line + '\n')

    print(f"Saved {len(all_embeddings)} embeddings to {file_path}")