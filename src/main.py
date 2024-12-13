import torch
import numpy as np
import matplotlib.pyplot as plt
from lsh.lsh import LSH
from cnn.contrastiveLoss import (
    PairedDataset,
    ImageDataset,
    DataLoader,
    random_split,
    ContrastiveCNN,
    ContrastiveLoss,
    train_model,
    evaluate,
    save_embeddings
)
from lsh.lsh_testing import (
    embeddings_to_dataframe,
    evaluate_lsh
)

import torchvision.transforms as transforms

#device = cpu if run locally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

real_dir = "cnn/train_realImages"
fake_dir = "cnn/train_fakeImages"
test_real = "cnn/test_realImages"
test_fake = "cnn/test_fakeImages"

# image preprocessing + transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def main():
    # run_base_evaluate()
    num_hash_funcs_list = [1, 2, 3, 5, 10, 20]
    avg_times = []
    false_positive_rates = []
    false_negative_rates = []

    for num_hash_funcs in num_hash_funcs_list:
        avg_time, real_rate, fake_rate = run_lsh_evaluate(num_hash_funcs)
        false_positive_rate = 1 - real_rate
        false_negative_rate = 1 - fake_rate
        avg_times.append(avg_time)
        false_positive_rates.append(false_positive_rate)
        false_negative_rates.append(false_negative_rate)

    plot_avg_time(num_hash_funcs_list, avg_times)
    plot_error_rates(num_hash_funcs_list, false_positive_rates, false_negative_rates)

def run_base_evaluate():
     # Create datasets
    dataset = PairedDataset(real_dir, fake_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    real_dataset = ImageDataset(real_dir, transform=transform)
    fake_dataset = ImageDataset(fake_dir, transform=transform)
    testReal_dataset = ImageDataset(test_real, transform=transform)
    testFake_dataset = ImageDataset(test_fake, transform=transform)

    # Create DataLoaders
    real_loader = DataLoader(real_dataset, batch_size=32, shuffle=True)
    fake_loader = DataLoader(fake_dataset, batch_size=32, shuffle=True)
    testReal_loader = DataLoader(testReal_dataset, batch_size=32, shuffle=True)
    testFake_loader = DataLoader(testFake_dataset, batch_size=32, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model, loss, and optimizer
    model = ContrastiveCNN().to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=8)

    # Evaluate the model
    accuracy_real, avgTime_real = evaluate(model, testReal_loader, real_loader, fake_loader, threshold=0.5)
    accuracy_fake, avgTime_fake = evaluate(model, testFake_loader, real_loader, fake_loader, threshold=0.5)
    accuracy_fake = 1 - accuracy_fake

    print(f"\nAccuracy of labeling real test images as real: {accuracy_real}")
    print(f"Accuracy of labeling fake test images as fake: {accuracy_fake}\n")
    avg_time = (avgTime_real + avgTime_fake) / 2
    print(f"Average time taken per prediction (in milliseconds): {avg_time}\n")

    # Save embeddings
    save_embeddings(model, real_loader, "embeddings/real_embeddings.txt", device)
    save_embeddings(model, fake_loader, "embeddings/fake_embeddings.txt", device)
    save_embeddings(model, testReal_loader, "embeddings/test_real_embeddings.txt", device)
    save_embeddings(model, testFake_loader, "embeddings/test_fake_embeddings.txt", device)

def run_lsh_evaluate(num_hash_functions: int):
    real_embeddings_df = embeddings_to_dataframe("embeddings/real_embeddings.txt")
    fake_embeddings_df = embeddings_to_dataframe("embeddings/fake_embeddings.txt")
    test_real_embeddings_df = embeddings_to_dataframe("embeddings/test_real_embeddings.txt")
    test_fake_embeddings_df = embeddings_to_dataframe("embeddings/test_fake_embeddings.txt")

    dimension = real_embeddings_df.shape[1] 


    lsh = LSH(dimension, num_hash_funcs=num_hash_functions)


    accuracy_real, accuracy_fake, avg_time = evaluate_lsh(lsh, test_real_embeddings_df, test_fake_embeddings_df, real_embeddings_df, fake_embeddings_df)

    print(f"Labeling test embeddings with {num_hash_functions} hash functions")
    print(f"Accuracy of labeling real test embeddings as real: {accuracy_real}")
    print(f"Accuracy of labeling fake test embeddings as fake: {accuracy_fake}")
    print()
    print(f"Average time per prediction (in milliseconds): {avg_time}")
    return avg_time, accuracy_real, accuracy_fake
    
def plot_avg_time(num_hash_funcs_list, avg_times):
    plt.figure(figsize=(8, 6))
    plt.plot(num_hash_funcs_list, avg_times, marker='o', label='Average Query Time')
    plt.xlabel('Number of Hash Functions')
    plt.ylabel('Average Time per Query (ms)')
    plt.title('Average Time per Query vs. Number of Hash Functions')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_error_rates(num_hash_funcs_list, false_positive_rates, false_negative_rates):
    plt.figure(figsize=(8, 6))
    plt.plot(num_hash_funcs_list, false_positive_rates, marker='o', label='False Positive Rate')
    plt.plot(num_hash_funcs_list, false_negative_rates, marker='o', label='False Negative Rate')
    plt.xlabel('Number of Hash Functions')
    plt.ylabel('Error Rate')
    plt.title('Error Rates vs. Number of Hash Functions')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

