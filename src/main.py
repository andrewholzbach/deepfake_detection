import torch
import numpy as np
from models.cnn import FeatureExtractor
from utils.data_loader import get_data_loader
from lsh.lsh import LSH

def main():
    # Initialize CNN and LSH
    cnn_model = FeatureExtractor()
    lsh = LSH(n_buckets=10, n_dimensions=128)

    # Load data
    data_loader = get_data_loader(data_path='../face_images', batch_size=32)

    # Example training loop (simplified)
    for images, labels in data_loader:
        embeddings = cnn_model(images)
        
        
        hashed_embeddings = lsh.fit_transform(embeddings.detach().numpy())

        vector_a = embeddings[0].detach().numpy()
        vector_b = embeddings[1].detach().numpy()

        euclidean_distance = np.linalg.norm(vector_a - vector_b)
        print(f'Euclidean Distance: {euclidean_distance}')
        # Further training steps...

if __name__ == "__main__":
    main()
