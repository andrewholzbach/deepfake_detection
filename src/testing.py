import unittest
import torch
import numpy as np
from models.cnn import FeatureExtractor
from utils.data_loader import get_data_loader
from lsh.lsh import LSH
import torch.nn.functional as F

class TestFeatureExtractorCNN(unittest.TestCase):
    def test_output_shape(self):
        model = FeatureExtractor()
        model.eval()
        x = torch.randn(1, 3, 224, 224)  # Simulate a single 224x224 image
        output = model(x)
        self.assertEqual(output.shape, (1, 128))  # Expected embedding shape (batch_size, embedding_dim)

    def test_load_images_compare(self):
        model = FeatureExtractor()
        model.eval()
        data_loader = get_data_loader(data_path='../face_images/adam_test', batch_size=32)
        adam_1 = np.array([])
        for images, labels in data_loader:
            embeddings = model(images)
            print("length of images: ", len(images))
            print(labels)
            print(len(embeddings[0]))

            euclidean_distance1 = F.cosine_similarity(embeddings[0], embeddings[1], dim=0)
            euclidean_distance2 = F.cosine_similarity(embeddings[0], embeddings[2], dim=0)
            print(f'Cosine similarity between adams: {euclidean_distance1}')
            print(f'Cosine similarity from adam to other: {euclidean_distance2}')
            # Further training steps...
        


if __name__ == '__main__':
    unittest.main()
