import numpy as np
from sklearn.random_projection import SparseRandomProjection

class LSH:
    def __init__(self, n_buckets=10, n_dimensions=128):
        self.n_buckets = n_buckets
        self.random_proj = SparseRandomProjection(n_components=n_buckets)

    #TODO
    def hash_embedding(self, embedding):
        return ""
