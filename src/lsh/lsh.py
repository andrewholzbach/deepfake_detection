import numpy as np
from nearpy import Engine
from nearpy.filters import DistanceThresholdFilter, NearestFilter
from nearpy.hashes import RandomBinaryProjections, LSHash
from nearpy.distances import CosineDistance


class LSH:
    def __init__(self, dimension, num_hash_funcs:int):
        hashes = [RandomBinaryProjections(f'rbp_{i}', 128) for i in range(num_hash_funcs)]
        self.engine = Engine(dimension, lshashes=hashes)

    def insert(self, embedding, real:bool):
        if real:
            self.engine.store_vector(embedding, 'real')
        else:
            self.engine.store_vector(embedding, 'fake')

    def query(self, embedding):
        neighbors = self.engine.neighbours(embedding)
        return neighbors
    
    def query_percent_real(self, embedding) -> float:
        neighbors = self.query(embedding)
        total_size = len(neighbors)
        total_real = 0
        for neighbor in neighbors:
            if neighbor[1] == "real":
                total_real += 1
        return total_real/total_size
        
