import numpy as np
import time
from nearpy import Engine
from nearpy.filters import DistanceThresholdFilter
from nearpy.hashes import RandomBinaryProjections


class LSH:
    def __init__(self, num_hash_funcs:int, threshold=0.99, dimension=100):
        hashes = [RandomBinaryProjections('rbp', 10) for _ in range(num_hash_funcs)]
        self.engine = Engine(dimension, lshashes=hashes, vector_filters=[DistanceThresholdFilter(threshold)])

    def insert(self, embedding, real:bool):
        if real:
            self.engine.store_vector(embedding, 'real')
        else:
            self.engine.store_vector(embedding, 'fake')

    def query(self, embedding):
        start = time.time()
        neighbors = self.engine.neighbours(embedding)
        end = time.time()
        length = end-start
        print("query took %i seconds", length)
        return neighbors
    
    def query_percent_real(self, embedding) -> float:
        neighbors = self.query(embedding)
        total_size = len(neighbors)
        total_real = 0
        for neighbor in neighbors:
            print("neighbor ", neighbor)
            if neighbor[1] == "real":
                total_real += 1
        return total_real/total_size
        


