import numpy as np
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
        return self.engine.neighbours(embedding)

