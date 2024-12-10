import numpy
from nearpy import Engine
from nearpy.filters import DistanceThresholdFilter
from nearpy.distances import CosineDistance
from nearpy.hashes import RandomBinaryProjections
from lsh import LSH

dimension = 100

# Create random query vector
query = numpy.random.randn(dimension)
query2 = numpy.random.randn(dimension)
query3 = numpy.random.randn(dimension)

lsh = LSH(10, threshold=0.99, dimension=100)
lsh.insert(query, True)
lsh.insert(query2, False)

print(lsh.query(query))
query2results = lsh.query(query2)
if len(query2results) > 0:
    print("results from query 2")
    result = query2results[0]
    print(result[-2])