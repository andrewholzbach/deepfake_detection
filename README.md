# Comp 480

## Requirements
First, ensure you have the following Python packages installed
- subprocess
- numpy
- matplotlib
- seaborn
- scikit-learn
- math
- random
- pickle
- pandas
- requests
- os
- time
- pillow
- torch
- nearpy

If you are missing any dependencies, please run pip install < missing_dependency_list > to resolve them

## Scripts
- run `python main.py` from the top level directory to generate all results for all parts. 

The results will be generated as described below.

## LSH PLOTS
- `timeByHashFunc.py`: Plots the number of hash functions used for LSH vs the average time per query 
- `errorRates.png`: Plots the number of hash functions used for LSH vs the average error rate

## CNN METRICS
These are generated to the terminal with every training of the model
- `src/cnn/metrics.txt` : Contains the training and validation loss per epoch of training, and the time per exhaustive query
