import pandas as pd
import numpy as np
from itertools import permutations

# These are the walk-centric indices
def walk_centric_shapley():


    return 0


# These are the data-centric indices
def walk_visitation(data):
    n = data.shape[0]           # get the number of sources
    r = data.shape[0]           # get the number of items we want combinations

    arr = np.arange(n)             # get the numbers that we'll be getting combos

    perms = list(permutations(arr))  # get all potential combinations
    walks = {key: 0 for key in perms}

    for i in range(0, data.shape[1]):
        ind = tuple(np.argsort(data[:,i]))
        walks[ind] = walks[ind] + 1

    return walks

def percentage_walks(data):
    walks = walk_visitation(data)
    n = data.shape[0]
    observed = 0
    for walk in walks.keys():
        if walks[walk] != 0:
            observed += 1

    possible_walks = np.math.factorial(n)
    percent = observed / possible_walks
    
    return percent, walks

if __name__ == '__main__':
    # instatiate ChoquetIntegral object
    # create data samples and labels to produce a max aggregation operation

    data = np.random.rand(5, 250)
    walks = walk_visitation(data)
    
    percent, walks = percentage_walks(data)
