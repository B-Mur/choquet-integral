import pandas as pd
import numpy as np
from itertools import permutations, combinations

# These are the walk-centric indices
def walk_centric_shapley(fm):




    return 0


# These are the data-centric indices
def walk_visitation(data):
    n = data.shape[0]           # get the number of sources
    r = data.shape[0]           # get the number of items we want combinations

    arr = np.arange(n) + 1             # get the numbers that we'll be getting combos

    perms = list(permutations(arr))  # get all potential combinations
    walks = {key: 0 for key in perms}

    for i in range(0, data.shape[1]):
        ind = tuple(np.argsort(data[:,i]) + 1)
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


def variable_visitation(data):
    n = data.shape[0]
    arr = np.arange(n) + 1
    # this sets up our basic variables.
    fm_variables = [0]
    for i in range(1, n + 1):
        current_combos = list(combinations(arr, i))
        for ele in current_combos:
            fm_variables.append(ele)

    # how do i count each variable??
    # get all the walks taken
    walk = walk_visitation(data)
    vars = {key: 0 for key in fm_variables}
    for key in walk.keys():
        for i in range(0, walk[key]):
            variable_ind = []
            for val in key:
                if val != 0:
                    variable_ind.append(val)
                    variable_ind.sort()
                    idx = tuple(variable_ind)
                    vars[idx] = vars[idx] + 1


    vars[0] = vars[idx]

    return vars


def harden_variable_visitation(data):
    n = data.shape[0]
    arr = np.arange(n) + 1

    fm_variables = [0]
    for i in range(1, n + 1):
        current_combos = list(combinations(arr, i))
        for ele in current_combos:
            fm_variables.append(ele)

    # get all the walks taken
    vars = {key: 0 for key in fm_variables}

    # how do i count each variable??
    # get all the walks taken
    walk = walk_visitation(data)
    hard_variables = {key: 0 for key in fm_variables}
    for key in walk.keys():
        for i in range(0, walk[key]):
            variable_ind = []
            for val in key:
                if val != 0:
                    variable_ind.append(val)
                    variable_ind.sort()
                    idx = tuple(variable_ind)
                    vars[idx] = 1

    return hard_variables


def percentage_variables(data):
    var_visitation = variable_visitation(data)
    n = data.shape[0]
    observed = 0
    for var in var_visitation.keys():
        if var_visitation[var] != 0:
            observed += 1


    number_vars = 2**(n)
    percent = observed / number_vars

    return percent, var_visitation


if __name__ == '__main__':
    # instatiate ChoquetIntegral object
    # create data samples and labels to produce a max aggregation operation

    data = np.random.rand(3, 2)
    walks = walk_visitation(data)
    
    percentage_of_walks, walks = percentage_walks(data)

    vars = variable_visitation(data)

    percentage_of_variables, vars = percentage_variables(data)

    print(f'Percentage of variables seen: {percentage_of_variables}')







