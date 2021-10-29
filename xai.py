import pandas as pd
import numpy as np
from itertools import permutations, combinations
from choquet_integral import ChoquetIntegral


def get_layer_vars(fm, l, s):
    '''
    This function returns the fm variable indices that are in the current layer that do
    contain the source of interest.
    :param fm: fuzzy measure
    :param l:  which layer
    :param s:  which source
    :return:   indices of fm variables of interest.
    '''

    inds = []
    for key in fm.keys():
        ind = str(key)
        ind = ind.strip('][ ').replace(' ', '').split(',')
        if len(ind[0]) == l:
            if str(s) not in ind[0]:
                inds.append(key)

    return inds


def get_source_var(fm, curr_inds, s, l):
    '''
    This function returns the fm variable indices that are in the previous layer that do not
    contain the source of interest.
    :param fm: fuzzy measure
    :param l:  which layer
    :param s:  which source
    :return:   indices of fm variables of interest.
    '''
    pairs = []
    for key in fm.keys():
        for cur in curr_inds:
            ind = str(key)
            ind = ind.strip('][ ').replace(' ', '').split(',')[0]
            cur_ind = cur.strip('][ ').replace(' ', '').split(',')[0]
            if len(ind) == l:
                if cur_ind in ind and str(s) in ind:
                    pairs.append([key, cur])

    return pairs


def get_shapl_coef(X, K):
    val = (np.math.factorial(X - K - 1) * np.math.factorial(K)) / np.math.factorial(X)
    return val


####################################
# These are the walk-centric indices
####################################
def walk_centric_shapley(fm, data):
    shapley_values = []
    seen_vars = harden_variable_visitation(data)
    num_layers = data.shape[0] + 1
    num_sources = data.shape[0]
    # Get the coefficient
    coefficients = 0
    # Do the subtractions
    for s in range(1, num_sources + 1):
        running_sum = 0
        for l in range(0, num_layers - 1):
            if l == 0:
                var = str([s])
                X, K = num_sources, 0
                coef = get_shapl_coef(X, K)
                running_sum += coef * (fm[var] - 0)
            else:
                curr_inds = get_layer_vars(fm, l, s)
                pairs = get_source_var(fm, curr_inds, s, l + 1)
                X, K = num_sources, l
                coef = get_shapl_coef(X, K)
                for set, seti in pairs:
                    if seen_vars[seti] == 1 and seen_vars[set] == 1:
                        running_sum += coef * (fm[seti] - fm[set])

        shapley_values.append(running_sum)

    shapley_values = np.asarray(shapley_values)
    shapley_values = shapley_values / np.sum(shapley_values)

    return shapley_values


####################################
# These are the data-centric indices
####################################
def walk_visitation(data):
    n = data.shape[0]  # get the number of sources
    r = data.shape[0]  # get the number of items we want combinations
    m = data.shape[1]  # get the number of samples

    arr = np.arange(n) + 1  # get the numbers that we'll be getting combos

    perms = list(permutations(arr))  # get all potential combinations
    walks = {key: 0 for key in perms}

    for i in range(0, data.shape[1]):
        ind = tuple(np.argsort(data[:, i]) + 1)
        walks[ind] = walks[ind] + 1

    z_walks = walks.copy()
    for key in walks.keys():
        z_walks[key] = z_walks[key] / m

    return walks, z_walks


def percentage_walks(data):
    walks, z_walks = walk_visitation(data)
    n = data.shape[0]
    observed = 0
    for walk in walks.keys():
        if walks[walk] != 0:
            observed += 1

    possible_walks = np.math.factorial(n)
    percent = observed / possible_walks

    return percent, z_walks


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
    walk, z_walk = walk_visitation(data)
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
            fm_variables.append(list(ele))

    # how do i count each variable??
    # get all the walks taken
    walk, z_walks = walk_visitation(data)
    hard_variables = {str(key).replace(',', ''): 0 for key in fm_variables}
    for key in walk.keys():
        for i in range(0, walk[key]):
            variable_ind = []
            for val in key:
                if val != 0:
                    variable_ind.append(val)
                    variable_ind.sort()
                    idx = str(variable_ind).replace(',', '')
                    hard_variables[idx] = 1

    return hard_variables


def percentage_variables(data):
    var_visitation = variable_visitation(data)
    n = data.shape[0]
    observed = 0
    for var in var_visitation.keys():
        if var_visitation[var] != 0:
            observed += 1

    number_vars = 2 ** (n)
    percent = observed / number_vars

    return percent, var_visitation


if __name__ == '__main__':
    # instatiate ChoquetIntegral object
    # create data samples and labels to produce a max aggregation operation

    data = np.random.rand(3, 2)
    labels = np.amax(data, 0)  # we're building a max

    walks = walk_visitation(data)

    percentage_of_walks, walks = percentage_walks(data)

    vars = variable_visitation(data)

    percentage_of_variables, vars = percentage_variables(data)

    print(f'Percentage of variables seen: {percentage_of_variables}')

