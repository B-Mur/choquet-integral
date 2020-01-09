import pandas as pd
import numpy as np
from itertools import permutations, combinations
from ChoquetIntegral.choquet_integral import ChoquetIntegral

def build_shapley_dist_mat(shap_mat):
    s = shap_mat.shape[1]
    mat = np.zeros((shap_mat.shape[0], s, s))
    for k, row in enumerate(shap_mat.iterrows()):
        for i in range(0, s):
            for j in range(0, s):
                mat[k, i, j] = np.abs(row[1][i] - row[1][j])

    return mat


def get_shapley_protoform_max(shap_mat):
    f = np.amax(np.amax(shap_mat, 2), 1)
    return f


def get_shapley_protoform_anderson(shap_mat):
    num = shap_mat.shape[0]
    n = shap_mat.shape[1]
    res = []
    for i in range(0, num):
        s = np.sum(shap_mat[i, :, :])
        result = n**2 - 1 - (n-1)**2
        result = s / result
        res.append(result)
    return res


def trapmf(x, args):
    a, b, c, d = args['a'], args['b'], args['c'], args['d']
    val = 0
    if x < a or x > d:
        val = 0
    elif a < x and x <= b:
        val = (x - a) / (b - a)
    elif b <= x and x <= c:
        val = 1
    elif c<=x and x <= d:
        val = (d - x) / (d-c)


    return val



