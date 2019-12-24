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


def get_shapley_protoform_feat(shap_mat):
    f = np.amax(np.amax(shap_mat, 2), 1)
    return f