# numerical_linear_algebra.py
#
# main.cpp
#
# Created by Wang Jiarui.
# Copyright © Wang Jiarui. All rights reserved.

import numpy as np
import scipy.sparse as SP
import os



def singular_value_shrinkage(A, alpha):
    U, lamb, VT = np.linalg.svd(A)
    idx = np.where(lamb > alpha)[0]
    print(idx)
    if len(idx) == 0:
        return np.zeros_like(A)
    elif len(idx) == 1:
        return np.outer(U[:,0], (lamb[0] - alpha) * VT[0,:])
    else:
        diags = np.maximum(lamb[idx] - alpha,0)
        return U[:,idx] @ SP.diags(diags).dot(VT[idx,:])

def singular_value_shrinkage_serial(para):
    return singular_value_shrinkage(para[0],para[1])




def solve_identity_plus_low_rank(P:np.array, Q:np.array, F:np.array, method=None):
    """
    Solving the diagonal-plus-low-rank system (I + PQ^T)X = F
    by block elimination.
    :param P: \in R^{n,m}
    :param Q: \in R^{n,m}
    :param F: \in R^{n,k}
    """

    n,m = P.shape
    n,k = F.shape

    # just a rough approximation of computation complexity
    # O(forming I + PQ^T) + O(solving by cholesky factorization)
    # c1 = n**3 + (n**2)*k + n**2 * m
    c1 = m * (n**2) + n**2 + n**3 + (n**2) * k

    # O(block elimination) + O(solving subsystem)
    c2 = (m**2)*n + m**2 + m*n*k + (m**2)*k + m**3 + n*m*k + n*k

    if c1 > c2 or method == 'block': # solve by block elimination
        print('solving linear system by block elimination',)
        y_lhs = -1 * Q.T @ P - np.eye(m)
        y_rhs = -1 * Q.T @ F
        y = np.linalg.solve(y_lhs, y_rhs)
        X = F - P @ y
    else: # solve directly
        print('solving linear system directly')
        X = np.linalg.solve(np.eye(n) + P @ Q.T, F)
    return X

if __name__ == '__main__':
    A = np.random.randn(100,100)
    singular_value_shrinkage(A, 0.1)












