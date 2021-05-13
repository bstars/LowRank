# L1_plus_Lf_square.py
#
# main.cpp
#
# Created by Wang Jiarui.
# Copyright © Wang Jiarui. All rights reserved.



import sys
sys.path.append('..')

import numpy as np
import scipy.sparse as SP


from algorithms.linear_plus_L2_square import solve_linear_plus_two_norm_square_st_matmul_inequality_primal_dual_interior_point_method_sparse

def solve_one_norm_plus_frobenius_norm_square(k, V):
    """
    Solve the optimization problem

        min(X). k * ||X||_1 + ||X - V||_2^2

    where X and V are 2 mxn matrices
    """
    m,n = V.shape
    v = np.reshape(V,[m*n])
    x = solve_one_norm_plus_two_norm_square(k, v)
    X = np.reshape(x, [m,n])
    return X

def solve_one_norm_plus_two_norm_square(k, v):
    """
    Solve the optimization problem

        min(x). k * ||x||_1 + ||x-v||_2^2

    by solving the equivalent problem

        min(x,y) k * 1'y + ||x-v||_2^2
        s.t.  -y <= x <= y

    which can be reformulated to the problem

                             [x]           [x]
        min.([x y]) k * [0 1][y] + || [I 0][y] - v ||_2^2

        s.t.    [ -I  -I  ] [x]  <= [0]
                [  I  -I  ] [y]     [0]

    a linear-plus-two-norm-square problem subject to a matmul inequality constraint.
    """
    n = len(v)
    c = np.concatenate([np.zeros([n]), np.ones([n])]) * k

    row_idx = [i for i in range(n)]
    col_idx = [i for i in range(n)]
    data = [1 for i in range(n)]

    A = SP.csr_matrix((data, (row_idx, col_idx)), shape=[n,2*n])

    row_idx = []
    col_idx = []
    data = []
    # upper left
    row_idx.extend([i for i in range(n)])
    col_idx.extend([i for i in range(n)])
    data.extend([-1 for i in range(n)])

    # upper right
    row_idx.extend([i for i in range(n)])
    col_idx.extend([i for i in range(n, 2*n)])
    data.extend([-1 for i in range(n)])

    # lower left
    row_idx.extend([i for i in range(n, 2*n)])
    col_idx.extend([i for i in range(n)])
    data.extend([1 for i in range(n)])

    # lower right
    row_idx.extend([i for i in range(n, 2*n)])
    col_idx.extend([i for i in range(n, 2 * n)])
    data.extend([-1 for i in range(n)])

    D = SP.csr_matrix((data, (row_idx, col_idx)), shape=[2*n, 2*n])

    xystart = np.zeros(shape=[2*n])
    xystart[n:2*n] = 1

    xy, lamb = solve_linear_plus_two_norm_square_st_matmul_inequality_primal_dual_interior_point_method_sparse(c, A, v, D, xystart)
    return xy[:n]

if __name__ == "__main__":
    from scipy.io import loadmat,savemat

    v = np.random.randn(20, 10)
    savemat('dataf.mat', {'v':v})


    mdict = loadmat('dataf.mat')
    v = mdict['v']

    X = solve_one_norm_plus_frobenius_norm_square(2, v)
    print(X)

