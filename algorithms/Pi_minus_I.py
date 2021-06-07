# Pi_minus_I.py
#
# main.cpp
#
# Created by Wang Jiarui.
# Copyright © Wang Jiarui. All rights reserved.

import sys
sys.path.append('..')


import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import scipy.sparse as SP
import scipy.sparse.linalg as SPLA
import time

from algorithms.load_data import load_yale, load_multi_site_yale, load_abide_processed
from algorithms.numerical_linear_algebra import singular_value_shrinkage, singular_value_shrinkage_serial, solve_identity_plus_low_rank
from algorithms.L1_plus_Lf_square import solve_one_norm_plus_frobenius_norm_square



def low_rank_self(Xs, alpha, beta):
    """
    Assemble some sparse matrices and pass it to the solver

    :param Xs:
    :param alpha:
    :param beta:
    :return:
    """

    n, _ = Xs[0].shape
    num_sites = len(Xs)

    Xd = SP.block_diag(Xs)

    row_idx = np.arange(n)
    col_idx = np.arange(n)
    row_indices = []
    col_indices = []
    data = []
    for i in range(num_sites):
        row_indices.extend(row_idx)
        col_indices.extend(col_idx + i * n)
        data.extend(np.ones(shape=[n]))


    Is = SP.csr_matrix(
        (data, (row_indices, col_indices))
    )
    P, Z, E = solve(Xd,Xs,Is,alpha,beta)
    return P @ Xd, P, Z, E

def solve(Xd, Xs, Is, alpha, beta):
    n, _ = Xs[0].shape
    _, m = Xd.shape
    k = len(Xs)
    num_sites = [X.shape[1] for X in Xs]

    # initialize primal variables
    # P = np.ones(shape=[n, n*k])
    P = Is.toarray()
    Z = np.eye(m)
    E = np.zeros(shape=[n, m])
    J = np.eye(m)
    # Q = np.ones(shape=[n, n*k])
    Q = Is.toarray()

    # initialize dual variables
    Y1 = np.zeros_like(E)
    Y2 = np.zeros_like(J)
    Y3 = np.zeros_like(P)

    c = 1e-3
    num_iter = 0
    while True:
        num_iter += 1
        print()
        print()
        print("iteration %d, c = %.6f" % (num_iter, c))

        # optimize J
        print("solving by singular value shrinkage")
        J = singular_value_shrinkage(Z - Y2 / c, 1 / c)

        # optimize Qi
        for i in range(k):
            print("solving by singular value shrinkage")
            Pi = P[:, i * n: (i + 1) * n]
            Y3i = Y3[:, i * n: (i + 1) * n]
            Q[:, i * n: (i + 1) * n] = singular_value_shrinkage(Pi - (SP.eye(n) + Y3i / c), alpha / c)

        # optimize Ei
        F = np.eye(m) - Z
        # PX = P @ Xd
        PX = Xd.transpose().dot(P.T).transpose()
        shift = 0
        for i in range(k):
            Fi = F[:, shift:shift + num_sites[i]]
            Y1i = Y1[:, shift:shift + num_sites[i]]
            V = PX @ Fi + Y1i / c
            Ei = solve_one_norm_plus_frobenius_norm_square(2 * beta / c, V)
            E[:, shift:shift + num_sites[i]] = Ei
            shift += num_sites[i]

        # optimize Z
        # PX = P @ Xd
        Z_rhs = PX.T @ (PX - E + Y1 / c) + J + Y2 / c
        Z = solve_identity_plus_low_rank(PX.T, PX.T, Z_rhs)


        # optimize Pi
        # F = (SP.eye(m) - Z).toarray()
        F = np.eye(m) - Z
        PXFs = []
        shift = 0
        for i in range(k):
            r = P[:, i * n:(i + 1) * n] @ Xs[i] @ F[shift:shift + num_sites[i], :]
            PXFs.append(
                r
            )
            shift += num_sites[i]
        PXFs = np.stack(PXFs)
        PXF = np.sum(PXFs, axis=0)

        shift = 0
        for i in range(k):
            G = PXF - PXFs[i] - E
            Qi = Q[:, i * n:(i + 1) * n]
            Y3i = Y3[:, i * n: (i + 1) * n]
            Fi = F[shift:shift + num_sites[i], :]

            XiFi = Xs[i] @ Fi
            PiT_rhs = np.transpose(SP.eye(n) + Y3i / c + Qi - (G + Y1 / c) @ XiFi.T)
            PiT = solve_identity_plus_low_rank(XiFi, XiFi, PiT_rhs)
            Pi = PiT.T
            P[:, i * n:(i + 1) * n] = Pi

            shift += num_sites[i]


        rp1s = P @ (Xd.dot(SP.eye(m) - Z)) - E
        rp2s = J - Z
        rp3s = Is + Q - P

        max_r = max(
            np.max(np.abs(rp1s)), np.max(np.abs(rp2s)), np.max(np.abs(rp3s)),
        )
        print('equality violation', max_r)
        if max_r <= 1e-4:
            return P, Z, E

        Y1 += c * rp1s
        Y2 += c * rp2s
        Y3 += c * rp3s

        c *= 1.2
        c = min(c, 1e7)


if __name__ == '__main__':
    sites = [0, 2, 3, 4, 5, 7, 8]
    sites = [0,2]
    Xs = load_multi_site_yale(5, sites)

    tic = time.time()
    Xd_sp = low_rank_self(Xs, 0.1, 0.1)
    toc = time.time()

    print(toc - tic)
    # Xs, labels, site_names = load_abide_processed(sites=['NYU', 'Leuven', 'UCLA', 'UM', 'USM'])
    # Xd_sp = low_rank_self(Xs, 0.1, 0.1)









