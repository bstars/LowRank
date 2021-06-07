import sys
sys.path.append('..')

import numpy as np
import scipy.linalg as scla
import matplotlib.pyplot as plt

from algorithms.numerical_linear_algebra import singular_value_shrinkage, solve_identity_plus_low_rank
from algorithms.L1_plus_Lf_square import solve_one_norm_plus_frobenius_norm_square
from algorithms.load_data import load_yale, load_multi_site_yale

def solve_multi_site_low_rank_representation(Xs: list, Xt: np.array, alpha, beta, epsilon=1e-8, cmax=1e7):
    """
    Solving the multi-site-low-rank-representation problem
        min.    ||P||_* + \sum_i ( ||Zi||_* + alpha * ||Esi||_1 + beta * ||Epi||_1 )
        (P, Pi, Zi, Epi, Esi)

        s.t.    Pi Xsi = P Xt Zi + Esi
                Pi = P + Epi

    by solving the equivalent problem
        min.    ||P||_* + \sum_i ( ||Zi||_* + alpha * ||Esi||_1 + beta * ||Epi||_1 )
        (P, Pi, Zi, Epi, Esi, J, Fi)

        s.t.    Zi = Fi                     dual variable Y1i
                Pi Xsi = P Xt Zi + Esi      dual variable Y2i
                Pi = P + Epi                dual variable Y3i
                P = J                       dual variable Y4
    with Augmented Lagrangian Multiplier and Alternating Optimization

    :param Xs:  A list contains data from k site,
                each element in Xs is a np.array with shape [n,m_i]
                where m_i is number of samples in this site
    :param Xt: A np.array of shape [n, m_t], samples from target site
    :param alpha: Hyperparameter associated with error in samples
    :param beta: Hyperparameter associated with error in projection matrices
    :return:
    """

    k = len(Xs)
    n, nk = Xt.shape

    # initialize primal variables
    P = np.eye(n)  # public projection matrix
    Ps = [np.zeros(shape=[n, n]) for i in range(k)]
    J = np.eye(n)  # public projection matrix slack variable
    Zs = [np.zeros(shape=[nk, X.shape[1]]) for X in Xs]  # coefficient matrices
    Fs = [np.zeros(shape=[nk, X.shape[1]]) for X in Xs]  # coefficient matrices slack variable
    ESs = [np.zeros_like(X) for X in Xs]  # error term in data
    EPs = [np.zeros(shape=[n, n]) for i in range(k)]  # error term in projection matrices

    # initialize dual variables
    Y1s = [np.zeros_like(Z) for Z in Zs]  # dual variables a.w. equality constraint   P=J
    Y2s = [np.zeros_like(E) for E in ESs]  # dual variables a.w. equality constraints  PiXi = PXtZi + Esi
    Y3s = [np.zeros_like(E) for E in EPs]  # dual variables a.w. equality constraints  Pi = P + Epi
    Y4 = np.zeros_like(P)  # dual variable a.w. equality constraint    P = J

    c = 1e-3
    num_iter = 0

    def f(__Ps, __Zs, __ESs, __EPs, __J, __Fs):
        val = 0
        for i in range(k):
            lamb = np.linalg.svd(__Fs[i], compute_uv=False)
            val += np.sum(lamb)
            val += alpha * np.sum(np.abs(__ESs[i]))
            val += beta * np.sum(np.abs(__EPs[i]))
        lamb = np.linalg.svd(__J, compute_uv=False)
        val += np.sum(lamb)
        return val

    while True:
        num_iter += 1
        print('Iteration %d, c = %.5f' % (num_iter, c))

        print('\t Solving for J : Singular value shrinkage')
        J = singular_value_shrinkage(P + 1 / c * Y4, 1 / c)

        print('\t Solving for Fs : Singular value shrinkage')
        for i in range(k):  # update slack variable Fi
            Fs[i] = singular_value_shrinkage(Zs[i] + 1 / c * Y1s[i], 1 / c)

        print('\t Solving for ESs : ')
        for i in range(k):  # update primal variable Esi
            print('\t\t', end='')
            V = Ps[i] @ Xs[i] - P @ Xt @ Zs[i] + Y2s[i] / c
            ESs[i] = solve_one_norm_plus_frobenius_norm_square(2 * alpha / c, V)

        print('\t Solving for EPs : ')

        for i in range(k):  # update primal variable Epi
            print('\t\t', end='')
            V = Ps[i] - P + Y3s[i] / c
            EPs[i] = solve_one_norm_plus_frobenius_norm_square(2 * beta / c, V)

        print('\t Solving for Zs : ')
        for i in range(k):  # update primal variable Zi
            print('\t\t', end='')
            Zi_rhs = (P @ Xt).T @ (Ps[i] @ Xs[i] - ESs[i] + Y2s[i] / c) + Fs[i] - Y1s[i] / c
            temp = np.transpose(P @ Xt)
            Zs[i] = solve_identity_plus_low_rank(temp, temp, Zi_rhs)

        print('\t Solving for Ps : ')
        for i in range(k):  # update primal variable Pi
            print('\t\t', end='')
            PiT_rhs = np.transpose(
                (P @ Xt @ Zs[i] + ESs[i] - Y2s[i] / c) @ Xs[i].T + P + EPs[i] - Y3s[i] / c
            )
            PiT = solve_identity_plus_low_rank(Xs[i], Xs[i], PiT_rhs)
            Ps[i] = PiT.T

        # update primal variable P
        P_lhs = np.eye(n) * (k + 1)
        P_rhs = -1 / c * Y4 + J

        for i in range(k):
            XtZi = Xt @ Zs[i]
            P_lhs += XtZi @ XtZi.T
            # P_rhs += Ps[i] @ Xs[i] @ XtZi.T - ESs[i] @ XtZi.T + Ps[i] - EPs[i] + 1/c * Y2s[i] @ XtZi.T + Y3s[i]
            P_rhs += (Y2s[i] @ (XtZi).T + Y3s[i]) / c + (Ps[i] @ Xs[i] - ESs[i]) @ XtZi.T + Ps[i] - EPs[i]

        P = np.linalg.solve(P_lhs, P_rhs)
        # P, _, _ = np.linalg.svd(P)
        # P = scla.orth(P)

        # primal residuals
        rp1s = [Zs[i] - Fs[i] for i in range(k)]
        rp2s = [Ps[i] @ Xs[i] - P @ Xt @ Zs[i] - ESs[i] for i in range(k)]
        rp3s = [Ps[i] - P - EPs[i] for i in range(k)]
        rp4 = P - J

        # exit if convergence
        max_error = 0
        for i in range(k):
            max_rp1 = np.max(np.abs(rp1s[i]))
            max_rp2 = np.max(np.abs(rp2s[i]))
            max_rp3 = np.max(np.abs(rp3s[i]))
            max_error = max(max_rp1, max_rp2, max_rp3, max_error)

        max_rp4 = np.max(np.abs(rp4))
        max_error = max(max_error, max_rp4)

        print('\t Objective: %.5f, max primal residual: %.5f' % (f(Ps, Zs, ESs, EPs, J, Fs), max_error))

        if max_error <= epsilon:
            return P, Ps, Zs, ESs, EPs

        # Update dual variables
        for i in range(k):
            Y1s[i] += c * rp1s[i]
            Y2s[i] += c * rp2s[i]
            Y3s[i] += c * rp3s[i]
        Y4 += c * rp4

        c *= 2
        c = min(cmax, c)


def recover_low_rank(Xs, Xt, P, Ps):
    X_sites = []
    num_sites = len(Xs)
    Xt = P @ Xt

    for i in range(num_sites):
        X_sites.append(
            Ps[i] @ Xs[i]
        )

    return X_sites, Xt








if __name__ == '__main__':
    num_obj = 10
    sites = [0, 2, 3, 4, 5, 7, 8]
    sites = [0,2,3,4]
    Xs = load_multi_site_yale(num_obj, sites)

    # Xt = Xs[0]
    # Xs = Xs[1:]

    # P, Ps, Zs, ESs, EPs = solve_multi_site_low_rank_representation(Xs, Xt, 1, 1)
    # Xs, Xt = recover_low_rank(Xs, Xt, P, Ps)

    num_sites =len(Xs)
    fig, axes = plt.subplots(num_obj, num_sites)

    for i in range(num_obj):
        for j in range(num_sites):
            axes[i,j].imshow(np.reshape(Xs[j][:,i], [48,42]), cmap='gray')
    plt.show()









