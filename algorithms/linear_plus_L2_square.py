# linear_plus_L2_square.py
#
# main.cpp
#
# Created by Wang Jiarui.
# Copyright © Wang Jiarui. All rights reserved.

import sys
sys.path.append('..')

import numpy as np
import scipy.sparse.linalg as SPLA
import scipy.sparse as SP

def solve_linear_plus_two_norm_square_st_matmul_inequality_log_barrier\
                (t, c:np.array, A:np.array, b:np.array, D:np.array, xstart, ATA=None, ATb=None,
                 ALPHA=0.01, BETA=0.5, EPS=1e-9):
    """
    Solve the optimization problem
        min(x). t * c'x + ||Ax - b||_2^2 - \sum \log(-Dx)
    with newton method
    """

    m1, n = A.shape
    m2, n = D.shape

    ATA = A.T @ A if ATA is None else ATA
    ATb = A.T @ b if ATb is None else ATb

    x = xstart

    def f(__x):
        Axmb = A @ __x - b
        return t * c @ __x + t * Axmb @ Axmb - np.sum(np.log(-D @ __x))

    def grad_hessian(__x):
        __g = t * c + 2 * t * ATA @ __x - 2 * t *  ATb - D.T @ (1 / (D @ __x))
        __h = 2 * t * ATA + D.T @ np.diag(1 / (D@__x)**2) @ D
        return __g, __h


    while True:
        val = f(x)
        g, h = grad_hessian(x)
        dx = np.linalg.solve(h, -g)
        # dx = np.linalg.pinv(h) @ -g

        decrement = -1 * g @ dx
        if decrement <= EPS:
            # print(decrement)
            return x

        # backtracking line search
        s = 1
        while np.min(-D @ (x + s * dx)) <= 0:
            s *= BETA

        while val + ALPHA * g @ (s * dx) <= f(x + s * dx):
            s *= BETA

        # print(s)
        x += s * dx

def solve_linear_plus_two_norm_square_st_matmul_inequality_interior_point_method(c:np.array, A:np.array, b:np.array, D:np.array, xstart, EPS=1e-4):
    """
    Solve the optimization problem
        min(x). c'x + ||Ax - b||_2^2
        s.t.    Dx <= 0
    with interior point method
    """
    m1, n = A.shape
    m2, n = D.shape

    ATA = A.T @ A
    ATb = A.T @ b

    def f(__x):
        Axmb = A @ __x - b
        return c @ __x + Axmb @ Axmb

    x = xstart

    t = 1
    while True:
        x = solve_linear_plus_two_norm_square_st_matmul_inequality_log_barrier(t, c, A, b, D, x, ATA, ATb)
        if (m2 / t) <= EPS:
            return x, 1 / (-1 * t * D @ x)  # primal variable and dual variable
        t *= 5

def solve_linear_plus_two_norm_square_st_matmul_inequality_primal_dual_interior_point_method_sparse(c:np.array, A:SP.csr_matrix, b:np.array, D:SP.csr_matrix,
                                                                                  xstart, mu=5, ALPHA=0.01, BETA=0.5, EPS=1e-6):

    """
    Solve the optimization problem
        min(x). c'x + ||Ax - b||_2^2
        s.t.    Dx <= 0
    with primal-dual interior point method
    """
    m1, n = A.shape
    m2, n = D.shape

    print("Primal-Dual Interior Point Method: %d variables, %d inequality constraints, " % (n, m2), end='')

    ATA = A.transpose().dot(A)
    ATb = A.transpose().dot(b)

    def f(__x):
        return c @ __x + np.sum((A.dot(__x) - b)**2)

    def residual(__x, __lamb, __t):
        rd = c + 2 * ATA @ __x - 2 * ATb + D.transpose().dot(__lamb) # dual residual
        rc = -1 * __lamb * (D.dot(__x)) - 1/__t
        ry = np.hstack([rd, rc])
        return rd, rc, ry

    def residual_norm(__x, __lamb, __t):
        rd, rc, ry = residual(__x, __lamb, __t)
        return np.linalg.norm(ry)


    lamb = np.ones(shape=[m2])
    x = xstart

    num_iter = 0
    while True:
        num_iter += 1
        eta = -1 * D.dot(x).dot(lamb)
        # print("Iteration %d, Objective: %.6f, Duality Gap: %.6f" % (num_iter, f(x), eta))

        t = mu * m2 / eta
        rd, rc, ry = residual(x, lamb, t)

        if (eta < EPS) and (np.linalg.norm(rd) < EPS):
            print(" %d primal-dual newton steps" % (num_iter))
            # print(f(x))
            return x, lamb

        Dx = np.array(D.dot(x))
        oneoverDx = 1 / Dx

        dx_lhs = 2 * ATA - D.transpose().dot(SP.diags(oneoverDx)).dot(SP.diags(lamb)).dot(D)
        dx_rhs = -rd - D.transpose().dot(oneoverDx * rc)
        dx = SPLA.spsolve(dx_lhs, dx_rhs)

        dlamb_lhs = Dx
        dlamb_rhs = rc - lamb * (D.dot(dx))
        dlamb = dlamb_rhs / dlamb_lhs

        # backtracking line search
        s = 1
        while np.max(D @ (x + s * dx)) >= 0:
            s *= BETA

        while np.min(lamb + s * dlamb) <= 0:
            s *= BETA

        r = np.linalg.norm(ry)
        while residual_norm(x + s * dx, lamb + s * dlamb, t) >= (1 - ALPHA * s) * r:
            s *= BETA

        lamb += s * dlamb
        x += s * dx



if __name__ == "__main__":

    from scipy.io import loadmat, savemat

    m1 = 50
    m2 = 50
    n = 50

    # c = np.random.randn(n)
    # A = np.random.randn(m1, n)
    # b = np.random.randn(m1)
    # D = np.random.randn(m2,n)
    # savemat('data.mat', {
    #     'c' : c,
    #     'A' : A,
    #     'b' : b,
    #     'D' : D,
    # })


    mdict = loadmat('data.mat')
    c = mdict['c'][0]
    b = mdict['b'][0]
    A = mdict['A']
    D = mdict['D']

    xstart = np.linalg.solve(D, np.ones([m2]) * -1)

    # x, lamb = solve_linear_plus_two_norm_square_st_matmul_inequality_primal_dual_interior_point_method(c, A, b, D, xstart)
    # x, lamb = solve_linear_plus_two_norm_square_st_matmul_inequality_interior_point_method(c, A, b, D, xstart)

    # solve_linear_plus_two_norm_square_st_matmul_inequality_primal_dual_interior_point_method_sparse(c, A, b, D, xstart)








