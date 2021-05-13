import sys
sys.path.append('..')

import numpy as np

from scipy.linalg import orth
from scipy.io import loadmat
import cv2
import matplotlib.pyplot as plt

from algorithms.load_data import load_yale


def solve_l2(w:np.array, alpha):
    """
    Solve the optimization problem:
        min. alpha * ||x||_2 + 0.5 * ||x-w||_2^2
    """
    nw = np.linalg.norm(w)
    if nw > alpha:
        return (nw - alpha) / nw * w
    else:
        return np.zeros_like(w)

def solve_l21(W:np.array, alpha):
    """
    Solve the optimization problem
        min. alpha * ||E||_{2,1} + ||E - W||_2^2
    """
    E = np.zeros_like(W)
    m, n = W.shape
    for i in range(n):
        E[:,i] = solve_l2(W[:,i], alpha)
    return E

def singular_value_shrink(X:np.array, threshold):
    U, Sig, Vt = np.linalg.svd(X)
    idx = np.where(Sig > threshold)[0].astype(int)

    Sig = Sig[idx] - threshold
    U = U[:, idx]
    Vt = Vt[idx, :]
    return U @ np.diag(Sig) @ Vt

def __solve_low_rank_representation(X:np.array, A:np.array, lamb):
    """
    Solve the nuclear-norm optimization problem
        min. ||Z||_* + lamb * ||E||_{2,1}
        s.t. X = AZ + E
    by solving the equivalent problem
        min. ||J||_* + lamb * ||E||_{2,1}
        s.t. X = AZ + E
             Z = J
    using Augmented Lagrangian Multiplier

    :param X: Of shape [d, n], d is data dimensions, n is number of data
    :param A: Of shape [d, m], a dictionary
    :param lamb:
    :return:
    """
    print(X.shape, A.shape)

    tol = 1e-8
    maxIter = 1e6
    d, n = X.shape
    d, m = A.shape
    c = 1e-3
    maxc = 1e10
    rho = 1.2

    # Initialize primal variables
    J = np.zeros(shape=[m,n])
    Z = np.zeros(shape=[m,n])
    # E = np.zeros(shape=[d,n])
    E = np.random.randn(d,n)

    # Initialize dual variables
    Y1 = np.zeros(shape=[d,n])
    Y2 = np.zeros(shape=[m,n])

    # Some data that will be used many times
    inv_atapi = np.linalg.inv(A.T @ A + np.eye(m))
    atx = A.T @ X

    iter = 0
    convergence = False
    ranks = []
    while not convergence:
        iter += 1

        # Update primal variabble J
        J = singular_value_shrink(Z + Y2 / c, threshold=1/c)

        # Update primal variable Z
        Z = inv_atapi @ (atx - A.T@E + J + (A.T @ Y1 - Y2)/c)

        # Update primal variable E
        xmaz = X - A @ Z
        E = solve_l21(xmaz + Y1 / c, alpha=lamb/c)

        # Update dual variable Y1 and Y2
        leq1 = xmaz - E # linear equality 1
        leq2 = Z - J    # linear equality 2
        Y1 += c * leq1
        Y2 += c * leq2

        ranks.append(np.linalg.matrix_rank(Z,tol=1e-3*np.linalg.norm(Z, 2)))
        # if iter % 10 == 0:
        print("Iteration %d, c:%.6f, Rank:%d, Equality 1 violation: %.5f, Equality 2 violation: %.5f"
              %
              (iter, c, np.linalg.matrix_rank(Z,tol=1e-3*np.linalg.norm(Z, 2)), np.max(np.abs(leq1)), np.max(np.abs(leq2)))
        )
        c = min(c * rho, maxc)


        if max(np.max(np.abs(leq1)), np.max(np.abs(leq2))) < tol:
            convergence = True

    return Z, E, ranks


def solve_low_rank_representation(X, lamb):
    """

    :param X: Of shape [d, n], d is data dimensions, n is number of data
    :param lamb:
    :return:
    """
    # Q = orth(X.T)
    # A = X @ Q
    Z, E, ranks = __solve_low_rank_representation(X, X, lamb)


    return X @ Z, E, ranks




if __name__ == "__main__":
    Y = load_yale(num_obj=5)
    print(Y.shape)
    shape = Y.shape
    _Y = np.reshape(Y, newshape=[shape[0] * shape[1], shape[2] * shape[3]])
    _recover, _error, ranks = solve_low_rank_representation(_Y, lamb=0.05)
    plt.plot(ranks)
    plt.xlabel('iterations')
    plt.ylabel('ranks')
    plt.show()

    # recover = np.reshape(_recover, newshape=shape)
    # error = np.reshape(_error, newshape=shape)
    # for i in range(shape[3]):
    #     for j in range(shape[2]):
    #         fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    #         ax1.imshow(Y[:,:,j,i], cmap='gray')
    #         ax1.set_title('sample')
    #         ax2.imshow(recover[:,:,j,i], cmap='gray')
    #         ax2.set_title('recover')
    #         ax3.imshow(error[:,:,j,i], cmap='gray')
    #         ax3.set_title('error')
    #         # plt.legend()
    #         plt.show()







