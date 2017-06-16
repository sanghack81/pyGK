import collections

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse.linalg import LinearOperator, cg

from pygk.utils import floyd_warshall, is_not_inf, nrow, vec, invvec


def RWkernel(graphs, lambda_):
    return random_walk_kernel(graphs, lambda_)


def SPkernel(graphs):
    return shortest_path_kernel(graphs)


def random_walk_kernel(graphs, lambda_, n_jobs=1):
    if lambda_ >= 1.0:
        raise ValueError('lambda must be less than 1')
    N = len(graphs)
    K = np.zeros((N, N))
    with Parallel(n_jobs=n_jobs) as parallel:
        result = parallel(delayed(_random_walk)(graphs[i], graphs[j], lambda_)
                          for i in range(N) for j in range(i, N))
        ir = iter(result)
        for i in range(N):
            for j in range(i, N):
                K[i, j] = K[j, i] = next(ir)
                # for j in range(i, N):
                #     K[j, i] = K[i, j] = _random_walk(graphs[i], graphs[j], lambda_)

    return K


def shortest_path_kernel(graphs):
    """
    a Shortest path kernel by Karsten Borgwardt (ICDM 2005)
    :param graphs: an array-like N graphs
    :return: a pair of kernel matrix N x N and feature as a k X N matrix
    """
    # "compute Ds and the length of the maximal shortest path over the dataset"
    Ds = [floyd_warshall(g.am) for g in graphs]

    len_max_shortest_path = max(max(D[is_not_inf(D)]).astype('int') for D in Ds)

    phi = np.zeros((len_max_shortest_path + 1, len(graphs)))
    for i, D in enumerate(Ds):
        features = D[np.triu(is_not_inf(D))].astype('int').flatten()  # features
        for feat, count in collections.Counter(features).items():
            phi[feat, i] = count

    return phi.transpose() @ phi, phi


def _smt_filter(x, g1, g2, lambda_):
    # x.shape == (MN,) or (MN, 1)
    m, n = nrow(g1.am), nrow(g2.am)
    yy = vec(g1.am @ invvec(x, m, n) @ g2.am)
    return x - lambda_ * yy


def _random_walk(g1, g2, lambda_):
    # A*x=b
    mn = len(g1) * len(g2)
    A = LinearOperator((mn, mn), matvec=lambda x: _smt_filter(x, g1, g2, lambda_))
    b = np.ones(mn)
    x_sol, _ = cg(A, b, tol=1.0e-6, maxiter=500)
    return np.sum(x_sol)
