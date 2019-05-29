import collections
import itertools
import warnings

import numpy as np
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg

from pygk.utils import is_not_inf, as_column, \
    matlab_min_max, Lookup, floyd_warshall, vec, invvec, nrow, as_row


def lRWkernel(graphs, lambda_):
    return labeled_random_walk_kernel(graphs, lambda_)


def RGkernel():
    raise NotImplementedError()


def WL(graphs, h):
    return weisfeiler_lehman_kernel(graphs, h)


def WLedge():
    raise NotImplementedError()


def WLspdelta():
    raise NotImplementedError()


def untilpRWkernel():
    raise NotImplementedError()


def spkernel(graphs):
    return labeled_shortest_path_kernel(graphs)


def labeled_shortest_path_kernel(graphs):
    """
    Shortest path kernel by Karsten Borgwardt
    :param graphs: An array-like KGraphs
    :return:
    """

    def _sp_offset(L_, a_, b_):
        return a_ * L_ - (a_ * (a_ - 1) / 2) + b_ - a_

    N = len(graphs)

    lookup = Lookup()
    labels = [np.array([lookup[l] for l in g.nl]) for g in graphs]
    L = lookup.total  # num labels

    # "compute Ds and the length of the maximal shortest path over the dataset"
    Ds = [floyd_warshall(g.am) for g in graphs]
    maxpath = max(max(D[is_not_inf(D)]).astype('int') for D in Ds)

    phi = dok_matrix(((maxpath + 1) * L * (L + 1) // 2, N))  # sparse
    for i in range(N):
        labels_aux = np.tile(as_column(labels[i]), (1, len(labels[i])))
        a, b = matlab_min_max(labels_aux, labels_aux.transpose())
        I = np.triu(is_not_inf(Ds[i]))  # because symmetry.
        a, b = a[I], b[I]
        features = (Ds[i][I] * (L * (L + 1) / 2) + _sp_offset(L, a, b)).astype('int').flatten()
        for feat, count in collections.Counter(features).items():
            phi[feat, i] = count

    used_features = np.squeeze(np.asarray(phi.sum(1))) != 0
    # TODO used_features can be "True"
    # if used_features is True:
    #     used_features = np.array([True])
    phi = phi[used_features, :].toarray()
    return phi.transpose() @ phi, phi


def weisfeiler_lehman_kernel(graphs, h) -> list:
    """
    h-step Weisfeiler Lehman graph kernel by Nino Shervashidze
    :param graphs:  an array-like graphs
    :param h:   step
    :return:    a list of kernel matrices, where i th index corresponds to i-step WL kernel matrix
    """
    N = len(graphs)
    Ks = [None] * (h + 1)

    # step 0
    # assignment
    lookup = Lookup()
    labels = [np.array([lookup[l] for l in g.nl]) for g in graphs]

    # feature mapping
    phi = np.zeros((lookup.total, N))
    for i in range(N):
        for l, counts in collections.Counter(labels[i]).items():
            phi[l, i] = counts

    # Kernel computation
    Ks[0] = phi.transpose() @ phi

    def long_label(ll, v, ne_v):
        # return tuple((ll[v], tuple(sorted(ll[ne_v]))))
        return tuple((ll[v], *sorted(ll[ne_v])))

    # step 1 ~ h
    for step in range(1, h + 1):  # 1 <= step <= h
        # assignment
        lookup = Lookup()
        new_labels = [np.array([lookup[long_label(labels[i], v, ne_v)] for v, ne_v in enumerate(g.al)])
                      for i, g in enumerate(graphs)]

        # feature mapping
        phi = np.zeros((lookup.total, N))
        for i in range(N):
            for l, counts in collections.Counter(new_labels[i]).items():
                phi[l, i] = counts

        # Kernel computation
        Ks[step] = Ks[step - 1] + (phi.transpose() @ phi)

        # post-loop
        labels = new_labels

    return Ks


def labeled_random_walk_kernel(gs, lambda_):
    """

    :param gs: An array-like KGraphs
    :param lambda_:
    :return:
    """
    warnings.warn('not tested')
    if lambda_ >= 1.0:
        raise ValueError('lambda must be less than 1')

    N = len(gs)
    K = np.zeros((N, N))

    lookup = Lookup()
    labels = [np.array([lookup[l] for l in g.nl]) for g in gs]
    L = lookup.total

    amss = [[None] * L * L for _ in range(N)]
    for i, g in enumerate(gs):
        for k, (l1, l2) in enumerate(itertools.product(range(L), range(L))):
            selector = as_column(labels[i] == l1) @ as_row(labels[i] == l2)
            amss[i][k] = g.am * selector

    for i in range(N):
        for j in range(i, N):
            K[j, i] = K[i, j] = _labeled_random_walk(amss[i], amss[j], lambda_, len(gs[i]) * len(gs[j]))

    return K


def _labeled_smt_filter(x, g1s, g2s, lambda_):
    yy = 0
    for am1, am2 in zip(g1s, g2s):
        yy += vec(am1 @ invvec(x, nrow(am1), nrow(am2)) @ am2)
    return x - lambda_ * yy


def _labeled_random_walk(ams1, ams2, lambda_, mn):
    # A*x=b
    A = LinearOperator((mn, mn), matvec=lambda x: _labeled_smt_filter(x, ams1, ams2, lambda_))
    b = np.ones(mn)
    x_sol, _ = cg(A, b, tol=1.0e-6, maxiter=200)
    return np.sum(x_sol)
