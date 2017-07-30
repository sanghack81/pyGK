import functools
from collections import defaultdict

import itertools
import math
import numpy as np
from joblib import Parallel, delayed
from numpy import zeros, array, ix_
from numpy.random import choice
from scipy.special import comb

from pygk.utils import nrow, KGraph, Lookup


def all_graphlets_kernel(graphs, k=4):
    phi = np.array([count_all_graphlets(g.al, k) for i, g in enumerate(graphs)])
    return phi @ phi.transpose()


def _card_3inter(L1: set, L2: set, L3: set):
    """count cardinalities of subsets
       N.B. the 0th element is a dummy to match MATLAB indices
    """
    return array([len(x) for x in [set(),  # 0
                                   L1 - (L2 | L3),  # 1
                                   L2 - (L1 | L3),  # 2
                                   L3 - (L1 | L2),  # 3
                                   (L1 & L2) - L3,  # 4
                                   (L1 & L3) - L2,  # 5
                                   (L2 & L3) - L1,  # 6
                                   L1 & L2 & L3]])  # 7


def _first_index(xs, v):
    for i, x in enumerate(xs):
        if x == v:
            return i
    return -1


def count_all_graphlets(ne, k, with_dummy=False):
    if k == 3:
        return count_all_3_graphlets(ne, with_dummy)
    elif k == 4:
        return count_all_4_graphlets(ne, with_dummy)
    else:
        raise NotImplementedError('')


# neighbors is a list of adjacency "set".
def count_all_3_graphlets(ne, with_dummy=False):
    ne = [set(ne_v) for ne_v in ne]
    n = len(ne)
    count = zeros(4, dtype='int')
    w_div = array((6, 4, 2))
    for v1 in range(n):
        for v2 in ne[v1]:
            n_union, n_inter = len(ne[v1] | ne[v2]), len(ne[v1] & ne[v2])

            count[0] += n_inter
            count[1] += n_union - n_inter - 2
            count[2] += n - n_union

    count[0:3] = count[0:3] / w_div
    count[3] = comb(n, 3) - sum(count[0:3])
    return count if not with_dummy else [0] + count


def count_all_4_graphlets(ne, with_dummy=False):
    # ne is a list of neighbors.
    # ne[x] corresponds to a set of x's neighbors
    ne = [set(ne_v) for ne_v in ne]

    dummy = 1

    # N.B. the 0th element is a dummy to match MATLAB indices
    # N.B. the last element is non-zero to avoid divide-by-zero
    w_div = array([dummy, 12, 10, 8, 6, 8, 6, 6, 4, 4, 2, dummy])
    n = len(ne)
    # N.B. the 0th element is a dummy to match MATLAB indices
    count = zeros(1 + 11)
    m = sum(len(ne_v) for ne_v in ne) / 2

    for v1 in range(n):
        for v2 in sorted(ne[v1]):  # sorted is for debugging together with MATLAB.
            K = 0
            inner_cnt = zeros(1 + 11)
            inter, diff1, diff2 = ne[v1] & ne[v2], ne[v1] - ne[v2], ne[v2] - ne[v1]

            for v3 in sorted(inter):
                card = _card_3inter(ne[v1], ne[v2], ne[v3])
                inner_cnt[1] += card[7] / 2
                inner_cnt[2] += (card[4] - 1) / 2
                inner_cnt[2] += (card[5] - 1) / 2
                inner_cnt[2] += (card[6] - 1) / 2
                inner_cnt[3] += card[1] / 2
                inner_cnt[3] += card[2] / 2
                inner_cnt[3] += card[3]
                inner_cnt[7] += n - sum(card[1:])
                K += (card[7] + (card[5] - 1) + (card[6] - 1)) / 2 + card[3]
            for v3 in sorted(diff1 - {v2}):
                card = _card_3inter(ne[v1], ne[v2], ne[v3])
                inner_cnt[2] += card[7] / 2
                inner_cnt[3] += card[4] / 2
                inner_cnt[3] += card[5] / 2
                inner_cnt[5] += (card[6] - 1) / 2
                inner_cnt[4] += (card[1] - 2) / 2
                inner_cnt[6] += card[2] / 2
                inner_cnt[6] += card[3]
                inner_cnt[8] += n - sum(card[1:])
                K += (card[7] + card[5] + (card[6] - 1)) / 2 + card[3]
            for v3 in sorted(diff2 - {v1}):
                card = _card_3inter(ne[v1], ne[v2], ne[v3])
                inner_cnt[2] += card[7] / 2
                inner_cnt[3] += card[4] / 2
                inner_cnt[5] += (card[5] - 1) / 2
                inner_cnt[3] += card[6] / 2
                inner_cnt[6] += card[1] / 2
                inner_cnt[4] += (card[2] - 2) / 2
                inner_cnt[6] += card[3]
                inner_cnt[8] += n - sum(card[1:])
                K += (card[7] + (card[5] - 1) + card[6]) / 2 + card[3]

            n_union = len(ne[v1] | ne[v2])
            inner_cnt[9] += m + 1 - len(ne[v1]) - len(ne[v2]) - K
            inner_cnt[10] += (n - n_union) * (n - n_union - 1) / 2 - \
                             (m + 1 - len(ne[v1]) - len(ne[v2]) - K)

            count += inner_cnt / w_div

    count[11] = comb(n, 4) - sum(count[1:11])
    return count if with_dummy else count[1:]


@functools.lru_cache(3)
def _permutation_matrix(n):
    # graphlet_type equivalence as a matrix
    n_upper = n * (n - 1) // 2
    P = zeros((2 ** n_upper, 2 ** n_upper), dtype='int')
    for upper in itertools.product(range(2), repeat=n_upper):
        am = _fill_adj_matrix(n, upper)

        for perm in itertools.permutations(range(n)):
            P[_graphlet_type(am), _graphlet_type(am[ix_(perm, perm)])] = 1

    return P


# example
# >>> n = 5
# >>> upper = a,b,c,d,e,f,g,h,i,j
# >>> am = _fill_adj_matrix(n, upper)
# am = array([[0, a, b, c, d],
#                [a, 0, e, f, g],
#                [b, e, 0, h, i],
#                [c, f, h, 0, j],
#                [d, g, i, j, 0]])
def _fill_adj_matrix(n, upper):
    am = zeros((n, n), dtype='int')
    offset = 0
    for i in range(n - 1):
        am[i, i + 1:] = upper[offset:(offset + n - 1 - i)]
        offset += n - 1 - i
    am += am.transpose()  # make it symmetric
    return am


def _graphlet_type(am) -> int:
    upper = np.concatenate([am[i, i + 1:] for i in range(nrow(am) - 1)])
    factor = 2 ** array(range(len(upper)), dtype='int')
    return int(sum(factor * upper))


# Following is the original comment by Karsten Borgwardt
#
# % delta = confidence level (typically 0.05 or 0.1)
# % epsilon = precision level (typically 0.05 or 1)
# % a = number of isomorphism classes of graphlets
# %
# % Karsten Borgwardt
# % 4/11/2008
def sample_size(delta, epsilon, a):
    if delta < 0 or delta > 1:
        raise ValueError('delta must be in [0,1].')
    if epsilon < 0 or epsilon > 1:
        raise ValueError('epsilon must be in [0,1].')
    return 2 * (a * math.log(2) + math.log(1 / delta)) / (epsilon ** 2)


def gest_kernel(graphs, k, num_samples=-1, n_jobs=1):
    """Graphlet-based Graph Kernel based on sampling

    :param graphs: an array-like sequence of N graphs
    :param k: the size of graphlets to be sampled. (3 <= k <= 5)
    :param num_samples: size of samples, if -1, all possible samples are counted
    :param n_jobs: number of CPUs to use
    :return: a kernel matrix (N x N)
    """
    if num_samples < -1 or round(num_samples) != num_samples:
        raise ValueError('The number of samples must be -1 or non-negative integer.')
    if k < 3 or k > 6:
        raise ValueError('{}-graphlet is not supported.'.format(k))
    FEAT_LEN = {3: 8, 4: 64, 5: 1024}

    seeds = iter([np.random.randint(np.iinfo(np.int32).max) for _ in range(len(graphs))])
    outs = Parallel(n_jobs)(delayed(_inner_gest_kernel)(g, k, num_samples, next(seeds)) for g in graphs)

    phi = zeros((len(graphs), FEAT_LEN[k]))
    for i, phi_i in enumerate(outs):
        if phi_i is not None:
            phi[i, :] = phi_i

    return phi @ _permutation_matrix(k) @ phi.transpose()


def _inner_gest_kernel(g, k, num_samples, seed):
    np.random.seed(seed)
    FEAT_LEN = {3: 8, 4: 64, 5: 1024}
    m = len(g)
    if m < k:
        return None

    phi = np.zeros(FEAT_LEN[k])
    # by sample
    if num_samples != -1:
        adder = 1.0 / num_samples if num_samples != 0 else 0
        for idx in (choice(m, k, replace=False) for _ in range(num_samples)):
            phi[_graphlet_type(g.am[ix_(idx, idx)])] += adder
    else:
        adder = 1.0 / comb(m, k)
        for idx in itertools.combinations(range(m), k):
            phi[_graphlet_type(g.am[ix_(idx, idx)])] += adder
    return phi


def count_connected_3_graphlets(graph: KGraph):
    am = graph.am
    ne = [set(ne_v) for ne_v in graph.al]
    w = np.array([1 / 2, 1 / 6])
    n = nrow(am)
    count = zeros(2, dtype='float')
    # i -- j -- k
    for i in range(n):
        for j in ne[i]:
            for k in ne[j] - {i}:
                if am[i, k] == 1:
                    count[1] += w[1]
                else:
                    count[0] += w[0]

    return count


def count_connected_4_graphlets(graph: KGraph):
    am = graph.am
    ne = [set(ne_v) for ne_v in graph.al]
    dummy = 1
    # MATLAB compatibility
    w = np.array([dummy, 1 / 24, 1 / 12, 1 / 4, 0, 1 / 8, 1 / 2])
    n = nrow(am)
    count = zeros(dummy + 6, dtype='float')
    # i -- j -- k -- l
    for i in range(n):
        for j in ne[i]:
            for k in ne[j] - {i}:
                for l in ne[k] - {i, j}:
                    aux = am[i, k] + am[i, l] + am[j, l]
                    if aux == 3:  # a clique of i,j,k,l
                        count[1] += w[1]
                    elif aux == 2:
                        count[2] += w[2]
                    elif aux == 1:
                        if am[i, l] == 1:
                            count[5] += w[5]
                        else:
                            count[3] += w[3]
                    else:  # i--j--k--l
                        count[6] += w[6]

        for j, k, l in itertools.combinations(ne[i], 3):
            if am[j, k] == 0 and am[j, l] == 0 and am[k, l] == 0:
                count[4] += 1

    return count[1:]


def count_connected_5_graphlets(graph: KGraph):
    am = graph.am
    ne = [set(ne_v) for ne_v in graph.al]
    dummy = 1
    # MATLAB compatibility
    w = np.array(
        [dummy, 1 / 120, 1 / 72, 1 / 48, 1 / 36, 1 / 28, 1 / 20, 1 / 14, 1 / 10, 1 / 12, 1 / 8, 1 / 8, 1 / 4, 1 / 2,
         1 / 12, 1 / 12, 1 / 4, 1 / 4, 1 / 2, 0, 0, 0])
    n = nrow(am)
    # MATLAB compatibility
    count = zeros(dummy + 21, dtype='float')

    for i in range(n):
        for j in ne[i]:
            for k in ne[j] - {i}:
                for l in ne[k] - {i, j}:
                    for m in ne[l] - {i, j, k}:

                        ijklm = (i, j, k, l, m)
                        am_ijklm = am[ix_(ijklm, ijklm)]
                        aux2 = sum(am_ijklm, 1 - dummy)
                        aux1 = sorted(aux2)  # MATLAB compatibility
                        aux = am[i, k] + am[i, l] + am[i, m] + am[j, l] + am[j, m] + am[k, m]

                        if aux == 6:
                            count[1] += w[1]
                        elif aux == 5:
                            count[2] += w[2]
                        elif aux == 4:
                            if min(aux2) == 2:
                                count[4] += w[4]
                            else:
                                count[3] += w[3]
                        elif aux == 3:
                            if aux1[0] == 1:
                                count[9] += w[9]
                            elif aux1[1] == 3:
                                count[5] += w[5]
                            elif aux1[2] == 2:
                                count[14] += w[14]
                            else:
                                count[6] += w[6]
                        elif aux == 2:
                            if aux1[0] == 1:
                                if aux1[2] == 2:
                                    count[16] += w[16]
                                else:
                                    count[10] += w[10]
                            elif aux1[3] == 2:
                                count[11] += w[11]
                            else:  # Three 2s, Two 3s
                                ind = np.where(aux2 == 3)[0]
                                if am[ijklm[ind[0]], ijklm[ind[1]]] == 1:
                                    count[7] += w[7]
                                else:
                                    count[15] += w[15]
                        elif aux == 1:
                            if aux1[0] == 2:
                                count[8] += w[8]
                            elif aux1[1] == 1:
                                count[18] += w[18]
                            else:  # One 3, One 1
                                # ind = np.concatenate((np.where(aux2 == 3)[0], np.where(aux2 == 1)[0]))
                                if am[ijklm[_first_index(aux2, 3)], ijklm[_first_index(aux2, 1)]] == 1:
                                    count[17] += w[17]
                                else:
                                    count[12] += w[12]
                        else:
                            count[13] += w[13]

        for j in ne[i]:
            for k in filter(lambda k_: am[i, k_] == 0, ne[j] - {i}):
                for l in filter(lambda l_: all(am[[i, j], l_] == 0), ne[k] - {i, j}):
                    for _ in filter(lambda m_: all(am[[i, j, l], m_] == 0), ne[k] - {i, j, l}):
                        count[20] += 0.5

        for j, k, l, m in itertools.combinations(ne[i], 4):
            aux = am[j, k] + am[j, l] + am[j, m] + am[k, l] + am[k, m] + am[l, m]
            if aux == 1:
                count[19] += 1
            elif aux == 0:
                count[21] += 1

    count_in_float = count[1:]
    count_in_int = np.around(count_in_float).astype('int')
    return count_in_int


def labeled_3_graphlet_kernel(graphs):
    """Labeled connected 3-graphlet Kernel.

    :param graphs: an array-like KGraphs
    :return: a kernel matrix in np.array format
    """
    # a unique integer is assigned to every label
    lookup = Lookup()
    labels = [np.array([lookup[l] for l in g.nl]) for g in graphs]

    lookup = Lookup()
    phi_dict = defaultdict(lambda: defaultdict(lambda: 0))
    for gi, g in enumerate(graphs):
        am = g.am
        ne = [set(ne_v) for ne_v in g.al]
        n = nrow(am)
        # i -- j -- k
        for i in range(n):
            ni = labels[gi][i]
            for j in ne[i]:
                nj = labels[gi][j]
                for k in ne[j] - {i}:
                    nk = labels[gi][k]
                    feat = lookup[(am[i, k], min(ni, nk), nj, max(ni, nk))]
                    if am[i, k] == 0:  # i--j--k, a chain of i,j,k
                        phi_dict[gi][feat] += 1 / 2
                    else:  # (clique of i,j,k)
                        phi_dict[gi][feat] += 1 / 6

    # dict to
    phi = np.zeros((len(graphs), lookup.total))
    for gi, feats in phi_dict.items():
        for fk, fv in feats.items():
            phi[gi, fk] = fv

    return phi @ phi.transpose()
