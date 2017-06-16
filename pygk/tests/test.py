import unittest

import numpy as np
from joblib import Parallel, delayed
from networkx import fast_gnp_random_graph
from numpy.random import randint, rand, choice
from scipy.io import loadmat

from pygk.graphlets import count_all_3_graphlets, count_all_4_graphlets, gest_kernel, \
    count_connected_5_graphlets, count_connected_4_graphlets, count_connected_3_graphlets, labeled_3_graphlet_kernel
from pygk.labeled import weisfeiler_lehman_kernel, labeled_shortest_path_kernel
from pygk.unlabeled import shortest_path_kernel, random_walk_kernel
from pygk.utils import KGraph, as_column, as_row, matlab_graphs_2_kgraphs


def rand_g():
    g = fast_gnp_random_graph(randint(5, 40), rand() * rand())
    labels = list(range(randint(2, 7)))
    for v in g:
        g.node[v]['label'] = choice(labels)
    return KGraph(g)


def normalize(K):
    Kd = np.sqrt(np.diag(K))
    K = K / (as_column(Kd) @ as_row(Kd))
    return K


def read_mutag(f_name=None):
    if f_name is None:
        f_name = '../../data/MUTAG'

    full_data = loadmat(f_name)
    return matlab_graphs_2_kgraphs(full_data['MUTAG'])


def read_enzyme(f_name=None):
    if f_name is None:
        f_name = '../../data/ENZYMES'

    full_data = loadmat(f_name)
    return matlab_graphs_2_kgraphs(full_data['ENZYMES'])


class TestEquivalentToMatlabCode(unittest.TestCase):
    def test_WL(self):
        graphs = read_mutag()
        my_Ks = weisfeiler_lehman_kernel(graphs, 3)

        full_data = loadmat('../../data/validation/WL_K3')
        nino_Ks = list(full_data['Ks'].flatten())

        tests = my_Ks[-1] == nino_Ks[-1]
        assert all(tests.flatten())

    def test_spkernel(self):
        full_data = loadmat('../../data/validation/spkernel_K')
        nino_K = full_data['K']

        graphs = read_mutag()
        my_K, phi = labeled_shortest_path_kernel(graphs)

        tests = (my_K == nino_K)
        assert all(tests.flatten())

    def test_unlabeled_spkernel(self):
        full_data = loadmat('../../data/validation/unlabeled_spkernel_K')
        nino_K = full_data['K']

        graphs = read_mutag()
        my_K, phi = shortest_path_kernel(graphs)

        tests = (my_K == nino_K)
        assert all(tests.flatten())

    def test_count_all_graphlets(self):
        graphs = read_mutag()
        full_data = loadmat('../../data/validation/mutag_cag')
        nino3 = full_data['mutag_ca3g']
        nino4 = full_data['mutag_ca4g']
        for i, g in enumerate(graphs):
            lee3 = count_all_3_graphlets(g.al)
            lee4 = count_all_4_graphlets(g.al)
            assert np.allclose(nino3[i, :], lee3) and np.allclose(nino4[i, :], lee4)

        graphs = read_enzyme()
        full_data = loadmat('../../data/validation/enz_cag')
        nino3 = full_data['enz_ca3g']
        nino4 = full_data['enz_ca4g']
        for i, g in enumerate(graphs):
            lee3 = count_all_3_graphlets(g.al)
            lee4 = count_all_4_graphlets(g.al)
            assert np.allclose(nino3[i, :], lee3) and np.allclose(nino4[i, :], lee4)

    def test_gest_kernel(self):
        np.random.seed(0)
        mutag_graphs = read_mutag()

        ninos = [loadmat('../../data/validation/gest_K3')['K3'],
                 loadmat('../../data/validation/gest_K4')['K4'],
                 loadmat('../../data/validation/gest_K5')['K5']]

        mines = [gest_kernel(mutag_graphs, 3, n_jobs=-1),
                 gest_kernel(mutag_graphs, 4, n_jobs=-1),
                 gest_kernel(mutag_graphs, 5, n_jobs=-1)]

        assert all(np.allclose(n, m) for n, m in zip(ninos, mines))

    def test_count_connected_graphlets(self):
        data = loadmat('../../data/validation/ccgs')

        # MUTAG
        mutag_graphs = read_mutag()
        m3 = data['mutag_ccg3']
        m4 = data['mutag_ccg4']
        m5 = data['mutag_ccg5']
        lee_m3 = np.zeros((188, 2))
        lee_m4 = np.zeros((188, 6))
        lee_m5 = np.zeros((188, 21))

        lee_m3s = Parallel(-1)(delayed(count_connected_3_graphlets)(g) for g in mutag_graphs)
        lee_m4s = Parallel(-1)(delayed(count_connected_4_graphlets)(g) for g in mutag_graphs)
        lee_m5s = Parallel(-1)(delayed(count_connected_5_graphlets)(g) for g in mutag_graphs)

        for i in range(len(mutag_graphs)):
            lee_m3[i, :] = lee_m3s[i]
            lee_m4[i, :] = lee_m4s[i]
            lee_m5[i, :] = lee_m5s[i]

        assert np.allclose(m3, lee_m3) and np.allclose(m4, lee_m4) and np.allclose(m5, lee_m5)

        # ENZYMES
        enz_graphs = read_enzyme()
        e3 = data['enz_ccg3']
        e4 = data['enz_ccg4']
        e5 = data['enz_ccg5']
        lee_e3 = np.zeros((600, 2))
        lee_e4 = np.zeros((600, 6))
        lee_e5 = np.zeros((600, 21))

        lee_e3s = Parallel(-1)(delayed(count_connected_3_graphlets)(g) for g in enz_graphs)
        lee_e4s = Parallel(-1)(delayed(count_connected_4_graphlets)(g) for g in enz_graphs)
        lee_e5s = Parallel(-1)(delayed(count_connected_5_graphlets)(g) for g in enz_graphs)

        for i in range(len(enz_graphs)):
            lee_e3[i, :] = lee_e3s[i]
            lee_e4[i, :] = lee_e4s[i]
            lee_e5[i, :] = lee_e5s[i]

        assert np.allclose(e3, lee_e3) and np.allclose(e4, lee_e4) and np.allclose(e5, lee_e5)

    def test_l3g(self):
        graphs = read_mutag()
        k = loadmat('../../data/validation/mutag_l3g')['mutag_l3g']
        assert np.allclose(k, labeled_3_graphlet_kernel(graphs))

    def test_random_walk(self):
        m = read_mutag()
        random_walk_kernel(m, 0.5, -1)


if __name__ == '__main__':
    unittest.main()
