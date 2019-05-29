import networkx as nx
import numpy as np
from scipy.sparse import spmatrix


class Lookup:
    def __init__(self, data=None):
        self.count = 0
        self.table = {}

        # preloaded data
        if data is not None:
            for x in data:
                self[x]

    def __getitem__(self, x):
        if x in self.table:
            return self.table[x]
        else:
            self.table[x] = v = self.count
            self.count += 1
            return v

    @property
    def total(self):
        return self.count

    def clear(self):
        self.count = 0
        self.table = {}


def as_column(x: np.array):
    assert x.ndim == 1
    return np.reshape(x, (len(x), 1))


def as_row(x: np.array):
    assert x.ndim == 1
    return np.reshape(x, (1, len(x)))


def nnz(X: np.array):
    return (X != 0.0).sum()


def matlab_min_max(X: np.array, Y: np.array):
    assert X.shape == Y.shape
    min_xy, max_xy = X.copy(), X.copy()

    small_y = Y < X
    min_xy[small_y] = Y[small_y]

    large_y = np.invert(small_y)
    max_xy[large_y] = Y[large_y]

    return min_xy, max_xy


def matlab_min(X: np.array, Y: np.array):
    assert X.shape == Y.shape
    Z = X.copy()
    indices = Y < X
    Z[indices] = Y[indices]
    return Z


def matlab_max(X: np.array, Y: np.array):
    assert X.shape == Y.shape
    Z = X.copy()
    indices = Y > X
    Z[indices] = Y[indices]
    return Z


def is_not_inf(x: np.array):
    return np.invert(np.isinf(x))


def nrow(x: np.array):
    assert np.ndim(x) == 2
    return x.shape[0]


def ncol(x: np.array):
    assert np.ndim(x) == 2
    return x.shape[1]


def column_of(x: np.array, c):
    return as_column(x[:, c])


def row_of(x: np.array, r):
    return as_row(x[r, :])


def repmat(x, r, c):
    assert np.ndim(x) == 2
    return np.tile(x, (r, c))


def floyd_warshall(am: np.array, sym: bool = True, w: np.array = None):
    n = am.shape[0]  # num row

    D = (am if w is None else (am * w)).astype('float')
    D[am == 0] = float("inf")
    np.fill_diagonal(D, 0.0)

    if sym:
        for k in range(n):
            D_aux = repmat(column_of(D, k), 1, n)
            sum_dist = D_aux + D_aux.transpose()
            D[sum_dist < D] = sum_dist[sum_dist < D]  # D = min(D,sum_dist)
    else:
        for k in range(n):
            D_aux_1 = repmat(column_of(D, k), 1, n)
            D_aux_2 = repmat(row_of(D, k), n, 1)
            sum_dist = D_aux_1 + D_aux_2
            D[sum_dist < D] = sum_dist[sum_dist < D]  # D = min(D,sum_dist)
    return D


class KGraph:
    """
    Graph representation used in pyGK project.
    """

    def __init__(self, graph: nx.Graph, label_key='label', special_labels=None):
        if special_labels is None:
            special_labels = {}

        n = self.num_nodes = graph.number_of_nodes()
        # fix order!
        nodes = list(graph.nodes())
        node_id = {v: i for i, v in enumerate(nodes)}

        self.am = np.zeros((n, n))  # adjacency matrix, should it be integer?
        self.nl = np.zeros(n, 'object')  # a vector
        self.al = [None] * n  # adjacency list

        for nv, v in enumerate(nodes):
            if v in special_labels:
                self.nl[nv] = special_labels[v]
            else:
                # self.nl[nv] = graph.node[v][label_key]
                self.nl[nv] = graph.nodes[v][label_key]
            self.al[nv] = list(graph.neighbors(v))

        for v1, v2 in graph.edges():
            a, b = node_id[v1], node_id[v2]
            self.am[a, b] = self.am[b, a] = 1

    def __len__(self):
        return self.num_nodes


def vec(mat):
    assert mat.ndim == 2
    if isinstance(mat, spmatrix):
        mat = mat.toarray()
    return mat.flatten('F')


def invvec(M, m, n):
    return np.reshape(M, (m, n), 'F')


def matlab_graphs_2_kgraphs(graphs_in_a_cell, label_key='label'):
    # Copy 'am', 'nl', and 'el' from MATLAB graphs.
    # Adjacency list 'al' will be constructed from 'am'.
    graphs_in_a_cell = graphs_in_a_cell.flatten()
    kgraphs = [None] * len(graphs_in_a_cell)
    for gi, g in enumerate(graphs_in_a_cell):
        am = g['am']  # N x N
        m, N = am.shape
        assert m == N

        # create an intermediate networkx undirected graph
        graph = nx.Graph()
        graph.add_nodes_from(range(am.shape[0]))
        graph.add_edges_from(list(zip(*np.nonzero(am))))

        try:
            node_labels = g['nl'][0][0][0].flatten()  # N x 1
            assert len(node_labels) == N
            for i, label in enumerate(node_labels):
                # graph.node[i]['label'] = label
                graph.nodes[i]['label'] = label
        except ValueError:  # no node labels...
            pass

        try:
            one_based_edge_labels = g['el'][0][0][0]  # |E| x 3
            for i in range(one_based_edge_labels.shape[0]):
                r, c, edge_label = one_based_edge_labels[i, :]
                # graph.edge[r - 1][c - 1][label_key] = edge_label
                graph.edges[r - 1, c - 1][label_key] = edge_label
        except ValueError:  # no edge labels...
            pass

        kgraphs[gi] = KGraph(graph)

    return kgraphs
