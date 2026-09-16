import networkx as nx
import numpy as np
import pytest

from pygk.graphlets import gest_kernel, sample_size
from pygk.labeled import labeled_random_walk_kernel
from pygk.unlabeled import random_walk_kernel
from pygk.utils import KGraph


def make_path_graph():
    graph = nx.Graph()
    graph.add_nodes_from(["left", "middle", "right"])
    graph.add_edges_from([("left", "middle"), ("middle", "right")])
    nx.set_node_attributes(graph, "node", "label")
    return graph


def test_kgraph_remaps_arbitrary_node_ids():
    graph = KGraph(make_path_graph())

    assert graph.al == [[1], [0, 2], [1]]
    np.testing.assert_array_equal(
        graph.am,
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
    )


def test_kgraph_allows_missing_labels_for_unlabeled_kernels():
    graph = KGraph(nx.path_graph(3))

    assert graph.nl.tolist() == [None, None, None]
    kernel = random_walk_kernel([graph], 0.1)
    assert kernel.shape == (1, 1)
    assert np.isfinite(kernel).all()


def test_random_walk_kernels_support_current_scipy():
    graph = KGraph(make_path_graph())

    unlabeled = random_walk_kernel([graph], 0.1)
    with pytest.warns(UserWarning, match="not tested"):
        labeled = labeled_random_walk_kernel([graph], 0.1)

    assert np.isfinite(unlabeled).all()
    assert np.isfinite(labeled).all()


@pytest.mark.parametrize("k", [2, 6, 3.5, True])
def test_gest_kernel_rejects_unsupported_graphlet_sizes(k):
    with pytest.raises(ValueError, match="not supported"):
        gest_kernel([], k)


@pytest.mark.parametrize("num_samples", [-2, 1.5, True])
def test_gest_kernel_rejects_invalid_sample_counts(num_samples):
    with pytest.raises(ValueError, match="number of samples"):
        gest_kernel([], 3, num_samples=num_samples)


@pytest.mark.parametrize(("delta", "epsilon"), [(0, 0.1), (0.1, 0)])
def test_sample_size_rejects_zero_denominators(delta, epsilon):
    with pytest.raises(ValueError):
        sample_size(delta, epsilon, 2)
