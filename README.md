[![CI](https://github.com/sanghack81/pyGK/actions/workflows/ci.yml/badge.svg)](https://github.com/sanghack81/pyGK/actions/workflows/ci.yml)

# pyGK

`pyGK` is a Python implementation of graph kernels based on MATLAB code by
Nino Shervashidze.

Implemented kernels include:

- shortest-path kernels for unlabeled and discretely labeled graphs;
- random-walk kernels for unlabeled and discretely labeled graphs;
- the Weisfeiler-Lehman kernel for discretely labeled graphs;
- the labeled 3-graphlet kernel;
- unlabeled 3-, 4-, and 5-graphlet kernels.

## Installation

pyGK requires Python 3.10 or later.

```bash
python -m pip install .
```

For an editable development install with the test and build tools:

```bash
python -m pip install -e ".[test]"
```

## Example

```python
import networkx as nx

from pygk.unlabeled import shortest_path_kernel
from pygk.utils import KGraph

graphs = [
    KGraph(nx.path_graph(5)),
    KGraph(nx.cycle_graph(5)),
]
kernel, features = shortest_path_kernel(graphs)
print(kernel)
```

`kernel` is a 2 × 2 Gram matrix here, and `features` is the shortest-path
feature matrix with one column per graph. Other kernel functions are in
[`pygk.unlabeled`](pygk/unlabeled.py),
[`pygk.labeled`](pygk/labeled.py), and
[`pygk.graphlets`](pygk/graphlets.py).

For labeled kernels, set a `label` attribute on each NetworkX node before
constructing `KGraph`:

```python
graph = nx.path_graph(5)
nx.set_node_attributes(graph, "carbon", "label")
labeled_graph = KGraph(graph)
```

## Development

```bash
python -m pytest -q
python -m ruff check .
python -m build
```

The numerical regression tests compare results with the original MATLAB
implementation using the datasets under `data/`.

## License

Apache License 2.0. See `NOTICE.txt`.
