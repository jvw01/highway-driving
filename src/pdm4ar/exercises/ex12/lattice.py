from typing import TypeVar, Set, Mapping
from networkx import MultiDiGraph

# A generic type for nodes in a graph.
X = TypeVar("X")


class WeightedGraph:
    adj_list: Mapping[X, Set[X]]
    weights: Mapping[tuple[X, X], float]
    graph: MultiDiGraph

    def __init__(self, adj_list: dict, weights: dict, graph: MultiDiGraph):
        self.adj_list = adj_list  # dict
        self.weights = weights  # dict
        self.graph = graph  # MultiDiGraph
