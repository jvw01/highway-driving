from typing import TypeVar, Set, Mapping
from networkx import DiGraph

# A generic type for nodes in a graph.
X = TypeVar("X")


class WeightedGraph:
    adj_list: Mapping[X, Set[X]]
    weights: Mapping[tuple[X, X], float]
    graph: DiGraph

    def __init__(self, adj_list: dict, weights: dict, graph: DiGraph):
        self.adj_list = adj_list
        self.weights = weights
        self.graph = graph
