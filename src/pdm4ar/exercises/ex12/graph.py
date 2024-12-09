from typing import TypeVar, Set, Mapping
from networkx import DiGraph
import numpy as np
from dg_commons.sim.models.vehicle import VehicleState
from typing import List
from dg_commons.sim.models.vehicle import VehicleCommands

# A generic type for nodes in a graph.
X = TypeVar("X")


class EdgeNotFound(Exception):
    pass


AdjacencyList = Mapping[X, Set[X]]
"""An adjacency list from node to a set of nodes."""


class WeightedGraph:
    adj_list: AdjacencyList
    weights: Mapping[tuple[X, X], float]
    graph: DiGraph

    def __init__(self, adj_list: dict, weights: dict, graph: DiGraph):
        self.adj_list = adj_list
        self.weights = weights
        self.graph = graph

    def get_weight(self, u: X, v: X) -> float:
        """
        :param u: The "from" of the edge
        :param v: The "to" of the edge
        :return: The weight associated to the edge, raises an Exception if the edge does not exist
        """
        try:
            return self.weights[(u, v)]
        except KeyError:
            raise EdgeNotFound(f"Cannot find weight for edge: {(u, v)}")


def generate_graph(
    current_state: VehicleState,
    end_states_traj: List[VehicleState],
    controls_traj: List[List[VehicleCommands]],
    depth: int,
    lanelet_network,
) -> WeightedGraph:
    graph = DiGraph()

    # calculate deltas
    deltas = [
        (state.x - current_state.x, state.y - current_state.y, state.psi - current_state.psi)
        for state in end_states_traj
    ]

    # add root node
    graph.add_node((0, current_state.x, current_state.y, current_state.psi))  # (level, x, y)

    # recursive function to generate children
    def add_children(level, x, y, psi):
        if level >= depth:
            return

        for i, (dx, dy, dpsi) in enumerate(deltas):
            # check if child is within lane boundaries TODO: add clearance to road boundaries
            position = np.array([x + dx * np.cos(psi) - dy * np.sin(psi), y + dy * np.cos(psi) + dx * np.sin(psi)])
            lanelet_id = lanelet_network.find_lanelet_by_position([position])
            if not lanelet_id[0]:
                continue

            # only add child if psi is acceptable
            if (
                psi + dpsi > np.pi / 4 or psi + dpsi < -np.pi / 4
            ):  # TODO: necessary?, heuristically chose 45deg, needs refinement
                continue

            cmds = controls_traj[i]  # control commands to get from parent to child
            child = (
                level + 1,
                x + dx * np.cos(psi) - dy * np.sin(psi),
                y + dy * np.cos(psi) + dx * np.sin(psi),
                psi + dpsi,
                tuple(cmds),
            )

            graph.add_node(child)
            graph.add_edge((level, x, y, psi), child)

            # recursively add children for child
            add_children(child[0], child[1], child[2], child[3])  # (level, x, y, psi)

    add_children(0, current_state.x, current_state.y, current_state.psi)

    # convert networkx DiGraph to the custom WeightedGraph structure
    adj_list = {node: set(neighbors.keys()) for node, neighbors in graph.adjacency()}
    weights = {(u, v): data.get("weight", 1.0) for u, v, data in graph.edges(data=True)}  # TODO: need to define weights

    return WeightedGraph(adj_list, weights, graph)
