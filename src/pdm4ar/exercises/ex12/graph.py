from typing import TypeVar, Set, Mapping
from networkx import DiGraph
import numpy as np
from dg_commons.sim.models.vehicle import VehicleState
from typing import List
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.maps import DgLanelet

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
    half_lane_width: float,
    lane_orientation: float,
) -> WeightedGraph:
    graph = DiGraph()

    # calculate deltas
    deltas = [
        (state.x - current_state.x, state.y - current_state.y, state.psi - current_state.psi)
        for state in end_states_traj
    ]

    # add root node
    graph.add_node((0, current_state.x, current_state.y, current_state.psi))  # (level, x, y, psi)

    # recursive function to generate children
    def add_children(level, x, y, psi):
        if level >= depth:
            return

        for i, (dx, dy, dpsi) in enumerate(deltas):
            # check if child is within lane boundaries
            position = np.array(
                [
                    x + dx * np.cos(psi - lane_orientation) - dy * np.sin(psi - lane_orientation),
                    y + dy * np.cos(psi - lane_orientation) + dx * np.sin(psi - lane_orientation),
                ]
            )
            lanelet_id = lanelet_network.find_lanelet_by_position([position])
            if not lanelet_id[0]:
                continue

            # add clearance to road boundaries
            lanelet = lanelet_network.find_lanelet_by_id(lanelet_id[0][0])
            no_adjacent_left = lanelet.adj_left is None
            no_adjacent_right = lanelet.adj_right is None
            if no_adjacent_left:
                # alternative way to calculate half_lane_width
                # left_boundary = lanelet.left_vertices
                # right_boundary = lanelet.right_vertices
                # widths = [
                #     np.linalg.norm(np.array([left[0], left[1]]) - np.array([right[0], right[1]]))
                #     for left, right in zip(left_boundary, right_boundary)
                # ]
                # average_width = np.mean(widths)
                # half_lane_width = average_width / 2
                position[0] -= half_lane_width * np.sin(psi - lane_orientation)
                position[1] += half_lane_width * np.cos(psi - lane_orientation)
                lanelet_id = lanelet_network.find_lanelet_by_position([position])
                if not lanelet_id[0]:
                    continue

            elif no_adjacent_right:
                position[0] += half_lane_width * np.sin(psi - lane_orientation)
                position[1] -= half_lane_width * np.cos(psi - lane_orientation)
                lanelet_id = lanelet_network.find_lanelet_by_position([position])
                if not lanelet_id[0]:
                    continue

            # half_lane_width * np.cos(lane_orientation)

            # only add child if psi is acceptable
            # if (
            #     psi + dpsi > np.pi / 4 or psi + dpsi < -np.pi / 4
            # ):  # TODO: necessary?, heuristically chose 45deg, needs refinement
            #     continue

            cmds = controls_traj[i]  # control commands to get from parent to child
            child = (
                level + 1,
                x + dx * np.cos(psi - lane_orientation) - dy * np.sin(psi - lane_orientation),
                y + dy * np.cos(psi - lane_orientation) + dx * np.sin(psi - lane_orientation),
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
