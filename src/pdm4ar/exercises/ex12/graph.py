from calendar import c
from hmac import new
from operator import is_
from typing import TypeVar, Set, Mapping
from networkx import DiGraph
import numpy as np
from dg_commons.sim.models.vehicle import VehicleState
from typing import List
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.maps import DgLanelet
from shapely.geometry import Polygon
from commonroad.scenario.lanelet import LaneletNetwork
from shapely.affinity import translate, rotate
from rtree import index

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
    lanelet_network: LaneletNetwork,
    half_lane_width: float,
    lane_orientation: float,
    goal_id: int,
) -> WeightedGraph:
    start = time.time()

    graph = DiGraph()

    # calculate deltas
    deltas = [
        (state.x - current_state.x, state.y - current_state.y, state.psi - current_state.psi)
        for state in end_states_traj
    ]

    # add root node
    graph.add_node((0, current_state.x, current_state.y, current_state.psi))

    # recursive function to generate children
    def add_children(level, x, y, psi):
        if level >= depth:
            return

        for i, (dx, dy, dpsi) in enumerate(deltas):
            # check if child is within lane boundaries
            delta_pos = np.array(
                [
                    dx * np.cos(psi - lane_orientation) - dy * np.sin(psi - lane_orientation),
                    dy * np.cos(psi - lane_orientation) + dx * np.sin(psi - lane_orientation),
                ]
            )
            position = np.array([x, y]) + delta_pos
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

            # only add child if psi is acceptable
            # if (
            #     psi + dpsi > np.pi / 4 or psi + dpsi < -np.pi / 4
            # ):  # TODO: necessary?, heuristically chose 45deg, needs refinement
            #     continue

            # create new child node
            cmds = controls_traj[i]  # control commands to get from parent to child

            # boolean to indicate goal node
            is_goal = True if lanelet_id == goal_id else False

            child = (
                level + 1,
                x + dx * np.cos(psi - lane_orientation) - dy * np.sin(psi - lane_orientation),
                y + dy * np.cos(psi - lane_orientation) + dx * np.sin(psi - lane_orientation),
                psi + dpsi,
                is_goal,
                tuple(cmds),
            )  # (level, x, y, psi, boolean goal, control commands)

            graph.add_node(child)
            graph.add_edge((level, x, y, psi), child)

            # recursively add children for child
            add_children(child[0], child[1], child[2], child[3])

    add_children(0, current_state.x, current_state.y, current_state.psi)

    # convert networkx DiGraph to the custom WeightedGraph structure
    adj_list = {node: set(neighbors.keys()) for node, neighbors in graph.adjacency()}
    weights = {(u, v): data.get("weight", 1.0) for u, v, data in graph.edges(data=True)}  # TODO: need to define weights

    end = time.time()
    print(f"Generating the graph took {end - start} seconds.")

    start = time.time()
    plot_adj_list(adj_list, lanelet_network.lanelet_polygons)
    end = time.time()
    print(f"Plotting the graph took {end - start} seconds.")

    return WeightedGraph(adj_list, weights, graph)


def calc_new_occupancy(current_occupancy: Polygon, delta_pos: np.ndarray, dpsi: float) -> Polygon:
    translated_occupancy = translate(current_occupancy, xoff=delta_pos[0], yoff=delta_pos[1])
    return rotate(translated_occupancy, angle=dpsi, origin=translated_occupancy.centroid, use_radians=True)


### ADDITIONAL HELPER FUNCTIONS ###
import matplotlib.pyplot as plt
import os
import time
from matplotlib.collections import LineCollection


def plot_adj_list(adj_list, lanelet_polygons):
    plt.figure(figsize=(30, 25), dpi=250)
    ax = plt.gca()

    # Collect all node coordinates
    node_coords = []
    for parent in adj_list.keys():
        parent_x, parent_y = parent[1], parent[2]
        node_coords.append((parent_x, parent_y))

    # Collect all edge coordinates
    edge_coords = []
    for parent, children in adj_list.items():
        parent_x, parent_y = parent[1], parent[2]
        for child in children:
            child_x, child_y = child[1], child[2]
            edge_coords.append([(parent_x, parent_y), (child_x, child_y)])

    # Plot all nodes at once
    node_coords = np.array(node_coords)
    plt.scatter(node_coords[:, 0], node_coords[:, 1], color="blue", s=1)

    # Plot all edges at once using LineCollection
    edge_collection = LineCollection(edge_coords, colors="blue", linewidths=0.3)
    ax.add_collection(edge_collection)

    # Plot static obstacles (from on_episode_init)
    for lanelet in lanelet_polygons:
        x, y = lanelet.shapely_object.exterior.xy
        plt.plot(x, y, linestyle="-", linewidth=0.8, color="darkorchid")

    ax.set_aspect("equal", adjustable="box")

    output_dir = "/tmp"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "graph.png")
    plt.savefig(filename, bbox_inches="tight")  # Save the plot with tight bounding box
    plt.close()
    print(f"Graph saved to {filename}")
