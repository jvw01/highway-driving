from calendar import c
from hmac import new
from mimetypes import init
from operator import is_
from typing import TypeVar, Set, Mapping
from networkx import DiGraph
import numpy as np
from dg_commons.sim.models.vehicle import VehicleState
from typing import List
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.maps import DgLanelet
from commonroad.scenario.lanelet import LaneletNetwork
from dg_commons.sim import extract_pose_from_state
from shapely import LineString
from shapely.geometry import Polygon
from commonroad.scenario.lanelet import LaneletNetwork
from shapely.affinity import translate, rotate
from rtree import index

from pdm4ar.exercises_def.ex09 import goal
import time


class EdgeNotFound(Exception):  # TODO: necessary?
    pass


# A generic type for nodes in a graph.
X = TypeVar("X")

# An adjacency list from node to a set of nodes.
AdjacencyList = Mapping[X, Set[X]]


class WeightedGraph:
    graph: DiGraph
    adj_list: AdjacencyList
    start_node: tuple
    virtual_goal_node: tuple

    def __init__(self, graph: DiGraph, adj_list: AdjacencyList, start_node: tuple, goal_node: tuple):
        self.graph = graph
        self.adj_list = adj_list
        self.start_node = start_node
        self.goal_node = goal_node

    def get_weight(self, u: tuple, v: tuple) -> float:
        """
        :param u: The "from" of the edge
        :param v: The "to" of the edge
        :return: The weight associated to the edge, raises an Exception if the edge does not exist
        """
        try:
            return self.graph[u][v]["weight"]
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
    goal_id: list,
) -> WeightedGraph:

    graph = DiGraph()

    # calculate deltas
    deltas = [
        (state.x - current_state.x, state.y - current_state.y, state.psi - current_state.psi)
        for state in end_states_traj
    ]

    # add root node
    init_node = (0, current_state, False, tuple([]))
    graph.add_node(init_node)

    # recursive function to generate children
    def add_children(previous_node):
        level = previous_node[0]
        x = previous_node[1].x
        y = previous_node[1].y
        psi = previous_node[1].psi
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
            is_goal = True if lanelet_id[0][0] in goal_id else False

            # start_veh = time.time()
            child_vehicle_state = VehicleState(
                x=x + delta_pos[0], y=y + delta_pos[1], psi=psi + dpsi, vx=current_state.vx, delta=current_state.delta
            )
            # end_veh = time.time()
            # print(f"Generating the VehicleState took {end_veh - start_veh} seconds.")

            child = (
                level + 1,
                child_vehicle_state,
                is_goal,
                tuple(cmds),
            )  # (level, VehicleState, boolean goal, control commands)

            graph.add_node(child)
            graph.add_edge(
                previous_node,
                child,
                weight=cost_function(
                    current_node=previous_node, lanelet_network=lanelet_network, lanelet_id=lanelet_id[0][0]
                ),
            )

            # recursively add children for child
            add_children(child)

    add_children(init_node)

    # add virtual goal node
    # start_virt_goal = time.time()
    virtual_goal_vehicle_state = VehicleState(x=0, y=0, psi=0, vx=0, delta=0)  # TODO: what values for virtual goal?
    goal_nodes = [node for node in graph.nodes if node[2] == True]  # list of all nodes on the goal lane
    virtual_goal_node = (-1, virtual_goal_vehicle_state, False, tuple([]))  # virtual goal node
    graph.add_node(virtual_goal_node)  # virtual goal node
    for goal_node in goal_nodes:
        graph.add_edge(goal_node, virtual_goal_node, weight=0.0)  # edge from goal nodes to virtual goal node is 0
    # end_virt_goal = time.time()
    # print(f"Generating the virtual goal node took {end_virt_goal - start_virt_goal} seconds.")

    # create adjacency list for A* algorithm
    adj_list = {node: set(neighbors.keys()) for node, neighbors in graph.adjacency()}

    return WeightedGraph(graph=graph, adj_list=adj_list, start_node=init_node, goal_node=virtual_goal_node)


def cost_function(
    current_node: tuple, lanelet_network: LaneletNetwork, lanelet_id: int
) -> float:  # add virtual goal node
    # TODO missing penalties:
    # - Collision rate
    # - Success rate
    # - Time until changed to goal lane
    # - Computation time
    # - Risk (minimum time to collision of trajectory) here we will need our predition module
    # - Speed should be not too fast and not too slow
    # - Discomfort level (RMSE of acc)
    # (- primitive_length maybe, but it is not penalized in the perf_metrics.py)

    current_vehicle_state = current_node[1]

    # TODO implement something that lets us go to the goal, e.g. time to goal or distnace to goal

    # relative heading penalty
    # start_rel_heading = time.time()
    # lane_heading_angle = get_relative_heading(
    #     state=current_vehicle_state, lanelet_network=lanelet_network, lanelet_id=lanelet_id
    # )
    # heading_penalty = (np.abs(lane_heading_angle) - 0.1) * 10.0
    # heading_penalty = np.clip(heading_penalty, 0.0, 1.0)
    # heading_cost = 5.0 * heading_penalty
    # end_rel_heading = time.time()
    # print(f"Calculating the relative heading took {end_rel_heading - start_rel_heading} seconds.")

    # risk penalty
    # TODO

    # speed penalty
    # start_speed = time.time()
    v_diff = np.maximum(current_vehicle_state.vx - 25.0, 5.0 - current_vehicle_state.vx)
    velocity_penalty = v_diff / 5.0
    velocity_penalty = np.clip(velocity_penalty, 0.0, 1.0)
    speed_cost = 5.0 * velocity_penalty
    # end_speed = time.time()
    # print(f"Calculating the speed penalty took {end_speed - start_speed} seconds.")

    # TODO Discomfort cost (weighted acc. RMSE according to ISO 2631-1)
    # for the weighted acc. use the butterworth bandpass filter with the iso params
    # RMSE: np.sqrt(np.mean(np.square(signal)))

    # TODO primitive_length (trajectory distance)
    # primitive_length = sum(
    #     np.sqrt((next_state.x - state.x)**2 + (next_state.y - state.y)**2)
    #     for state, next_state in zip(primitive.values[:-1], primitive.values[1:]) # produces paris of consecutive states
    # )

    # takes the level of the node as cost (increases with int steps from the starting node)
    heuristic_cost = (current_node[0] + 1) if current_node[2] else 0.0
    heuristic_weighting_factor = 1.0

    cost = speed_cost + (heuristic_weighting_factor * heuristic_cost)

    return cost


### BACKUP ###
# def get_relative_heading(state: VehicleState, lanelet_network: LaneletNetwork, lanelet_id: int) -> float:
#     lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
#     dg_lanelet = DgLanelet.from_commonroad_lanelet(lanelet)

#     pose = extract_pose_from_state(state) # VechicleState to SE2
#     lane_pose = dg_lanelet.lane_pose_from_SE2_generic(pose)

#     relative_heading = lane_pose.relative_heading

#     return relative_heading


# def calc_new_occupancy(current_occupancy: Polygon, delta_pos: np.ndarray, dpsi: float) -> Polygon:
#     translated_occupancy = translate(current_occupancy, xoff=delta_pos[0], yoff=delta_pos[1])
#     return rotate(translated_occupancy, angle=dpsi, origin=translated_occupancy.centroid, use_radians=True)
