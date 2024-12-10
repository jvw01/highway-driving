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
    lanelet_id: int,
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
    graph.add_node((0, current_state.x, current_state.y, current_state.psi))  # (level, x, y)

    # recursive function to generate children
    def add_children(level, x, y, psi):
        if level >= depth:
            return

        for i, (dx, dy, dpsi) in enumerate(deltas):
            # check if child is within lane boundaries
            position = np.array(
                [
                    x + dx * np.cos(psi) - dy * np.sin(psi),
                    y + dy * np.cos(psi) + dx * np.sin(psi),
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
                position[0] += half_lane_width * np.sin(np.abs(lane_orientation))  # TODO: angle
                position[1] += half_lane_width * np.cos(np.abs(lane_orientation))  # TODO: angle
                lanelet_id = lanelet_network.find_lanelet_by_position([position])
                if not lanelet_id[0]:
                    continue

            elif no_adjacent_right:
                position[0] -= half_lane_width * np.sin(np.abs(lane_orientation))  # TODO: angle
                position[1] -= half_lane_width * np.cos(np.abs(lane_orientation))  # TODO: angle
                lanelet_id = lanelet_network.find_lanelet_by_position([position])
                if not lanelet_id[0]:
                    continue

            # half_lane_width * np.cos(lane_orientation)

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

    # these weights subsequently land in the cost-to-go of the A* algo
    weights = {(u, v): cost_function(current_node=u, lanelet_network=lanelet_network, lanelet_id=lanelet_id) 
               for u, v, _ in graph.edges(data=True)}

    # Debugging prints:
    # for edge, weight in weights.items():
    #     print(f"Edge: {edge}, Weight: {weight}")

    return WeightedGraph(adj_list, weights, graph)


def cost_function(current_node: VehicleState, lanelet_network: LaneletNetwork, lanelet_id: int) -> float:
    # TODO missing penalties:
    # - Collision rate
    # - Success rate
    # - Time until changed to goal lane
    # - Computation time
    # - Risk (minimum time to collision of trajectory) here we will need our predition module
    # - Speed should be not too fast and not too slow
    # - Discomfort level (RMSE of acc)
    # (- primitive_length maybe, but it is not penalized in the perf_metrics.py)

    # TODO implement something that lets us go to the goal, e.g. time to goal or distnace to goal

    # relative heading penalty
    lane_heading_angle = get_relative_heading(lanelet_id, current_node)
    heading_penalty = (np.abs(lane_heading_angle) - 0.1) * 10.0
    heading_penalty = np.clip(heading_penalty, 0.0, 1.0)
    heading_cost = 5.0 * heading_penalty

    # risk penalty
    # TODO

    # speed penalty
    v_diff = np.maximum(current_node.vx - 25.0, 5.0 - current_node.vx)
    velocity_penalty = v_diff / 5.0
    velocity_penalty = np.clip(velocity_penalty, 0.0, 1.0)
    speed_cost = 5.0 * velocity_penalty

    # TODO Discomfort cost (weighted acc. RMSE according to ISO 2631-1)
    # for the weighted acc. use the butterworth bandpass filter with the iso params
    # RMSE: np.sqrt(np.mean(np.square(signal)))


    # TODO primitive_length (trajectory distance)
    # primitive_length = sum(
    #     np.sqrt((next_state.x - state.x)**2 + (next_state.y - state.y)**2)
    #     for state, next_state in zip(primitive.values[:-1], primitive.values[1:]) # produces paris of consecutive states
    # )

    cost = heading_cost + speed_cost

    return cost


def get_relative_heading(self, state: VehicleState, lanelet_network: LaneletNetwork, lanelet_id: int) -> float:
    lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
    dg_lanelet = DgLanelet.from_commonroad_lanelet(lanelet)

    pose = extract_pose_from_state(state) # VechicleState to SE2
    lane_pose = dg_lanelet.lane_pose_from_SE2_generic(pose)

    relative_heading = lane_pose.relative_heading

    return relative_heading

