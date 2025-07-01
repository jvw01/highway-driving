from typing import TypeVar, Set, Mapping
from networkx import DiGraph
import numpy as np
from dg_commons.sim.models.vehicle import VehicleState
from typing import List
from dg_commons.sim.models.vehicle import VehicleCommands
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.lanelet import LaneletNetwork

class EdgeNotFound(Exception):
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
    num_goal_nodes: int

    def __init__(
        self, graph: DiGraph, adj_list: AdjacencyList, start_node: tuple, goal_node: tuple, num_goal_nodes: int
    ):
        self.graph = graph
        self.adj_list = adj_list
        self.start_node = start_node
        self.goal_node = goal_node
        self.num_goal_nodes = num_goal_nodes

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
    end_states_traj: List[List[VehicleState]],
    controls_traj: List[List[List[VehicleCommands]]],
    depth: int,
    lanelet_network: LaneletNetwork,
    goal_id: list,
    steps_lane_change: int,
) -> WeightedGraph:

    graph = DiGraph()

    # add root node
    init_node = (0, current_state, False, tuple([]))
    graph.add_node(init_node)

    previous_node = init_node
    next_start_node = None

    # move all states by the distance the car drives in one time step (when driving straight)
    dx = end_states_traj[0][0].x - current_state.x
    dy = end_states_traj[0][0].y - current_state.y
    for i in range(depth):
        level = i
        for k, state_list in enumerate(end_states_traj):
            level += 1  # start node == node at level 0 has already been added
            for n, state in enumerate(state_list):
                next_state = VehicleState(
                    x=state.x + i * dx,
                    y=state.y + i * dy,
                    psi=state.psi,
                    vx=state.vx,
                    delta=state.delta,
                )

                cmds = controls_traj[k][n]

                # calculate boolean to indicate goal node
                position = np.array([next_state.x, next_state.y])
                lanelet_id = lanelet_network.find_lanelet_by_position([position])
                if not lanelet_id[0]:
                    continue

                # only the last state of the lane change is a goal state
                is_goal = True if lanelet_id[0][0] in goal_id and k == steps_lane_change - 1 else False

                child = (
                    level,
                    next_state,
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

                # extract node where we attach the next lane change procedure
                if k == 0 and n == 0:
                    # to avoid deepcopy problem ...
                    next_start_state = VehicleState(
                        x=state.x + i * dx,
                        y=state.y + i * dy,
                        psi=state.psi,
                        vx=state.vx,
                        delta=state.delta,
                    )

                    next_start_node = (level, next_start_state, is_goal, tuple(cmds))

            # to avoid deepcopy problem ...
            previous_state = VehicleState(
                x=child[1].x,
                y=child[1].y,
                psi=child[1].psi,
                vx=child[1].vx,
                delta=child[1].delta,
            )
            previous_node = (level, previous_state, is_goal, tuple(cmds))

        previous_node = next_start_node

    # add virtual goal node
    virtual_goal_vehicle_state = VehicleState(x=0, y=0, psi=0, vx=0, delta=0)
    goal_nodes = [node for node in graph.nodes if node[2] == True]  # list of all nodes on the goal lane
    num_goal_nodes = len(goal_nodes)
    virtual_goal_node = (-1, virtual_goal_vehicle_state, False, tuple([]))  # virtual goal node
    graph.add_node(virtual_goal_node)  # virtual goal node
    for goal_node in goal_nodes:
        graph.add_edge(goal_node, virtual_goal_node, weight=0.0)  # edge from goal nodes to virtual goal node is 0

    # create adjacency list for Dijkstra algorithm
    adj_list = {node: set(neighbors.keys()) for node, neighbors in graph.adjacency()}

    return WeightedGraph(
        graph=graph, adj_list=adj_list, start_node=init_node, goal_node=virtual_goal_node, num_goal_nodes=num_goal_nodes
    )


def cost_function(
    current_node: tuple, lanelet_network: LaneletNetwork, lanelet_id: int
) -> float:  # add virtual goal node
    # takes the level of the node as cost (increases with int steps from the starting node)
    heuristic_cost = (current_node[0] + 1) if current_node[2] else 0.0
    heuristic_weighting_factor = 1.0

    cost = heuristic_weighting_factor * heuristic_cost

    return cost
