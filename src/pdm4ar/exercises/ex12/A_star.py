from .graph import WeightedGraph
import heapq
from abc import ABC, abstractmethod
from typing import Optional, List, TypeVar
from dataclasses import dataclass
from dg_commons.sim.models.vehicle import VehicleState
import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork
from dg_commons.sim import extract_pose_from_state
from dg_commons.maps.lanes import DgLanelet

X = TypeVar("X")

Path = Optional[List[X]]
"""A path as a list of nodes."""

OpenedNodes = Optional[List[X]]
"""Also the opened nodes is a list of nodes"""

@dataclass
class InformedGraphSearch(ABC):
    graph: WeightedGraph

    @abstractmethod
    def path(self, start: X, goal: X) -> Path:
        # Abstract function. Nothing to do here.
        pass

@dataclass
class Astar(InformedGraphSearch):

    # TODO do we want to use X or VehicleState as type

    def heuristic(self, u: X, v: X) -> float:
        # TODO implement by usage of the cost_function in the cost_function branch
        pass
        
        
    def path(self, start: X, goal: X, lanelet_id: int) -> Path:
        graph = self.graph
        heap: list[tuple[float, X]] = [(0, start)]  # (total_cost, node) it's important that cost comes first s.t. heapq can prioritize correctly
        cost_to_reach: dict[X, float] = {start: 0}  # node: cost. Is the cost for the path from the start to the node
        predecessor: dict[X, Optional[X]] = {start: None}
        already_expanded_nodes: set[X] = set()

        while heap: # while heap is not empty
            current_total_cost, root = heapq.heappop(heap)

            if root in already_expanded_nodes:
                continue # Skip already processed nodes
            already_expanded_nodes.add(root)

            if root == goal:
                found_path: Path = []
                node: X = root
                while node is not None:
                    found_path.append(node)
                    node = predecessor[node]
                
                found_path.reverse()
                return found_path
            
            new_states: OpenedNodes = graph.adj_list[root]
            for new_state in new_states:
                if new_state in already_expanded_nodes:
                    continue # Skip already processed neighbors

                weight = graph.get_weight(root, new_state)
                new_cost_to_reach = cost_to_reach[root] + weight # costToReach(s) + w(s,a,s')       w is the cost from the current node to the new / subsequent node

                total_cost = new_cost_to_reach + self.heuristic(new_state, goal)  # heuristik muss nur von aktueller node dazugerechnet werden und nicht die bisherigen heuristiken auch in den costToReach reinrechnen!
                
                # if new state state has not been visited yet or if new path is cheaper
                # it's possible that the new_state has already been reached through another (potentially cheaper) path which is already stored in cost_to_reach[new_state]
                if new_state not in cost_to_reach or new_cost_to_reach < cost_to_reach[new_state]:
                    cost_to_reach[new_state] = new_cost_to_reach
                    predecessor[new_state] = root
                    heapq.heappush(heap, (total_cost, new_state))

        return [] # return failure
    
    def cost_function(self, current_node: VehicleState, goal: VehicleState, lanelet_network: LaneletNetwork, lanelet_id: int) -> float:
        # def cost_function(self, current_node: VehicleState, primitive: Trajectory, goal: VehicleState) -> float:
        # TODO missing penalties:
        # - Collision rate
        # - Success rate
        # - Time until changed to goal lane
        # - Computation time

        # h(n): Heuristic cost (distance to the goal)
        # - heading difference from car to current lane

        # g(n): Cost-so-far (cost of the motion primitive)
        # - Risk (minimum time to collision of trajectory) here we will need our predition module
        # - Speed should be not too fast and not too slow
        # - Discomfort level (RMSE of acc)
        # (- primitive_length maybe, but it is not penalized in the perf_metrics.py)


        # relative heading penalty
        lane_heading_angle = self.get_relative_heading(lanelet_id, current_node)
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

        g_n = (
            heading_cost + speed_cost
        )

        # h(n): Heuristic cost (distance to the goal)
        dist_to_goal_x = goal.x - current_node.x
        dist_to_goal_y = goal.y - current_node.y
        euclidean_distance = np.sqrt(dist_to_goal_x**2 + dist_to_goal_y**2)

        # Orientation difference to goal
        orientation_penalty = abs(goal.psi - current_node.psi)

        h_n = euclidean_distance + 0.1 * orientation_penalty

        return g_n + h_n
    

    def get_relative_heading(self, state: VehicleState, lanelet_network: LaneletNetwork, lanelet_id: int) -> float:
        lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
        dg_lanelet = DgLanelet.from_commonroad_lanelet(lanelet)

        pose = extract_pose_from_state(state) # VechicleState to SE2
        lane_pose = dg_lanelet.lane_pose_from_SE2_generic(pose)

        relative_heading = lane_pose.relative_heading

        return relative_heading

