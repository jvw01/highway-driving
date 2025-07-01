import heapq
from abc import ABC, abstractmethod
from typing import Optional, List, TypeVar
from dataclasses import dataclass
from .graph import WeightedGraph

X = TypeVar("X")

Path = Optional[List[X]]
"""A path as a list of nodes."""

OpenedNodes = Optional[List[X]]
"""Also the opened nodes is a list of nodes"""

@dataclass
class InformedGraphSearch(ABC):
    weighted_graph: WeightedGraph

    @abstractmethod
    def path(self, start: X, goal: X) -> Path:
        # Abstract function. Nothing to do here.
        pass


# node structure: (level: int, state: VehicleState, is_goal: bool, cmds: tuple[List[VehicleCommands]])
@dataclass
class Dijkstra(InformedGraphSearch):
    def path(self, start_node: tuple, goal_node: tuple) -> Path:
        weighted_graph = self.weighted_graph
        heap: list[tuple[float, tuple]] = [(0, start_node)]  # (total_cost, node) it's important that cost comes first s.t. heapq can prioritize correctly
        cost_to_reach: dict[tuple, float] = {start_node: 0}  # node: cost. Is the cost for the path from the start to the node
        predecessor: dict[tuple, Optional[tuple]] = {start_node: None}
        already_expanded_nodes: set[tuple] = set()

        while heap: # while heap is not empty
            current_total_cost, root = heapq.heappop(heap)

            if root in already_expanded_nodes:
                continue # Skip already processed nodes
            already_expanded_nodes.add(root)

            if root == goal_node:
                found_path: Path = []
                node: tuple = root
                while node is not None:
                    found_path.append(node)
                    node = predecessor[node]

                found_path.reverse()
                return found_path

            new_states: OpenedNodes = weighted_graph.adj_list[root]
            for new_state in new_states:
                if new_state in already_expanded_nodes:
                    continue # Skip already processed neighbors

                weight = weighted_graph.get_weight(root, new_state)
                new_cost_to_reach = cost_to_reach[root] + weight # costToReach(s) + w(s,a,s') --> w is the cost from the current node to the new / subsequent node
                total_cost = new_cost_to_reach

                # if new state state has not been visited yet or if new path is cheaper
                # it's possible that the new_state has already been reached through another (potentially cheaper) path which is already stored in cost_to_reach[new_state]
                if new_state not in cost_to_reach or new_cost_to_reach < cost_to_reach[new_state]:
                    cost_to_reach[new_state] = new_cost_to_reach
                    predecessor[new_state] = root
                    heapq.heappush(heap, (total_cost, new_state))

        return [] # return failure
