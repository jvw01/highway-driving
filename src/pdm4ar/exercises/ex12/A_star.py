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
    graph: WeightedGraph

    @abstractmethod
    def path(self, start: X, goal: X) -> Path:
        # Abstract function. Nothing to do here.
        pass

@dataclass
class Astar(InformedGraphSearch):

    # TODO adapt the in the agent.py that it only calls A* if a point in the graph is intersecting
    # with a goal_line (+buffer) and if not execute it and generate subsequently a new graph until
    # a goal was found

    # TODO do we want to use X or VehicleState as type

    def heuristic(self, u: X, v: X) -> float:
        # dist_to_goal_x = v.x - u.x
        # dist_to_goal_y = v.y - u.y
        # euclidean_distance = np.sqrt(dist_to_goal_x**2 + dist_to_goal_y**2)

        pass
        
    def path(self, start: X, goal: X) -> Path:
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

                # total_cost = new_cost_to_reach + self.heuristic(new_state, goal)  # heuristik muss nur von aktueller node dazugerechnet werden und nicht die bisherigen heuristiken auch in den costToReach reinrechnen!
                total_cost = new_cost_to_reach # TODO implement heuristic

                # if new state state has not been visited yet or if new path is cheaper
                # it's possible that the new_state has already been reached through another (potentially cheaper) path which is already stored in cost_to_reach[new_state]
                if new_state not in cost_to_reach or new_cost_to_reach < cost_to_reach[new_state]:
                    cost_to_reach[new_state] = new_cost_to_reach
                    predecessor[new_state] = root
                    heapq.heappush(heap, (total_cost, new_state))

        return [] # return failure
    

