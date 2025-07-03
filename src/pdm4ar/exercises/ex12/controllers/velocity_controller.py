from dataclasses import dataclass
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.goals import PlanningGoal
import frozendict
import math
import numpy as np

@dataclass
class VelocityControllerParams:
    d_ref: float = 2.0
    d_ref_K: float = 0.5
    T: float = 8.0

class VelocityController:
    def __init__(self, params: VelocityControllerParams, vp, goal: PlanningGoal, lanelet_network, name, dt):
        self.params = params
        self.vp = vp
        self.goal = goal
        self.lanelet_network = lanelet_network
        self.name = name
        self.dt = dt
        self.init_abstand = False
        self.last_e = 0.0

    def abstandhalteassistent(self, current_state: VehicleState, players: frozendict) -> float:
        """
        Velocity controller that maintains a safe distance to the vehicle ahead.
        If no vehicle is ahead, it adjusts the speed to avoid velocity penalties.
        If a vehicle is ahead, it calculates the distance to the vehicle and adjusts the speed accordingly.
        The controller uses a proportional control strategy based on the distance to the vehicle ahead.
        """

        player_ahead = None
        goal_x = self.goal.goal_polygon.centroid.x
        my_lanelet_id = self.lanelet_network.find_lanelet_by_position([np.array([current_state.x, current_state.y])])

        if my_lanelet_id[0]:
            current_lane_ids = self.retrieve_current_lane_ids(my_lanelet_id[0][0])
            for player in players:
                if player != self.name:
                    player_lanelet_id = self.lanelet_network.find_lanelet_by_position(
                        [np.array([players[player].state.x, players[player].state.y])]
                    )
                    if not player_lanelet_id[0]:
                        continue
                    if player_lanelet_id[0][0] in current_lane_ids:
                        if np.abs(goal_x - players[player].state.x) < np.abs(
                            goal_x - current_state.x
                        ):  # note: has singularity at lane heading of 90 deg
                            player_ahead = players[player]

        if player_ahead:
            dist_to_player = math.sqrt(
                (player_ahead.state.x - current_state.x) ** 2 + (player_ahead.state.y - current_state.y) ** 2
            )
            d_ref = self.params.d_ref_K * player_ahead.state.vx
            poly_edges = list(player_ahead.occupancy.exterior.coords)
            l_half_ahead = math.sqrt(
                (poly_edges[0][0] - player_ahead.state.x) ** 2 + (poly_edges[0][1] - player_ahead.state.y) ** 2
            )
            if d_ref < l_half_ahead + 1.5:
                d_ref = l_half_ahead + 1.5

            e = dist_to_player - d_ref
            if not self.init_abstand:
                self.init_abstand = True
                self.last_e = e
                return (player_ahead.state.vx / self.params.T + self.params.d_ref) * (
                    self.last_e
                )  # P-controler for first time step, here last_e is also the current error

            de = (e - self.last_e) / float(self.dt)
            self.last_e = e
            K_p = player_ahead.state.vx / self.params.T + self.params.d_ref

            return K_p * (self.params.T * de + e)

        else:
            # increase/decrease speed if car is too slow/fast (avoid velocity penalty) - else keep speed
            if current_state.vx > 25:  #
                return self.vp.acc_limits[0]
            elif current_state.vx < 5:
                return self.vp.acc_limits[1]
            else:
                return 0.0

    def retrieve_current_lane_ids(self, lane_id) -> list:
        """
        Retrieve the current lane IDs starting from the given lane ID.
        A lane is a concatenation of multiple lanelets.
        """

        current_lane_ids = [lane_id]
        lanelet = self.lanelet_network.find_lanelet_by_id(current_lane_ids[-1])
        successor_id = lanelet.successor

        while successor_id:
            current_lane_ids.append(successor_id[0])
            successor_id = self.lanelet_network.find_lanelet_by_id(successor_id[0]).successor

        return current_lane_ids
