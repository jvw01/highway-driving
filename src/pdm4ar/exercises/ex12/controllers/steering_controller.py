from dataclasses import dataclass
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.controllers.steer import SteerController
import math
import numpy as np

@dataclass
class SteeringControllerParams:
    K_psi: float = 1.0
    K_dist: float = 0.1
    K_delta: float = 2.0
    K_d_psi: float = 0.5
    K_d_dist: float = 0.1
    K_d_delta: float = 0.1

class CustomSteerController:
    def __init__(self, params: SteeringControllerParams, dt, vp, lanelet_network):
        self.params = params
        self.dt = dt
        self.vp = vp
        self.lanelet_network = lanelet_network
        self.steer_controller = SteerController.from_vehicle_params(vehicle_param=vp)
        self.init_control = False
        self.last_dpsi = 0.0
        self.last_dist = 0.0
        self.last_delta = 0.0

    def spurhalteassistent(self, current_state: VehicleState, t: float) -> float:
        """
        Custom steering controller to keep the vehicle in the lane.
        This controller calculates the steering angle based on the vehicle's current state and the lanelet network.
        It uses the vehicle's position and heading to determine the distance to the lane center and the heading error.
        The controller applies a PD-like control strategy to adjust the steering angle.
        """

        cur_lanelet_id = self.lanelet_network.find_lanelet_by_position([np.array([current_state.x, current_state.y])])

        if not cur_lanelet_id[0]:
            print("No lanelet found")
            return 0.0

        cur_lanelet = self.lanelet_network._lanelets[cur_lanelet_id[0][0]]
        center_vertices = cur_lanelet.center_vertices
        lanelet_heading = math.atan2(
            center_vertices[-1][1] - center_vertices[0][1], center_vertices[-1][0] - center_vertices[0][0]
        )
        line_vec = center_vertices[-1] - center_vertices[0]
        normal_vec = np.array([-line_vec[1], line_vec[0]])
        normal_vec = normal_vec / np.linalg.norm(normal_vec)
        dist = np.dot(
            normal_vec, np.array([current_state.x, current_state.y]) - center_vertices[0]
        )  # should already have correct sign
        dpsi = current_state.psi - lanelet_heading

        if not self.init_control:
            self.init_control = True
            self.last_dist = dist
            self.last_dpsi = dpsi
            self.last_delta = current_state.delta
            return -1 * dpsi - 0.1 * dist - 2 * current_state.delta

        d_dist = (dist - self.last_dist) / float(self.dt)
        d_dpsi = (dpsi - self.last_dpsi) / float(self.dt)
        d_delta = (current_state.delta - self.last_delta) / float(self.dt)
        self.last_dist = dist
        self.last_dpsi = dpsi
        self.last_delta = current_state.delta

        ddelta = (
            - self.params.K_psi * dpsi
            - self.params.K_d_psi * d_dpsi
            - self.params.K_dist * dist
            - self.params.K_d_dist * d_dist
            - self.params.K_delta * current_state.delta
            - self.params.K_d_delta * d_delta
        )

        if abs(dist) < 0.05 and abs(current_state.delta) < 0.01:
            self.steer_controller.update_measurement(measurement=current_state.delta)
            self.steer_controller.update_reference(reference=0)
            ddelta = self.steer_controller.get_control(t)
            return min(max(ddelta, -self.vp.ddelta_max), self.vp.ddelta_max)

        return ddelta
