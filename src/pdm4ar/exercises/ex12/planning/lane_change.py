from dg_commons.sim.models.vehicle import VehicleState, VehicleCommands
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters

import numpy as np
from pdm4ar.exercises.ex12.planning.motion_primitves import calc_next_state

class LaneChangePlanner:

    def __init__(
        self,
        vp: VehicleParameters,
        vg: VehicleGeometry,
        dt_integral: float,
        scaling_dt: float,
        half_lane_width: float,
    ):
        self.vp = vp
        self.vg = vg
        self.dt_integral = dt_integral
        self.scaling_dt = scaling_dt
        self.half_lane_width = half_lane_width

    def calc_time_horizon_and_ddelta(self, current_state: VehicleState) -> tuple[float, float]:
        """
        Calculate the time horizon and ddelta for the lane change maneuver.
        The time horizon is calculated based on the distance the car drives in one motion primitive.
        The ddelta is calculated based on the distance the car drives in one motion primitive and the desired lane change distance.
        The desired lane change distance is approximately one third of the lane width in the y' direction.
        The ddelta is calculated iteratively until the desired lane change distance is reached within a tolerance of 1e-2.
        """

        # heurisitic approach: in first motion primitive with delta_max, the car drives approximately a sixth of the lane width in y' direction
        delta_y = self.half_lane_width * (1 / 3)
        n_steps = 0
        init_state_y = current_state.y
        next_state = VehicleState(
            x=current_state.x,
            y=current_state.y,
            psi=0,  # to make calculation easier - it does not matter for this calculation whether the road is at an angle
            vx=current_state.vx,
            delta=current_state.delta,
        )

        while (np.abs(next_state.y - init_state_y)) < delta_y:
            n_steps += 1
            next_state = calc_next_state(
                next_state, VehicleCommands(acc=0, ddelta=self.vp.ddelta_max), self.dt_integral, self.vp, self.vg
            )

        goal_state_y = next_state.y
        time_horizon = np.ceil(n_steps * self.dt_integral * 10) / 10  # round to nearest multiple of 0.1

        # calculate ddelta with new time horizon
        ddelta = self.vp.ddelta_max / 2  # start with ddelta_max / 2 as init guess for correct ddelta
        ddelta_interval = np.array([0, self.vp.ddelta_max])
        n_steps = int(time_horizon / self.dt_integral)
        init_state_y = current_state.y
        next_state = VehicleState(
            x=current_state.x,
            y=current_state.y,
            psi=0,  # to make calculation easier - it does not matter for this calculation whether the road is at an angle
            vx=current_state.vx,
            delta=current_state.delta,
        )

        calc_goal_state_y = current_state.y
        while np.abs(goal_state_y - calc_goal_state_y) > 1e-2:
            for _ in range(n_steps):
                next_state = calc_next_state(
                    next_state,
                    VehicleCommands(acc=0, ddelta=ddelta),
                    self.dt_integral,
                    self.vp,
                    self.vg,
                )

            if goal_state_y > 0:
                if goal_state_y - next_state.y > 0:
                    ddelta_interval = np.array([ddelta, ddelta_interval[1]])
                    ddelta = np.mean(ddelta_interval)
                else:
                    ddelta_interval = np.array([ddelta_interval[0], ddelta])
                    ddelta = np.mean(ddelta_interval)
            else:
                if np.abs(goal_state_y) - np.abs(next_state.y) > 0:
                    ddelta_interval = np.array([ddelta_interval[0], ddelta])
                    ddelta = np.mean(ddelta_interval)
                else:
                    ddelta_interval = np.array([ddelta, ddelta_interval[1]])
                    ddelta = np.mean(ddelta_interval)

            calc_goal_state_y = next_state.y
            next_state = VehicleState(
                x=current_state.x,
                y=current_state.y,
                psi=0,  # to make calculation easier - it does not matter for this calculation whether the road is at an angle
                vx=current_state.vx,
                delta=current_state.delta,
            )
        return (time_horizon, ddelta)
