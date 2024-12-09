from dg_commons.sim.models.vehicle import VehicleState
from typing import List
import numpy as np
from dg_commons.sim.models.vehicle import VehicleCommands
from itertools import product
from dg_commons.planning.motion_primitives import MotionPrimitivesGenerator


def generate_primat(
    x0: VehicleState,
    mpg: MotionPrimitivesGenerator,
    steer_range: float,
    n_steer: int,
) -> tuple[List[VehicleState], List[List[VehicleCommands]]]:
    """
    Reimplement method mpg.generate(x) to generate motion primitives specifically for our problem.
    """
    end_states_traj = []
    controls_traj = []

    # create samples in user-defined range - assume constant velocity and variable steering angles
    v_samples = np.array([x0.vx])
    sa_samples = np.linspace(
        *(x0.delta - steer_range, x0.delta + steer_range, n_steer)
    )  # TODO: need to check if steering angle is valid

    for v_sample, sa_sample in product(v_samples, sa_samples):
        # calculate acceleration and steering angle rate
        horizon = float(mpg.param.dt * mpg.param.n_steps)
        acc = (v_sample - x0.vx) / horizon
        sa_rate = (sa_sample - x0.delta) / horizon

        # vehicle commands to follow trajectory
        cmds = VehicleCommands(acc=acc, ddelta=sa_rate)

        # initial values
        states = [x0]
        control_inputs = [cmds]
        next_state = x0

        for _ in range(1, mpg.param.n_steps + 1):
            next_state = mpg.vehicle_dynamics(next_state, cmds, float(mpg.param.dt))
            states.append(next_state)
            control_inputs.append(cmds)

        end_states_traj.append(next_state)
        controls_traj.append(control_inputs)

    return end_states_traj, controls_traj
