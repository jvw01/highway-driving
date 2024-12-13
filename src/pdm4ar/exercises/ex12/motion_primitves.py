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
    goal_lane: str,
    case: int,
) -> tuple[List[VehicleState], List[List[VehicleCommands]]]:
    """
    Reimplement method mpg.generate(x) to generate motion primitives specifically for our problem.
    """
    end_states_traj = []
    controls_traj = []

    if case == 0:
        # create samples in user-defined range - assume constant velocity and variable steering angles
        v_samples = np.array([x0.vx])
        if goal_lane == "left":
            sa_samples = [x0.delta, x0.delta + steer_range]
        else:
            sa_samples = [x0.delta, x0.delta - steer_range]

        horizon = float(mpg.param.dt * mpg.param.n_steps)

        for v_sample, sa_sample in product(v_samples, sa_samples):
            # calculate acceleration and steering angle rate
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

    if case == 1 or case == 2:
        if goal_lane == "left":
            sa_sample = x0.delta - steer_range
        else:
            sa_sample = x0.delta + steer_range

        # calculate acceleration and steering angle rate
        acc = 0
        horizon = float(mpg.param.dt * mpg.param.n_steps)
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

    if case == 3:
        if goal_lane == "left":
            sa_sample = x0.delta + steer_range
        else:
            sa_sample = x0.delta - steer_range

        # calculate acceleration and steering angle rate
        acc = 0
        horizon = float(mpg.param.dt * mpg.param.n_steps)
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


def generate_primat_curve(
    x0: VehicleState,
    mpg: MotionPrimitivesGenerator,
) -> tuple[list, List[VehicleCommands]]:
    """
    Reimplement method mpg.generate(x) to generate motion primitives specifically for our problem.
    """
    # vehicle commands to follow trajectory - continue with same velocity and steering angle
    cmds = VehicleCommands(acc=0, ddelta=0)

    # initial values
    control_inputs = [cmds]
    next_state = x0

    for _ in range(1, mpg.param.n_steps + 1):
        next_state = mpg.vehicle_dynamics(next_state, cmds, float(mpg.param.dt))
        control_inputs.append(cmds)

    delta = [next_state.x - x0.x, next_state.y - x0.y, next_state.psi - x0.psi, next_state.delta - x0.delta]

    return delta, control_inputs
