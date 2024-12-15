from dg_commons.sim.models.vehicle import VehicleState
from typing import List
import numpy as np
from dg_commons.sim.models.vehicle import VehicleCommands
from itertools import product
from dg_commons.planning.motion_primitives import MotionPrimitivesGenerator
from dataclasses import replace
import math
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters


# note: the code for the calculation of the next state / motion primitives is taken from the dg.commons library


def generate_primat(
    x0: VehicleState,
    steer_range: float,
    goal_lane: str,
    case: int,
    vp: VehicleGeometry,
    vg: VehicleParameters,
    dt: float,
    n_steps: int,
    scaling_dt: float,
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

        horizon = dt * n_steps

        for v_sample, sa_sample in product(v_samples, sa_samples):
            # calculate acceleration and steering angle rate
            acc = (v_sample - x0.vx) / horizon
            sa_rate = (sa_sample - x0.delta) / horizon
            check_input_constraints(acc, sa_rate, vp)

            # vehicle commands to follow trajectory
            cmds = VehicleCommands(acc=acc, ddelta=sa_rate)

            # initial values
            control_inputs = int(n_steps / scaling_dt) * [cmds]
            next_state = x0

            for _ in range(1, n_steps + 1):
                next_state = calc_next_state(next_state, cmds, dt, vp, vg)
                # control_inputs.append(cmds)

            end_states_traj.append(next_state)
            controls_traj.append(control_inputs)

    if case == 1:
        # calculate acceleration and steering angle rate
        acc = 0
        horizon = dt * n_steps
        # sa_rate = (sa_sample - x0.delta) / horizon

        if goal_lane == "left":
            sa_rate = steer_range / horizon
        else:
            sa_rate = -steer_range / horizon

        check_input_constraints(acc, sa_rate, vp)

        # vehicle commands to follow trajectory
        cmds = VehicleCommands(acc=acc, ddelta=sa_rate)

        # initial values
        control_inputs = int(n_steps / scaling_dt) * [cmds]
        next_state = x0

        for _ in range(1, n_steps + 1):
            next_state = calc_next_state(next_state, cmds, dt, vp, vg)
            # control_inputs.append(cmds)

        end_states_traj.append(next_state)
        controls_traj.append(control_inputs)

    if case == 2:
        # calculate acceleration and steering angle rate
        acc = 0
        horizon = dt * n_steps
        # sa_rate = (sa_sample - x0.delta) / horizon

        if goal_lane == "left":
            sa_rate = -steer_range / horizon
        else:
            sa_rate = steer_range / horizon

        check_input_constraints(acc, sa_rate, vp)

        # vehicle commands to follow trajectory
        cmds = VehicleCommands(acc=acc, ddelta=sa_rate)

        # initial values
        control_inputs = int(n_steps / scaling_dt) * [cmds]
        next_state = x0

        for _ in range(1, n_steps + 1):
            next_state = calc_next_state(next_state, cmds, dt, vp, vg)
            # control_inputs.append(cmds)

        end_states_traj.append(next_state)
        controls_traj.append(control_inputs)

    return end_states_traj, controls_traj


def calc_next_state(x0: VehicleState, u: VehicleCommands, dt: float, vp, vg) -> VehicleState:
    """Perform Euler forward integration to propagate state using actions for time dt.
    This method is very inaccurate for integration steps above 0.1[s]"""
    # input constraints
    acc = float(np.clip(u.acc, vp.acc_limits[0], vp.acc_limits[1]))
    ddelta = float(np.clip(u.ddelta, -vp.ddelta_max, vp.ddelta_max))

    state_rate = dynamics(x0, replace(u, acc=acc, ddelta=ddelta), vg)
    x0 += state_rate * dt

    # state constraints
    vx = float(np.clip(x0.vx, vp.vx_limits[0], vp.vx_limits[1]))
    delta = float(np.clip(x0.delta, -vp.delta_max, vp.delta_max))

    new_state = replace(x0, vx=vx, delta=delta)
    return new_state


def dynamics(x0: VehicleState, u: VehicleCommands, vg) -> VehicleState:
    """Get rate of change of states for given control inputs"""

    dx = x0.vx
    dtheta = dx * math.tan(x0.delta) / vg.wheelbase
    dy = dtheta * vg.lr
    costh = math.cos(x0.psi)
    sinth = math.sin(x0.psi)
    xdot = dx * costh - dy * sinth
    ydot = dx * sinth + dy * costh
    x_rate = VehicleState(x=xdot, y=ydot, psi=dtheta, vx=u.acc, delta=u.ddelta)
    return x_rate


def check_input_constraints(acc, sa_rate, vp):
    if not (-vp.ddelta_max <= sa_rate <= vp.ddelta_max):
        print("ddelta input constraint violated")
    if not (vp.acc_limits[0] <= acc <= vp.acc_limits[1]):
        print("acc input constraint violated")


### BACKUP ###
# def generate_primat_curve(
#     x0: VehicleState,
#     mpg: MotionPrimitivesGenerator,
# ) -> tuple[list, List[VehicleCommands]]:
#     """
#     Reimplement method mpg.generate(x) to generate motion primitives specifically for our problem.
#     """
#     # vehicle commands to follow trajectory - continue with same velocity and steering angle
#     cmds = VehicleCommands(acc=0, ddelta=0)

#     # initial values
#     control_inputs = [cmds]
#     next_state = x0

#     for _ in range(1, mpg.param.n_steps + 1):
#         next_state = mpg.vehicle_dynamics(next_state, cmds, float(mpg.param.dt))
#         control_inputs.append(cmds)

#     delta = [next_state.x - x0.x, next_state.y - x0.y, next_state.psi - x0.psi, next_state.delta - x0.delta]

#     return delta, control_inputs
