import random
from dataclasses import dataclass
from typing import Sequence

from commonroad.scenario.lanelet import LaneletNetwork
from dg_commons import PlayerName
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters, steering_constraint

# our imports:
from dg_commons.planning.motion_primitives import MotionPrimitivesGenerator, MPGParam
from dg_commons.sim.models.vehicle import VehicleState, VehicleModel
from decimal import Decimal
from dg_commons import logger, Timestamp, LinSpaceTuple
import math
from dg_commons.sim.models.model_utils import apply_full_acceleration_limits
from dg_commons.dynamics import BicycleDynamics
import numpy as np
from sympy import Mul, primitive
from dataclasses import dataclass
from decimal import Decimal
from itertools import product
from typing import List, Callable, Set, Optional
from dg_commons import logger, Timestamp, LinSpaceTuple
from dg_commons.planning.trajectory import Trajectory
from dg_commons.sim.models import vehicle_ligths
from dg_commons.sim.models.vehicle_ligths import LightsCmd
from networkx import MultiDiGraph


@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 0.2


class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    name: PlayerName
    goal: PlanningGoal
    sg: VehicleGeometry
    sp: VehicleParameters
    dt: Decimal
    lane_width: float

    def __init__(self):
        # feel free to remove/modify  the following
        self.params = Pdm4arAgentParams()
        self.myplanner = ()

    def on_episode_init(self, init_obs: InitSimObservations):
        """This method is called by the simulator only at the beginning of each simulation.
        Do not modify the signature of this method."""
        self.name = init_obs.my_name
        self.goal = init_obs.goal
        self.sg = init_obs.model_geometry
        self.sp = init_obs.model_params
        self.dt = init_obs.dg_scenario.scenario.dt
        self.lane_width = 2 * init_obs.goal.ref_lane.control_points[0].r

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """
        # TODO: default values for testing
        n_vel = 1
        steer_range = 0.3
        n_steer = 3
        n_steps = 10

        x = sim_obs.players["Ego"].state  # type: ignore
        bd = BicycleDynamics(self.sg, self.sp)
        mpg_params = MPGParam.from_vehicle_parameters(
            dt=Decimal(self.dt), n_steps=n_steps, n_vel=n_vel, n_steer=n_steer, vp=self.sp
        )
        mpg = MotionPrimitivesGenerator(param=mpg_params, vehicle_dynamics=bd.successor, vehicle_param=self.sp)
        end_states_traj, controls_traj = self.generate_primat(x0=x, mpg=mpg, steer_range=steer_range, n_steer=n_steer)

        plot_trajectory(x, end_states_traj)  # TODO: add lane boundaries for visualization
        # TODO: use end_states_traj to build graph

        return VehicleCommands(
            acc=controls_traj[0][0].acc, ddelta=controls_traj[0][0].ddelta, lights=LightsCmd("turn_left")
        )

    def generate_primat(
        self,
        x0: VehicleState,
        mpg: MotionPrimitivesGenerator,
        steer_range: float,
        n_steer: int,
    ) -> tuple[List[VehicleState], List[List[VehicleCommands]]]:
        """
        Reimplement method mpg.generate(x) to generate motion primitives specifically for our problem
        """
        end_states_traj = []
        controls_traj = []

        # create samples in user-defined range - assume constant velocity and variable steering angles
        v_samples = np.array([x0.vx])
        sa_samples = np.linspace(
            *(x0.delta - steer_range, x0.delta + steer_range, n_steer)
        )  # TODO: need to check if steering angle is valid

        # n = len(v_samples) * len(sa_samples)
        # print(f"Attempting to generate {n} motion primitives")

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

    def generate_graph() -> MultiDiGraph:
        pass


### BACKUP ###
import matplotlib.pyplot as plt
import os


def plot_trajectory(state: VehicleState, end_states_traj: List[VehicleState]):
    for end_state_traj in end_states_traj:
        plt.plot(state.x, state.y, marker="o", linestyle="-", color="r")
        plt.plot(end_state_traj.x, end_state_traj.y, marker="o", linestyle="-", color="b")

    # Plot the trajectory
    output_dir = "/tmp"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"trajectory.png")
    plt.grid()
    plt.savefig(filename)
    plt.close()
    print(f"Image saved to {filename} \n")
