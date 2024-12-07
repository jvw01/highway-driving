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

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """
        x = sim_obs.players["Ego"].state  # type: ignore
        mpg_params = MPGParam.from_vehicle_parameters(Decimal(self.dt), 10, 5, 5, self.sp)
        mpg = MotionPrimitivesGenerator(mpg_params, self.get_next_state, self.sp)
        # cast x here to vehicle state
        mp = mpg.generate()

        # todo implement here some better planning
        rnd_acc = random.random() * self.params.param1
        rnd_ddelta = (random.random() - 0.5) * self.params.param1

        return VehicleCommands(acc=rnd_acc, ddelta=rnd_ddelta)

    def get_next_state(self, x0: VehicleState, u: VehicleCommands, dt: Timestamp) -> VehicleState:
        """Kinematic bicycle model, returns state derivative for given control inputs"""
        vx = x0.vx
        dtheta = vx * math.tan(x0.delta) / self.sg.wheelbase
        vy = dtheta * self.sg.lr
        costh = math.cos(x0.psi)
        sinth = math.sin(x0.psi)
        xdot = vx * costh - vy * sinth
        ydot = vx * sinth + vy * costh

        ddelta = steering_constraint(x0.delta, u.ddelta, self.sp)
        acc = apply_full_acceleration_limits(x0.vx, u.acc, self.sp)

        t = float(dt)

        return VehicleState(
            x=x0.x + xdot * t,
            y=x0.y + ydot * t,
            psi=x0.psi + dtheta * t,
            vx=x0.vx + t * acc,
            delta=x0.delta + t * ddelta,
        )
