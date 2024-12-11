from hmac import new
import random
from dataclasses import dataclass
from tracemalloc import start
from typing import Sequence

from commonroad.scenario.lanelet import LaneletNetwork
from dg_commons import PlayerName
from dg_commons.sim.goals import PlanningGoal, RefLaneGoal
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
from matplotlib import markers
import numpy as np
from shapely import linestrings
from sympy import Mul, primitive
from dataclasses import dataclass
from decimal import Decimal
from itertools import product
from typing import List, Callable, Set, Optional
from dg_commons import logger, Timestamp, LinSpaceTuple
from dg_commons.planning.trajectory import Trajectory
from dg_commons.sim.models import vehicle_ligths
from dg_commons.sim.models.vehicle_ligths import LightsCmd
from networkx import DiGraph
from shapely.geometry import Polygon
import time
from matplotlib.collections import LineCollection
from shapely.geometry import LineString

from .graph import WeightedGraph
from .A_star import Astar
import time

from .motion_primitves import generate_primat
from .graph import generate_graph


@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 0.2


class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    name: PlayerName
    goal: PlanningGoal
    vg: VehicleGeometry
    vp: VehicleParameters
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
        self.goal = init_obs.goal  # Attention: calling the attributes of RefLaneGoal works but pylance gives an error  # type: ignore
        self.vg = init_obs.model_geometry  # type: ignore
        self.vp = init_obs.model_params  # type: ignore
        self.dt = init_obs.dg_scenario.scenario.dt  # type: ignore

        # self.goal_lines = self.define_goal_points()

        # additional class variables
        # self.lanelet_polygons = init_obs.dg_scenario.lanelet_network.lanelet_polygons  # type: ignore
        self.lanelet_network = init_obs.dg_scenario.lanelet_network  # type: ignore
        self.half_lane_width = init_obs.goal.ref_lane.control_points[1].r  # type: ignore
        self.goal_id = init_obs.dg_scenario.lanelet_network.find_lanelet_by_position([init_obs.goal.ref_lane.control_points[1].q.p])[0][0]  # type: ignore
        # self.lane_orientation = init_obs.goal.ref_lane.control_points[
        #     0
        # ].q.theta  # note: the lane is not entirely straight
        self.further_initialization = True  # boolean to further initialize parameters in the first call of get_commands
        self.recompute = True  # boolean value that indicates if the graph needs to be recomputed
        self.goal_lane_ids = self.retrieve_goal_lane_ids()

        #########
        # static obstacles (contains a LineString, m, I_z and e)
        self.static_obs = init_obs.dg_scenario.static_obstacles
        # print(self.static_obs)
        #########

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """
        if self.recompute:

            # get current lane by using the current position
            current_pos = np.array([sim_obs.players[self.name].state.x, sim_obs.players[self.name].state.y])
            try:
                self.current_lanelet_id = self.lanelet_network.find_lanelet_by_position([current_pos])[0][0]
            except IndexError:
                print("No lanelet found or out of bounds")

            current_state = sim_obs.players["Ego"].state  # type: ignore

            if self.further_initialization:
                # TODO: default values for testing
                self.n_vel = 1
                self.steer_range = 0.3
                self.n_steer = 3
                self.n_steps = 5
                self.lane_orientation = (
                    current_state.psi
                )  # assuming that lane orientation == initial orientation vehicle
                self.further_initialization = False

            bd = BicycleDynamics(self.vg, self.vp)
            mpg_params = MPGParam.from_vehicle_parameters(
                dt=Decimal(self.dt), n_steps=self.n_steps, n_vel=self.n_vel, n_steer=self.n_steer, vp=self.vp
            )
            mpg = MotionPrimitivesGenerator(param=mpg_params, vehicle_dynamics=bd.successor, vehicle_param=self.vp)
            end_states_traj, controls_traj = generate_primat(
                x0=current_state, mpg=mpg, steer_range=self.steer_range, n_steer=self.n_steer
            )

            # build graph
            depth = 8  # TODO: default value - need to decide how deep we want our graph to be
            current_occupancy = sim_obs.players["Ego"].occupancy  # type: ignore
            dyn_obs_current = []
            for player in sim_obs.players:
                if player != "Ego":
                    dyn_obs_current.append(
                        (
                            sim_obs.players[player].state.x,  # type: ignore
                            sim_obs.players[player].state.y,  # type: ignore
                            sim_obs.players[player].state.vx,  # type: ignore
                            sim_obs.players[player].occupancy,
                        )
                    )

            # test1 = (dyn_obs_current[1][0], dyn_obs_current[1][1])
            # test2 = self.propagate_state(dyn_obs_current[1][0], dyn_obs_current[1][1], dyn_obs_current[1][2], depth)
            # plot_other_cars(test1, test2, self.lanelet_network.lanelet_polygons)

            weighted_graph = generate_graph(
                current_state,
                end_states_traj,
                controls_traj,
                depth,
                self.lanelet_network,
                self.current_lanelet_id,
                self.half_lane_width,
                self.lane_orientation,
                self.goal_lane_ids,
            )

            # astar_solver = Astar(weighted_graph)
            # shortest_path = astar_solver.path(start=current_state, goal=self.goal_lines)

            # TODO do stuff with shortest path

            self.recompute = False

        # return VehicleCommands(
        #     acc=controls_traj[0][0].acc, ddelta=controls_traj[0][0].ddelta, lights=LightsCmd("turn_left")
        # )

        rnd_acc = random.random() * self.params.param1
        rnd_ddelta = (random.random() - 0.5) * self.params.param1

        return VehicleCommands(acc=rnd_acc, ddelta=rnd_ddelta)

    def retrieve_goal_lane_ids(self) -> list:
        goal_lane_ids = [self.goal_id]
        lanelet = self.lanelet_network.find_lanelet_by_id(goal_lane_ids[-1])
        predecessor_id = lanelet.predecessor
        while predecessor_id:
            goal_lane_ids.append(predecessor_id[0])
            predecessor_id = self.lanelet_network.find_lanelet_by_id(predecessor_id[0]).predecessor

        return goal_lane_ids

    def propagate_state(self, x_pos: float, y_pos: float, vx: float, depth: int) -> list:
        propagated_states = []
        time_horizon = self.n_steps * float(self.dt)
        s = vx * time_horizon  # note: cars do not change lanes
        for i in range(depth):
            propagated_states.append(
                (x_pos + i * s * math.cos(self.lane_orientation), y_pos + i * s * math.sin(self.lane_orientation))
            )

        return propagated_states

    # # function that takes the goal and creates a list of shapely lines that mark the goal lane
    # def define_goal_lines(self) -> List[LineString]:

    #     line_segments = []
    #     current_line_points = []
    #     previous_theta = None

    #     control_points = self.goal.ref_lane.control_points
    #     for idx, centerline_point in enumerate(control_points):
    #         # as long as theta is the same, create a shapely line connecting them
    #         centerpoint = tuple(centerline_point.q.p)
    #         theta = centerline_point.q.theta

    #         if idx == 0:
    #             current_line_points.append(centerpoint)
    #             previous_theta = theta
    #         else:
    #             if theta != previous_theta:
    #                 if len(current_line_points) > 1:
    #                     line_segments.append(LineString(current_line_points))
    #                 current_line_points = [current_line_points[-1], centerpoint]

    #             else:
    #                 current_line_points.append(centerpoint)
    #             previous_theta = theta

    #     if len(current_line_points) > 1:
    #         line_segments.append(LineString(current_line_points))

    #     print(control_points)
    #     print(line_segments)

    #     return line_segments


### ADDITIONAL HELPER FUNCTIONS ###
import matplotlib.pyplot as plt
import os
import time


def plot_other_cars(init_pos, future_pos, lanelet_polygons):
    plt.figure(figsize=(30, 25), dpi=250)
    ax = plt.gca()

    # Plot static obstacles (from on_episode_init)
    for lanelet in lanelet_polygons:
        x, y = lanelet.shapely_object.exterior.xy
        plt.plot(x, y, linestyle="-", linewidth=0.8, color="darkorchid")

    plt.scatter(init_pos[0], init_pos[1], color="red", s=10)
    for pos in future_pos:
        plt.scatter(pos[0], pos[1], color="red", s=10)

    ax.set_aspect("equal", adjustable="box")

    output_dir = "/tmp"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "other_cars.png")
    plt.savefig(filename, bbox_inches="tight")  # Save the plot with tight bounding box
    plt.close()
    print(f"Graph saved to {filename}")
