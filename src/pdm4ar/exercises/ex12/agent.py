import random
from dataclasses import dataclass
from tracemalloc import start
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
from matplotlib import markers
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
from networkx import DiGraph
from shapely.geometry import Polygon
import time
from matplotlib.collections import LineCollection

from .graph import WeightedGraph
from .A_star import Astar
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
        self.goal = init_obs.goal
        self.vg = init_obs.model_geometry
        self.vp = init_obs.model_params
        self.dt = init_obs.dg_scenario.scenario.dt

        # additional class variables
        self.lanelet_polygons = init_obs.dg_scenario.lanelet_network.lanelet_polygons
        self.lanelet_network = init_obs.dg_scenario.lanelet_network
        self.half_lane_width = init_obs.goal.ref_lane.control_points[0].r
        # self.lane_orientation = init_obs.goal.ref_lane.control_points[
        #     0
        # ].q.theta  # note: the lane is not entirely straight
        self.further_initailization = True  # boolean to further initialize parameters in the first call of get_commands

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """
        # TODO: default values for testing

        # get current lane by using the current position
        current_pos = np.array([sim_obs.players[self.name].state.x, sim_obs.players[self.name].state.y])
        try:
            self.current_lanelet_id = self.lanelet_network.find_lanelet_by_position([current_pos])[0][0]
        except IndexError:
            print("No lanelet found or out of bounds")

        current_state = sim_obs.players["Ego"].state  # type: ignore

        if self.further_initailization:
            self.n_vel = 1
            self.steer_range = 0.3
            self.n_steer = 3
            self.n_steps = 5
            self.lane_orientation = current_state.psi # assuming that lane orientation == initial orientation vehicle
            self.further_initailization = False

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
        start = time.time()
        weighted_graph = generate_graph(
            current_state,
            end_states_traj,
            controls_traj,
            depth,
            self.lanelet_network,
            self.half_lane_width,
            self.lane_orientation,
        )
        end = time.time()
        print(f"Generating the graph took {end - start} seconds.")

        start = time.time()
        plot_adj_list(weighted_graph.adj_list, self.lanelet_polygons)
        end = time.time()
        print(f"Plotting the graph took {end - start} seconds.")

        # astar_solver = Astar.path(graph=weighted_graph)
        # TODO: need to define finite_horizon_goal
        # shortest_path = astar_solver.path(start=current_state, goal=finite_horizon_goal, lanelet_network=self.lanelet_network, lanelet_id=self.current_lanelet_id)

        return VehicleCommands(
            acc=controls_traj[0][0].acc, ddelta=controls_traj[0][0].ddelta, lights=LightsCmd("turn_left")
        )


### ADDITIONAL HELPER FUNCTIONS ###
import matplotlib.pyplot as plt
import os


def plot_adj_list(adj_list, lanelet_polygons):
    plt.figure(figsize=(30, 25), dpi=250)
    ax = plt.gca()

    # Collect all node coordinates
    node_coords = []
    for parent in adj_list.keys():
        parent_x, parent_y = parent[1], parent[2]
        node_coords.append((parent_x, parent_y))

    # Collect all edge coordinates
    edge_coords = []
    for parent, children in adj_list.items():
        parent_x, parent_y = parent[1], parent[2]
        for child in children:
            child_x, child_y = child[1], child[2]
            edge_coords.append([(parent_x, parent_y), (child_x, child_y)])

    # Plot all nodes at once
    node_coords = np.array(node_coords)
    plt.scatter(node_coords[:, 0], node_coords[:, 1], color="blue", s=1)

    # Plot all edges at once using LineCollection
    edge_collection = LineCollection(edge_coords, colors="blue", linewidths=0.3)
    ax.add_collection(edge_collection)

    # Plot static obstacles (from on_episode_init)
    for lanelet in lanelet_polygons:
        x, y = lanelet.shapely_object.exterior.xy
        plt.plot(x, y, linestyle="-", linewidth=0.8, color="darkorchid")

    ax.set_aspect("equal", adjustable="box")

    output_dir = "/tmp"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "graph.png")
    plt.savefig(filename, bbox_inches="tight")  # Save the plot with tight bounding box
    plt.close()
    print(f"Graph saved to {filename}")