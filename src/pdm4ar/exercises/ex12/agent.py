from mimetypes import init
import random
from dataclasses import dataclass
from tracemalloc import start
from typing import Sequence

from commonroad.scenario.lanelet import LaneletNetwork
from cvxopt import normal
from cycler import K
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
import frozendict
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
from .graph import WeightedGraph
from .A_star import Astar
import time
from matplotlib.collections import LineCollection
from dg_commons.controllers.pid import PID
from dg_commons.controllers.steer import SteerController
from dg_commons.controllers.pure_pursuit import PurePursuit
from geometry.types import SE2value, E2value

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

    # parameters for the velocity controler:
    d_ref: float = 3
    T: float = 8
    init_abstand: bool = False
    last_e: float

    # parameters for the steering controller:
    K_psi: float = 1
    K_dist: float = 0.1
    K_delta: float = 2
    # okish results for P-only: k_psi=1, k_dist=0.1, k_delta=2
    K_d_psi: float = 0.5  # tried: 0.1, 0.5, already 0.1 helps to stabilize
    K_d_dist: float = 0.1
    K_d_delta: float = 0.1

    pure_pursuit: PurePursuit
    steer_controller: SteerController
    last_dpsi: float
    last_dist: float
    last_delta: float
    init_control: bool = False

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
        self.lane_width = 2 * init_obs.goal.ref_lane.control_points[0].r
        self.steer_controller = SteerController.from_vehicle_params(vehicle_param=self.vp)

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
        n_steps = 5

        current_state = sim_obs.players["Ego"].state  # type: ignore
        # bd = BicycleDynamics(self.vg, self.vp)
        # mpg_params = MPGParam.from_vehicle_parameters(
        #     dt=Decimal(self.dt), n_steps=n_steps, n_vel=n_vel, n_steer=n_steer, vp=self.vp
        # )
        # mpg = MotionPrimitivesGenerator(param=mpg_params, vehicle_dynamics=bd.successor, vehicle_param=self.vp)
        # end_states_traj, controls_traj = generate_primat(
        #     x0=current_state, mpg=mpg, steer_range=steer_range, n_steer=n_steer
        # )

        # # build graph
        # depth = 8  # TODO: default value - need to decide how deep we want our graph to be
        # start = time.time()
        # weighted_graph = generate_graph(current_state, end_states_traj, controls_traj, depth, self.lanelet_network)
        # end = time.time()
        # print(f"Generating the graph took {end - start} seconds.")

        # start = time.time()
        # plot_adj_list(weighted_graph.adj_list, self.lanelet_polygons)
        # end = time.time()
        # print(f"Plotting the graph took {end - start} seconds.")

        # # astar_solver = Astar.path(graph=weighted_graph)
        # # TODO: need to define finite_horizon_goal
        # # shortest_path = astar_solver.path(start=current_state, goal=finite_horizon_goal)

        # return VehicleCommands(
        #     acc=controls_traj[0][0].acc, ddelta=controls_traj[0][0].ddelta, lights=LightsCmd("turn_left")
        # )
        # if float(sim_obs.time) < 0.7:
        #     return VehicleCommands(acc=0.0, ddelta=-0.3)
        # return VehicleCommands(
        #     acc=0.0, ddelta=self.spurassistehaltent(current_state, self.K_psi, self.K_dist, self.K_delta)
        # )  # , lights=LightsCmd("turn_left"))
        acc = self.abstandhalteassistent(current_state, sim_obs.players)
        ddelta = self.spurhalteassistent(current_state, float(sim_obs.time))
        return VehicleCommands(acc, ddelta)  # self.spurhalteassistent(current_state, float(sim_obs.time)))

    def spurhalteassistent(self, current_state: VehicleState, t: float) -> float:
        cur_lanelet_id = self.lanelet_network.find_lanelet_by_position([np.array([current_state.x, current_state.y])])
        if not cur_lanelet_id[0]:
            print("No lanelet found")
            return 0.0
        cur_lanelet = self.lanelet_network._lanelets[cur_lanelet_id[0][0]]
        center_vertices = cur_lanelet.center_vertices
        lanelet_heading = math.atan2(
            center_vertices[-1][1] - center_vertices[0][1], center_vertices[-1][0] - center_vertices[0][0]
        )  # should not be necessary, just to prevent numerical drift
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
            return (
                -1 * dpsi - 0.1 * dist - 2 * current_state.delta  # hardcoded to tune the PD controller
            )  # -self.K_psi * dpsi - self.K_dist * dist - self.K_delta * current_state.delta
        d_dist = (dist - self.last_dist) / float(self.dt)
        d_dpsi = (dpsi - self.last_dpsi) / float(self.dt)
        d_delta = (current_state.delta - self.last_delta) / float(self.dt)
        self.last_dist = dist
        self.last_dpsi = dpsi
        self.last_delta = current_state.delta

        ddelta = (
            -self.K_psi * dpsi
            - self.K_d_psi * d_dpsi
            - self.K_dist * dist
            - self.K_d_dist * d_dist
            - self.K_delta * current_state.delta
            - self.K_d_delta * d_delta
        )
        if abs(dist) < 0.05 and abs(current_state.delta) < 0.01:
            self.steer_controller.update_measurement(measurement=current_state.delta)
            self.steer_controller.update_reference(reference=0)
            ddelta = self.steer_controller.get_control(t)
            return min(max(ddelta, -self.vp.ddelta_max), self.vp.ddelta_max)
        # print(ddelta)
        return ddelta

    def abstandhalteassistent(self, current_state: VehicleState, players: frozendict) -> float:
        player_ahead = None
        for player in players:
            if player != self.name:
                diff_angle = np.arctan2(
                    players[player].state.y - current_state.y, players[player].state.x - current_state.x
                )
                if abs(diff_angle - current_state.psi) < 0.04:
                    print(player)
                    player_ahead = players[player]

                    # player_lanelet_id = self.lanelet_network.find_lanelet_by_position([np.array([player.state.x, player.state.y])])
                    # if player_lanelet_id[0][0]==self.lanelet_network.find_lanelet_by_position([np.array([current_state.x, current_state.y])])[0][0]:
                    #     play
        if player_ahead is None:
            return self.gib_ihm(current_state)
        else:
            dist_to_player = math.sqrt(
                (player_ahead.state.x - current_state.x) ** 2 + (player_ahead.state.y - current_state.y) ** 2
            )
            e = dist_to_player - self.d_ref
            if not self.init_abstand:
                self.init_abstand = True
                self.last_e = e
                return (player_ahead.state.vx / self.T + 3) * (
                    self.last_e
                )  # P-controler for first time step, here last_e is also the current error
            de = (e - self.last_e) / float(self.dt)
            self.last_e = e
            K_p = player_ahead.state.vx / self.T + 3
            acc = K_p * (self.T * de + e)

            return acc

    def gib_ihm(self, current_state: VehicleState) -> float:
        """ "Probably delete this function and integrate it into abstandhalteassistent for final version, thought it was funny"""
        if current_state.vx >= 24.5:
            return 0.0
        return self.vp.acc_limits[1]


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
    filename = os.path.join(output_dir, "adj_list_plot.png")
    plt.savefig(filename, bbox_inches="tight")  # Save the plot with tight bounding box
    plt.close()
    print(f"Graph saved to {filename}")
