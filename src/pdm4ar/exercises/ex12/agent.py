from hmac import new
from mimetypes import init
from dataclasses import dataclass
from re import S
from sre_parse import State
from tracemalloc import start
from typing import Sequence

from commonroad.scenario.lanelet import LaneletNetwork
from cvxopt import normal
from cycler import K
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
from dg_commons.sim.models.model_utils import apply_full_acceleration_limits
from dg_commons.dynamics import BicycleDynamics
import frozendict
from matplotlib import markers
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
from matplotlib.collections import LineCollection
from shapely.geometry import LineString
from dg_commons.controllers.pid import PID
from dg_commons.controllers.steer import SteerController
from dg_commons.controllers.pure_pursuit import PurePursuit
from geometry.types import SE2value, E2value
import numpy as np
import math
import time
import random
import enum
from shapely.affinity import translate, rotate
from shapely.strtree import STRtree


from .graph import WeightedGraph, generate_graph
from .motion_primitves import generate_primat
from .astar import Astar

class StateMachine(enum.Enum):
    INITIALIZATION = 1
    ATTEMPT_LANE_CHANGE = 2
    EXECUTE_LANE_CHANGE = 3
    HOLD_LANE = 4
    GOAL_STATE = 5


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
        self.goal = init_obs.goal  # type: ignore
        self.vg = init_obs.model_geometry  # type: ignore
        self.vp = init_obs.model_params  # type: ignore
        self.dt = init_obs.dg_scenario.scenario.dt  # type: ignore

        # additional class variables
        self.dt_integral = 0.01  # time step for integral part of motion primitives
        self.scaling_dt = 0.1 / self.dt_integral  # scaling factor for get_commands frequency (0.1s)

        self.lanelet_network = init_obs.dg_scenario.lanelet_network  # type: ignore
        self.half_lane_width = init_obs.goal.ref_lane.control_points[1].r  # type: ignore
        self.goal_id = init_obs.dg_scenario.lanelet_network.find_lanelet_by_position([init_obs.goal.ref_lane.control_points[1].q.p])[0][0]  # type: ignore
        # self.lane_orientation = init_obs.goal.ref_lane.control_points[
        #     0
        # ].q.theta  # note: the lane is not entirely straight
        self.state = StateMachine.INITIALIZATION
        self.goal_lane_ids = self.retrieve_goal_lane_ids()
        self.steer_controller = SteerController.from_vehicle_params(vehicle_param=self.vp)
        self.freq_counter = 0
        self.shortest_path = []

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """
        current_state = sim_obs.players["Ego"].state  # type: ignore

        if self.state == StateMachine.INITIALIZATION:
            self.lane_orientation = sim_obs.players[
                "Ego"
            ].state.psi  # type: ignore # assuming that lane orientation == initial orientation vehicle
            # check whether goal lane is left or right of current lane
            lanelet_id = self.lanelet_network.find_lanelet_by_position([np.array([current_state.x, current_state.y])])[
                0
            ][0]
            lanelet = self.lanelet_network.find_lanelet_by_id(lanelet_id)
            if lanelet.adj_left in self.goal_lane_ids:
                self.goal_lane = "left"
            else:
                self.goal_lane = "right"
            self.state = StateMachine.ATTEMPT_LANE_CHANGE

        if self.state == StateMachine.ATTEMPT_LANE_CHANGE:
            # need 4 motion primitives to change lanes - calculate the time horizon for one motion primitive
            self.time_horizon = 0.9  # time horizon for one motion primitive; must be multiple of 0.1s, TODO: choose value depending on velocity; 0.5 for 2nd TC in config1
            self.n_steps = int(self.time_horizon / self.dt_integral)
            self.delta_steer = self.vp.ddelta_max * self.time_horizon

            end_states = []
            control_inputs = []
            state = current_state

            steps_lane_change = 4  # needs to be multiple of 4
            for i in range(steps_lane_change):
                if i == 0:
                    case = 0
                else:
                    multiples = steps_lane_change / 4

                    if i < multiples:
                        case = 1
                    elif i < 3 * multiples:
                        case = 2
                    else:
                        case = 1

                end_state, control_input = generate_primat(
                    x0=state,
                    steer_range=self.delta_steer,
                    goal_lane=self.goal_lane,
                    case=case,
                    vp=self.vp,
                    vg=self.vg,
                    dt=self.dt_integral,
                    n_steps=self.n_steps,
                    scaling_dt=self.scaling_dt,
                )

                end_states.append(end_state)
                control_inputs.append(control_input)

                if i == steps_lane_change - 1:  # to avoid unncecessary computations
                    break

                dx = end_state[-1].x - current_state.x
                dy = end_state[-1].y - current_state.y
                dpsi = end_state[-1].psi - current_state.psi
                delta = end_state[-1].delta - current_state.delta

                delta_pos = np.array(
                    [
                        dx * np.cos(current_state.psi - self.lane_orientation)
                        - dy * np.sin(current_state.psi - self.lane_orientation),
                        dy * np.cos(current_state.psi - self.lane_orientation)
                        + dx * np.sin(current_state.psi - self.lane_orientation),
                    ]
                )

                state = VehicleState(
                    x=current_state.x + delta_pos[0],
                    y=current_state.y + delta_pos[1],
                    psi=current_state.psi + dpsi,
                    vx=current_state.vx,
                    delta=current_state.delta + delta,
                )

            # n_steer = 1
            # n_vel = 1
            # bd = BicycleDynamics(self.vg, self.vp)
            # mpg_params = MPGParam.from_vehicle_parameters(
            #     dt=Decimal(self.dt), n_steps=self.n_steps, n_vel=n_vel, n_steer=n_steer, vp=self.vp
            # )
            # mpg = MotionPrimitivesGenerator(param=mpg_params, vehicle_dynamics=bd.successor, vehicle_param=self.vp)
            # test = mpg.generate(x0=current_state)

            # plot_end_states(end_states, self.lanelet_network.lanelet_polygons)

            # build graph
            self.depth = 9  # note: has to be greater than steps_lane_change! TODO: default value - need to decide how deep we want our graph to be
            start = time.time()
            weighted_graph = generate_graph(
                current_state=current_state,
                end_states_traj=end_states,
                controls_traj=control_inputs,
                depth=self.depth,
                lanelet_network=self.lanelet_network,
                goal_id=self.goal_lane_ids,
                steps_lane_change=steps_lane_change,
            )
            end = time.time()
            print(f"Generating the graph took {end - start} seconds.")

            plot_graph(weighted_graph.graph, self.lanelet_network.lanelet_polygons, [])

            # extract current state of dynamic obstacles to be able to propagate them for collision checking
            start_other_cars = time.time()
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

            # create rtree for dynamic obstacles at each level of the graph (i.e. prepare efficient search for collision checking)
            states_dyn_obs = self.states_other_cars(dyn_obs_current)
            end_other_cars = time.time()
            print(f"Generating the dynamic obstacles took {end_other_cars - start_other_cars} seconds.")

            # find shortest path with A*
            start_astar = time.time()
            for _ in range(weighted_graph.num_goal_nodes):
                astar_solver = Astar(weighted_graph)
                self.shortest_path = astar_solver.path(
                    start_node=weighted_graph.start_node, goal_node=weighted_graph.goal_node
                )
                # collision checking on shortest path with RTree for every time step
                if self.has_collision(
                    shortest_path=self.shortest_path, states_other_cars=states_dyn_obs, occupancy=current_occupancy
                ):
                    print("Collision detected. Recomputing shortest path.")
                    # eliminate nodes and edges of the shortest path from the graph
                    edges_to_remove = [
                        (self.shortest_path[i], self.shortest_path[i + 1]) for i in range(0, len(self.shortest_path) - 1)
                    ]
                    weighted_graph.graph.remove_edges_from(edges_to_remove)
                    weighted_graph.graph.remove_nodes_from(shortest_path[1:-1])  # exclude start and goal node
                    # Update adjacency list and weights
                    for u, v in edges_to_remove:
                        if u in weighted_graph.adj_list:
                            weighted_graph.adj_list[u].discard(v)
                    shortest_path = []
                    continue
                else:
                    # a path without collision was found
                    print("Found path without collision.")
                    break

            end_astar = time.time()
            print(f"A* took {end_astar - start_astar} seconds.")

            if self.shortest_path:
                self.state = StateMachine.EXECUTE_LANE_CHANGE
                self.num_steps_path = len(self.shortest_path) - 1  # exclude virtual goal node
                self.path_node = 0

            else:
                self.state = StateMachine.HOLD_LANE

            # self.plot_collisions(
            #     states_dyn_obs, self.lanelet_network.lanelet_polygons, self.shortest_path, dyn_obs_current, current_occupancy
            # )

            # start_plot = time.time()
            plot_graph(weighted_graph.graph, self.lanelet_network.lanelet_polygons, self.shortest_path)
            # plot_graph(weighted_graph.graph, self.lanelet_network.lanelet_polygons, [])
            # end_plot = time.time()
            # print(f"Plotting the graph took {end_plot - start_plot} seconds.")

        if self.state == StateMachine.EXECUTE_LANE_CHANGE:  # case: A* found a path -> lane change
            idx = int(self.freq_counter % np.floor(self.n_steps / self.scaling_dt))

            # remain in EXECUTE_LANE_CHANGE state as we are inbetween graph nodes
            if idx != 0:
                self.freq_counter += 1
                return VehicleCommands(
                    acc=self.shortest_path[self.path_node][3][idx].acc, ddelta=self.shortest_path[self.path_node][3][idx].ddelta,
                )

            # new graph node reached, switch to HOLD_LANE if we're at the goal node, else we stay until the goal node is reached
            else:
                self.path_node += 1
                if self.path_node == self.num_steps_path:  # goal node reached
                    self.state = StateMachine.HOLD_LANE
                    acc = 0
                    ddelta = self.spurhalteassistent(current_state, float(sim_obs.time))  # type: ignore
                    self.freq_counter += 1
                    return VehicleCommands(acc, ddelta)  # self.spurhalteassistent(current_state, float(sim_obs.time)))
                else:
                    self.freq_counter += 1
                    return VehicleCommands(
                        acc=self.shortest_path[self.path_node][3][idx].acc, ddelta=self.shortest_path[self.path_node][3][idx].ddelta
                    ) 

        if self.state == StateMachine.HOLD_LANE:  # default case: continue on lane
            # check if car is on goal lane
            player_lanelet_id = self.lanelet_network.find_lanelet_by_position(
                [np.array([current_state.x, current_state.y])]
            )

            if player_lanelet_id[0][0] == self.goal_id:
                self.state = StateMachine.GOAL_STATE

            acc = 0  # type: ignore
            ddelta = self.spurhalteassistent(current_state, float(sim_obs.time))  # type: ignore
            self.freq_counter += 1
            return VehicleCommands(acc, ddelta)  # self.spurhalteassistent(current_state, float(sim_obs.time)))

        if self.state == StateMachine.GOAL_STATE:
            acc = 0  # type: ignore
            ddelta = self.spurhalteassistent(current_state, float(sim_obs.time))  # type: ignore
            self.freq_counter += 1
            return VehicleCommands(acc, ddelta)  # self.spurhalteassistent(current_state, float(sim_obs.time)))

    def retrieve_goal_lane_ids(self) -> list:
        goal_lane_ids = [self.goal_id]
        lanelet = self.lanelet_network.find_lanelet_by_id(goal_lane_ids[-1])
        predecessor_id = lanelet.predecessor
        while predecessor_id:
            goal_lane_ids.append(predecessor_id[0])
            predecessor_id = self.lanelet_network.find_lanelet_by_id(predecessor_id[0]).predecessor

        return goal_lane_ids

    def states_other_cars(self, dyn_obs_current: list) -> list:
        states_other_cars = []
        for i in range(1, self.depth):
            states_i = []
            for dyn_obs in dyn_obs_current:
                vx = dyn_obs[2]
                s = vx * self.time_horizon  # note: other cars do not change lanes
                x_off = i * s * math.cos(self.lane_orientation)
                y_off = i * s * math.sin(self.lane_orientation)
                occupancy = dyn_obs[3]
                new_occupancy = translate(occupancy, xoff=x_off, yoff=y_off)
                states_i.append(new_occupancy)
            strtree = STRtree(states_i)
            states_other_cars.append(strtree)

        return states_other_cars

    def calc_new_occupancy(self, current_occupancy: Polygon, delta_pos: np.ndarray, dpsi: float) -> Polygon:
        translated_occupancy = translate(current_occupancy, xoff=delta_pos[0], yoff=delta_pos[1])
        return rotate(translated_occupancy, angle=dpsi, origin=translated_occupancy.centroid, use_radians=True)

    def has_collision(self, shortest_path: list, states_other_cars: list, occupancy: Polygon) -> bool:
        for i in range(1, len(shortest_path) - 1):  # exclude start and goal node
            strtree = states_other_cars[i - 1]
            delta_pos = np.array(
                [shortest_path[i][1].x - shortest_path[i - 1][1].x, shortest_path[i][1].y - shortest_path[i - 1][1].y]
            )
            dpsi = shortest_path[i][1].psi - shortest_path[i - 1][1].psi
            occupancy = self.calc_new_occupancy(current_occupancy=occupancy, delta_pos=delta_pos, dpsi=dpsi)

            # check for collision with each obstacle
            candidate_indices = strtree.query(occupancy)
            for idx in candidate_indices:
                obstacle = strtree.geometries[idx]
                if occupancy.intersects(obstacle):
                    return True
        return False

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
        # TODO: fix this function
        # if current_state.vx >= 24.5:
        #     return 0.0
        # return self.vp.acc_limits[1]
        return 0.0

    def plot_collisions(self, rtree, lanelet_polygons, shortest_path, states_other_cars, occupancy):
        """
        Plot the shapely.Polygon objects stored in the R-Tree.
        :param rtree_idx: R-Tree index containing the polygons.
        :param title: Title of the plot.
        """
        plt.figure(figsize=(30, 25), dpi=250)
        ax = plt.gca()

        # init occupancy other cars
        for car in states_other_cars:
            x, y = car[3].exterior.xy
            ax.plot(x, y, linestyle="-", linewidth=1, color="red")

        # Iterate through the R-Tree and plot each polygon
        cmap = plt.cm.get_cmap("tab20")
        for k, time_instance in enumerate(rtree):
            if k + 1 == len(shortest_path) - 1:  # exclude start and goal node
                break
            for i, polygon in enumerate(time_instance.geometries):
                x, y = polygon.exterior.xy
                color = cmap(i % cmap.N)
                ax.plot(x, y, linestyle="-", linewidth=1, color=color)

        # init occupancy my car
        x, y = occupancy.exterior.xy
        ax.plot(x, y, linestyle="-", linewidth=1, color="red")

        for i in range(1, len(shortest_path) - 1):  # exclude start and goal node
            delta_pos = np.array(
                [shortest_path[i][1].x - shortest_path[i - 1][1].x, shortest_path[i][1].y - shortest_path[i - 1][1].y]
            )
            dpsi = shortest_path[i][1].psi - shortest_path[i - 1][1].psi
            occupancy = self.calc_new_occupancy(current_occupancy=occupancy, delta_pos=delta_pos, dpsi=dpsi)
            x, y = occupancy.exterior.xy
            ax.plot(x, y, linestyle="-", linewidth=1, color="blue")

        # Plot static obstacles (from on_episode_init)
        for lanelet in lanelet_polygons:
            x, y = lanelet.shapely_object.exterior.xy
            plt.plot(x, y, linestyle="-", linewidth=0.8, color="darkorchid")

        ax.set_aspect("equal", adjustable="box")

        output_dir = "/tmp"
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, "plot_collisions.png")
        plt.savefig(filename, bbox_inches="tight")  # Save the plot with tight bounding box
        plt.close()
        print(f"Graph saved to {filename}")


### ADDITIONAL HELPER FUNCTIONS ###
import matplotlib.pyplot as plt
import os
import time
from matplotlib.collections import LineCollection


def plot_end_states(end_states, lanelet_polygons):
    plt.figure(figsize=(30, 25), dpi=250)
    ax = plt.gca()

    # Plot end states
    for state_list in end_states:
        x_coords = [state.x for state in state_list]
        y_coords = [state.y for state in state_list]
        plt.scatter(x_coords, y_coords, s=1, color="blue")  # Plot points and lines

    # Plot static obstacles
    for lanelet in lanelet_polygons:
        x, y = lanelet.shapely_object.exterior.xy
        plt.plot(x, y, linestyle="-", linewidth=0.8, color="darkorchid")

    ax.set_aspect("equal", adjustable="box")

    output_dir = "/tmp"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "end_states.png")
    plt.savefig(filename, bbox_inches="tight")  # Save the plot with tight bounding box
    plt.close()
    print(f"Graph saved to {filename}")


def plot_graph(graph, lanelet_polygons, shortest_path):
    plt.figure(figsize=(30, 25), dpi=250)
    ax = plt.gca()

    # Collect all node coordinates
    node_coords = []
    edge_coords = []
    for parent in graph.nodes:
        if parent[0] == -1:  # Skip the virtual goal node
            continue
        parent_x, parent_y = parent[1].x, parent[1].y
        node_coords.append((parent_x, parent_y))

        for child in graph.successors(parent):
            if child[0] == -1:  # Skip edges to the virtual goal node
                continue
            child_x, child_y = child[1].x, child[1].y
            edge_coords.append([(parent_x, parent_y), (child_x, child_y)])

    # Plot all nodes at once
    node_coords = np.array(node_coords)
    plt.scatter(node_coords[:, 0], node_coords[:, 1], color="blue", s=1)

    # Plot all edges at once using LineCollection
    edge_collection = LineCollection(edge_coords, colors="blue", linewidths=0.3)
    ax.add_collection(edge_collection)

    # Plot the shortest path
    if shortest_path:
        path_coords = []
        for node in shortest_path:
            if node[0] == -1:  # Skip the virtual goal node
                continue
            node_x, node_y = node[1].x, node[1].y
            path_coords.append((node_x, node_y))

        path_coords = np.array(path_coords)
        plt.plot(path_coords[:, 0], path_coords[:, 1], color="red", linewidth=1.5, marker="o", markersize=2)

    # Plot static obstacles
    for lanelet in lanelet_polygons:
        x, y = lanelet.shapely_object.exterior.xy
        plt.plot(x, y, linestyle="-", linewidth=0.8, color="darkorchid")

    ax.set_aspect("equal", adjustable="box")

    # Add ticks every 10 steps on the x-axis
    ax.set_xticks(np.arange(-120, max(node_coords[:, 0]) + 10, 10))

    output_dir = "/tmp"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "graph.png")
    plt.savefig(filename, bbox_inches="tight")  # Save the plot with tight bounding box
    plt.close()
    print(f"Graph saved to {filename}")


### BACKUP ###

# function that takes the goal and creates a list of shapely lines that mark the goal lane
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

# def propagate_state(self, x_pos: float, y_pos: float, vx: float, depth: int) -> list:
#     propagated_states = []
#     time_horizon = self.n_steps * float(self.dt)
#     s = vx * time_horizon  # note: cars do not change lanes
#     for i in range(depth):
#         propagated_states.append(
#             (x_pos + i * s * math.cos(self.lane_orientation), y_pos + i * s * math.sin(self.lane_orientation))
#         )

#     return propagated_states

# def plot_other_cars(init_pos, future_pos, lanelet_polygons):
#     plt.figure(figsize=(30, 25), dpi=250)
#     ax = plt.gca()

#     # Plot static obstacles (from on_episode_init)
#     for lanelet in lanelet_polygons:
#         x, y = lanelet.shapely_object.exterior.xy
#         plt.plot(x, y, linestyle="-", linewidth=0.8, color="darkorchid")

#     plt.scatter(init_pos[0], init_pos[1], color="red", s=10)
#     for pos in future_pos:
#         plt.scatter(pos[0], pos[1], color="red", s=10)

#     ax.set_aspect("equal", adjustable="box")

#     output_dir = "/tmp"
#     os.makedirs(output_dir, exist_ok=True)
#     filename = os.path.join(output_dir, "other_cars.png")
#     plt.savefig(filename, bbox_inches="tight")  # Save the plot with tight bounding box
#     plt.close()
#     print(f"Graph saved to {filename}")
