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
from .lattice import WeightedGraph
from .A_star import Astar
import time
from matplotlib.collections import LineCollection


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
        self.lane_width = 2 * init_obs.goal.ref_lane.control_points[0].r
        self.lanelet_polygons = init_obs.dg_scenario.lanelet_network.lanelet_polygons

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
        bd = BicycleDynamics(self.vg, self.vp)
        mpg_params = MPGParam.from_vehicle_parameters(
            dt=Decimal(self.dt), n_steps=n_steps, n_vel=n_vel, n_steer=n_steer, vp=self.vp
        )
        mpg = MotionPrimitivesGenerator(param=mpg_params, vehicle_dynamics=bd.successor, vehicle_param=self.vp)
        end_states_traj, controls_traj = self.generate_primat(
            x0=current_state, mpg=mpg, steer_range=steer_range, n_steer=n_steer
        )

        # build graph
        depth = 8  # TODO: default value - need to decide how deep we want our graph to be
        start = time.time()
        graph = self.generate_graph(current_state, end_states_traj, depth)
        end = time.time()
        print(f"Generating the graph took {end - start} seconds.")

        # retrieve adjacency list for weighted graph
        start = time.time()
        adj_list = self.networkx_2_adjacencylist(graph)
        end = time.time()
        print(f"Conversion networkx graph to adjacency list took {end - start} seconds.")

        # convert DIgraph to our costum WeightedGraph
        start = time.time()
        weighted_graph = self.digraph_to_weighted_graph(graph)
        end = time.time()
        print(f"Conversion DI graph to weighted graph took {end - start} seconds.")

        start = time.time()
        plot_adj_list(adj_list, self.lanelet_polygons)
        end = time.time()
        print(f"Plotting the graph took {end - start} seconds.")

        # astar_solver = Astar.path(graph=weighted_graph)
        # TODO: need to define finite_horizon_goal
        # shortest_path = astar_solver.path(start=current_state, goal=finite_horizon_goal)

        return VehicleCommands(
            acc=controls_traj[0][0].acc, ddelta=controls_traj[0][0].ddelta, lights=LightsCmd("turn_left")
        )

    def digraph_to_weighted_graph(self, graph: DiGraph) -> WeightedGraph:
        """
        Converts a NetworkX DiGraph to the custom WeightedGraph structure.
        """
        # Extract adjacency list
        adj_list = {node: set(neighbors.keys()) for node, neighbors in graph.adjacency()}

        # Extract edge weights
        weights = {(u, v): data.get('weight', 1.0) for u, v, data in graph.edges(data=True)}

        # Create WeightedGraph
        return WeightedGraph(adj_list, weights, graph)

    def generate_primat(
        self,
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

    def generate_graph(self, current_state: VehicleState, end_states_traj: List[VehicleState], depth: int) -> DiGraph:
        graph = DiGraph()

        # calculate deltas
        deltas = [
            (state.x - current_state.x, state.y - current_state.y, state.psi - current_state.psi)
            for state in end_states_traj
        ]

        # add root node
        graph.add_node((0, current_state.x, current_state.y, current_state.psi))  # (level, x, y)

        # recursive function to generate children
        def add_children(level, x, y, psi):
            if level >= depth:
                return

            for dx, dy, dpsi in deltas:
                # check if child is within lane boundaries TODO: add clearance to road boundaries
                position = np.array([x + dx * np.cos(psi) - dy * np.sin(psi), y + dy * np.cos(psi) + dx * np.sin(psi)])
                # lanelet_id = self.lanelet_network.find_lanelet_by_position([position])
                # if not lanelet_id[0]:
                #     continue

                # only add child if psi is acceptable
                # if (
                #     psi + dpsi > np.pi / 4 or psi + dpsi < -np.pi / 4
                # ):  # TODO: heuristically chose 45deg, needs refinement
                #     continue

                child = (
                    level + 1,
                    x + dx * np.cos(psi) - dy * np.sin(psi),
                    y + dy * np.cos(psi) + dx * np.sin(psi),
                    psi + dpsi,
                )
                graph.add_node(child)
                graph.add_edge((level, x, y, psi), child)

                # recursively add children for child
                add_children(child[0], child[1], child[2], child[3])  # (level, x, y, psi)

        add_children(0, current_state.x, current_state.y, current_state.psi)

        return graph

    def networkx_2_adjacencylist(self, graph: DiGraph) -> dict:
        adj_list = dict()
        atlas = graph.adj._atlas
        for n in atlas.keys():
            adj_list[n] = set(atlas[n].keys())
        return adj_list


### BACKUP ###
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
