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
from dg_commons.sim.models.vehicle_utils import VehicleParameters

# for plotting
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import os

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

    def __init__(self):
        # feel free to remove/modify  the following
        self.params = Pdm4arAgentParams()
        self.myplanner = ()

    def on_episode_init(self, init_obs: InitSimObservations):
        """This method is called by the simulator only at the beginning of each simulation.
        Do not modify the signature of this method."""
        self.name = init_obs.my_name # 'Ego'
        self.goal = init_obs.goal
        self.sg = init_obs.model_geometry
        self.sp = init_obs.model_params

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

        #########
        # sim_obs.players: a dictionary with key: name of player and value: PlayerObservations
        # PlayerObservations contains the state and occupancy of the player
        # state: {'x', 'y', 'psi', 'vx', 'delta'}
        # occupancy: polygons with 5 points (first and last points are the same) so essentially rectangles and parallelograms that represent the space the car uses
        # contains the own and other vehicles' states and occupancies and the vehicles are named 'Ego', 'P1', 'P2', ...

        print("iteration at: ", sim_obs.time)
        # for player_name, player_obs in sim_obs.players.items():
        #     player_state = player_obs.state
        #     player_obs = player_obs.occupancy
        #     print(player_name)
        #     print(player_state)
        #     print(player_obs)


        ######################## PLOT ########################
        plt.figure(figsize=(12, 10))

        for player_name, player_obs in sim_obs.players.items():
    
            polygons = [player_obs.occupancy for player_obs in sim_obs.players.values()]
            # Plot polygons
            for polygon_coords in polygons:
                # print(polygon_coords)
                # print(type(polygon_coords))
                x, y = polygon_coords.exterior.xy

                if player_name == self.name:
                    plt.fill(x, y, alpha=0.3, color='red')
                else:
                    plt.fill(x, y, alpha=0.3, color='blue')

            # Plot state points and connect them
            # print(player_obs.state)
            # print(type(player_obs.state))
            state_x = player_obs.state.x
            state_y = player_obs.state.y
            plt.plot(state_x, state_y, marker='o', color='black', label="State Path")

            # Plot static obstacles (from on_episode_init)
            for obs in self.static_obs:
                x, y = obs.shape.xy
                plt.plot(x, y, linestyle='-', linewidth=2, color='darkorchid')
            

        # Add labels and legend
        output_dir = "/tmp"
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"agent.png")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Polygon Occupancies, State Path, and Static Obstacles")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        print(f"Image saved to {filename} \n")
        ######################## PLOT ########################

        #########

        # todo implement here some better planning
        rnd_acc = random.random() * self.params.param1
        rnd_ddelta = (random.random() - 0.5) * self.params.param1

        return VehicleCommands(acc=rnd_acc, ddelta=rnd_ddelta)
