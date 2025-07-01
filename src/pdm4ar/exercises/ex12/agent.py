from dataclasses import dataclass
from dg_commons import PlayerName
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.models.vehicle import VehicleState
from decimal import Decimal
from dataclasses import dataclass
from decimal import Decimal
from dg_commons.controllers.steer import SteerController
from dg_commons.controllers.pure_pursuit import PurePursuit
import numpy as np
import enum

from pdm4ar.exercises.ex12.planning.motion_primitves import generate_primat
from pdm4ar.exercises.ex12.planning.graph import generate_graph
from pdm4ar.exercises.ex12.planning.dijkstra import Dijkstra
from pdm4ar.exercises.ex12.planning.collision_checker import CollisionChecker
from pdm4ar.exercises.ex12.controllers.velocity_controller import VelocityController, VelocityControllerParams
from pdm4ar.exercises.ex12.controllers.steering_controller import CustomSteerController, SteeringControllerParams
from pdm4ar.exercises.ex12.state_machine.lane_change import LaneChangePlanner

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
    """
    This is the PDM4AR agent.
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task
    """

    name: PlayerName
    goal: PlanningGoal
    vg: VehicleGeometry
    vp: VehicleParameters
    dt: Decimal
    lane_width: float

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
        """
        This method is called by the simulator only at the beginning of each simulation.
        Do not modify the signature of this fmethod.
        """
        self.name = init_obs.my_name
        self.goal = init_obs.goal
        self.vg = init_obs.model_geometry
        self.vp = init_obs.model_params
        self.dt = init_obs.dg_scenario.scenario.dt
        self.dt_integral = 0.01  # time step for integral part of motion primitives
        self.scaling_dt = 0.1 / self.dt_integral  # scaling factor for get_commands frequency (0.1s)
        self.lanelet_network = init_obs.dg_scenario.lanelet_network
        self.half_lane_width = init_obs.goal.ref_lane.control_points[1].r
        self.goal_id = init_obs.dg_scenario.lanelet_network.find_lanelet_by_position(
            [init_obs.goal.ref_lane.control_points[1].q.p]
        )[0][0]
        self.state = StateMachine.INITIALIZATION
        self.goal_lane_ids = self.retrieve_goal_lane_ids()
        self.steer_controller = SteerController.from_vehicle_params(vehicle_param=self.vp)
        self.freq_counter = 0
        self.shortest_path = []
        self.idx_previous_center_point = 0
        self.depth = 1  # note: has to be greater than steps_lane_change!
        self.steps_lane_change = 3  # number of motion primitives to change lanes

        # initialize velocity controller
        velocity_params = VelocityControllerParams()
        self.velocity_controller = VelocityController(
            params=velocity_params,
            vp=self.vp,
            goal=self.goal,
            lanelet_network=self.lanelet_network,
            name=self.name,
            dt=self.dt,
        )

        # initialize steering controller
        steering_controller_params = SteeringControllerParams()
        self.custom_steer_controller = CustomSteerController(
            params=steering_controller_params,
            dt=self.dt,
            vp=self.vp,
            lanelet_network=self.lanelet_network,
        )

        # initialize lane change planner
        self.lane_change_planner = LaneChangePlanner(
            vp=self.vp,
            vg=self.vg,
            dt_integral=self.dt_integral,
            scaling_dt=self.scaling_dt,
            half_lane_width=self.half_lane_width,
        )

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """
        current_state = sim_obs.players["Ego"].state

        if self.state == StateMachine.INITIALIZATION:
            self.lane_orientation = sim_obs.players[
                "Ego"
            ].state.psi  # assuming that lane orientation == initial orientation vehicle
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
            self.time_horizon, ddelta = self.lane_change_planner.calc_time_horizon_and_ddelta(current_state=current_state)
            self.n_steps = int(self.time_horizon / self.dt_integral)
            self.delta_steer = ddelta * self.time_horizon

            # initialize collision checker
            self.collision_checker = CollisionChecker(
                lane_orientation=self.lane_orientation,
                time_horizon=self.time_horizon,
                depth=self.depth,
                steps_lane_change=self.steps_lane_change,
            )

            end_states = []
            control_inputs = []
            state = current_state

            for i in range(self.steps_lane_change):
                if i == 0:
                    case = 0
                elif i == 1:
                    case = 2

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

                if i == self.steps_lane_change - 1:
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

            # build graph
            weighted_graph = generate_graph(
                current_state=current_state,
                end_states_traj=end_states,
                controls_traj=control_inputs,
                depth=self.depth,
                lanelet_network=self.lanelet_network,
                goal_id=self.goal_lane_ids,
                steps_lane_change=self.steps_lane_change,
            )

            current_occupancy = sim_obs.players["Ego"].occupancy
            dyn_obs_current = []
            magic_speed = 6
            for player in sim_obs.players:
                if player != "Ego":
                    if sim_obs.players[player].state.vx > magic_speed:
                        new_occupancy = self.collision_checker.calc_big_occupacy(
                            sim_obs.players[player].occupancy,
                            sim_obs.players[player].state.x,
                            sim_obs.players[player].state.y,
                        )

                        dyn_obs_current.append(
                            (
                                sim_obs.players[player].state.x,
                                sim_obs.players[player].state.y,
                                sim_obs.players[player].state.vx,
                                new_occupancy,
                            )
                        )

                    else:
                        dyn_obs_current.append(
                            (
                                sim_obs.players[player].state.x,
                                sim_obs.players[player].state.y,
                                sim_obs.players[player].state.vx,
                                sim_obs.players[player].occupancy,
                            )
                        )

            # create rtree for dynamic obstacles at each level of the graph (i.e. prepare efficient search for collision checking)
            states_dyn_obs = self.collision_checker.states_other_cars(dyn_obs_current)
            self.count_removed_branches = 0  # note: assume one branch per lane change
            for k in range(weighted_graph.num_goal_nodes):
                djikstra_solver = Dijkstra(weighted_graph)
                self.shortest_path = djikstra_solver.path(
                    start_node=weighted_graph.start_node, goal_node=weighted_graph.goal_node
                )
                # collision checking on shortest path with RTree for every time step
                if self.collision_checker.has_collision(
                    shortest_path=self.shortest_path, states_other_cars=states_dyn_obs, occupancy=current_occupancy
                ):
                    print("Collision detected. Recomputing shortest path.")
                    # eliminate nodes and edges of the shortest path from the graph
                    edges_to_remove = [
                        (self.shortest_path[k + i], self.shortest_path[k + i + 1])
                        for i in range(0, len(self.shortest_path) - k - 1)
                    ]
                    weighted_graph.graph.remove_edges_from(edges_to_remove)
                    weighted_graph.graph.remove_nodes_from(
                        self.shortest_path[(k + 1) : -1]
                    )  # exclude start and goal node
                    # Update adjacency list and weights
                    for u, v in edges_to_remove:
                        if u in weighted_graph.adj_list:
                            weighted_graph.adj_list[u].discard(v)
                    # plot_graph(weighted_graph.graph, self.lanelet_network.lanelet_polygons, [])
                    self.count_removed_branches += 1
                    self.shortest_path = []
                    continue
                else:
                    # a path without collision was found
                    print("Found path without collision.")
                    break

            if self.shortest_path and self.count_removed_branches < self.depth:
                self.state = StateMachine.EXECUTE_LANE_CHANGE
                self.num_steps_path = len(self.shortest_path) - 1  # exclude virtual goal node
                self.path_node = 0

            else:
                self.state = StateMachine.HOLD_LANE

        if self.state == StateMachine.EXECUTE_LANE_CHANGE:  # case: A* found a path -> lane change
            idx = int(self.freq_counter % (self.n_steps / self.scaling_dt))

            # remain in EXECUTE_LANE_CHANGE state as we are inbetween graph nodes
            if idx != 0:
                self.freq_counter += 1
                return VehicleCommands(
                    acc=self.shortest_path[self.path_node][3][idx].acc,
                    ddelta=self.shortest_path[self.path_node][3][idx].ddelta,
                )

            # new graph node reached, switch to HOLD_LANE if we're at the goal node, else we stay until the goal node is reached
            else:
                self.path_node += 1
                if self.path_node == self.num_steps_path:  # goal node reached
                    self.state = StateMachine.HOLD_LANE
                    acc = self.velocity_controller.abstandhalteassistent(current_state, sim_obs.players)
                    ddelta = self.custom_steer_controller.spurhalteassistent(current_state, float(sim_obs.time))
                    self.freq_counter = 0  # done with lane change -> reset freq_counter
                    return VehicleCommands(acc, ddelta)
                else:
                    self.freq_counter += 1
                    return VehicleCommands(
                        acc=self.shortest_path[self.path_node][3][idx].acc,
                        ddelta=self.shortest_path[self.path_node][3][idx].ddelta,
                    )

        if self.state == StateMachine.HOLD_LANE:  # default case: continue on lane
            # check if car is on goal lane
            player_lanelet_id = self.lanelet_network.find_lanelet_by_position(
                [np.array([current_state.x, current_state.y])]
            )

            if player_lanelet_id[0][0] in self.goal_lane_ids:  # stay on goal lane
                self.state = StateMachine.GOAL_STATE
            else:  # stay on current lane for 0.5s and attempt lane change again
                self.state = StateMachine.ATTEMPT_LANE_CHANGE

            acc = self.velocity_controller.abstandhalteassistent(current_state, sim_obs.players)
            ddelta = self.custom_steer_controller.spurhalteassistent(current_state, float(sim_obs.time))
            return VehicleCommands(acc, ddelta)

        if self.state == StateMachine.GOAL_STATE:
            acc = self.velocity_controller.abstandhalteassistent(current_state, sim_obs.players)
            ddelta = self.custom_steer_controller.spurhalteassistent(current_state, float(sim_obs.time))
            return VehicleCommands(acc, ddelta)

    def retrieve_goal_lane_ids(self) -> list:
        goal_lane_ids = [self.goal_id]
        lanelet = self.lanelet_network.find_lanelet_by_id(goal_lane_ids[-1])
        predecessor_id = lanelet.predecessor
        while predecessor_id:
            goal_lane_ids.append(predecessor_id[0])
            predecessor_id = self.lanelet_network.find_lanelet_by_id(predecessor_id[0]).predecessor

        return goal_lane_ids
