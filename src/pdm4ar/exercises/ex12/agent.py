from decimal import Decimal
import enum
import numpy as np

from dg_commons import PlayerName
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.controllers.steer import SteerController

from pdm4ar.exercises.ex12.planning.graph import generate_graph
from pdm4ar.exercises.ex12.planning.dijkstra import Dijkstra
from pdm4ar.exercises.ex12.planning.collision_checker import CollisionChecker
from pdm4ar.exercises.ex12.planning.motion_primitives_manager import MotionPrimitiveManager
from pdm4ar.exercises.ex12.controllers.velocity_controller import VelocityController, VelocityControllerParams
from pdm4ar.exercises.ex12.controllers.steering_controller import CustomSteerController, SteeringControllerParams
from pdm4ar.exercises.ex12.planning.lane_change import LaneChangePlanner


class StateMachine(enum.Enum):
    INITIALIZATION = 1
    ATTEMPT_LANE_CHANGE = 2
    EXECUTE_LANE_CHANGE = 3
    HOLD_LANE = 4
    GOAL_STATE = 5


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

    steer_controller: SteerController
    velocity_controller: VelocityController
    custom_steer_controller: CustomSteerController
    lane_change_planner: LaneChangePlanner
    motion_primitive_manager: MotionPrimitiveManager
    collision_checker: CollisionChecker

    def on_episode_init(self, init_obs: InitSimObservations):
        """
        This method is called by the simulator only at the beginning of each simulation.
        Do not modify the signature of this method.
        It initializes the agent with the initial observations from the simulation.
        """

        self.name = init_obs.my_name
        self.goal = init_obs.goal
        self.vg = init_obs.model_geometry
        self.vp = init_obs.model_params
        self.dt = init_obs.dg_scenario.scenario.dt
        self.lanelet_network = init_obs.dg_scenario.lanelet_network
        self.half_lane_width = init_obs.goal.ref_lane.control_points[1].r
        self.goal_id = init_obs.dg_scenario.lanelet_network.find_lanelet_by_position(
            [init_obs.goal.ref_lane.control_points[1].q.p]
        )[0][0]

        self.lane_orientation = None
        self.goal_lane = None
        self.time_horizon = None
        self.n_steps = None
        self.delta_steer = None
        self.count_removed_branches = None
        self.path_node = None
        self.num_steps_path = None

        self.state = StateMachine.INITIALIZATION
        self.goal_lane_ids = self.retrieve_goal_lane_ids()
        self.steer_controller = SteerController.from_vehicle_params(vehicle_param=self.vp)
        self.dt_integral = 0.01  # time step for integral part of motion primitives
        self.scaling_dt = 0.1 / self.dt_integral  # scaling factor for get_commands frequency (0.1s)
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

        # initialize motion primitve manager
        self.motion_primitive_manager = MotionPrimitiveManager(
            vp=self.vp,
            vg=self.vg,
            dt_integral=self.dt_integral,
            scaling_dt=self.scaling_dt,
        )

        # initialize collision checker
        self.collision_checker = CollisionChecker()

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """
        This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.
        It returns the vehicle commands for the current simulation step.
        The logic of the agent is implemented in this method in form of a state machine.
        """

        current_state = sim_obs.players["Ego"].state

        if self.state == StateMachine.INITIALIZATION:
            self.handle_init_state(current_state, sim_obs)

        if self.state == StateMachine.ATTEMPT_LANE_CHANGE:
            self.handle_attempt_lane_change_state(current_state, sim_obs)

        if self.state == StateMachine.EXECUTE_LANE_CHANGE:
            return self.handle_execute_lane_change_state(current_state, sim_obs)

        if self.state == StateMachine.HOLD_LANE:
            return self.handle_hold_lane_state(current_state, sim_obs)

        if self.state == StateMachine.GOAL_STATE:
            return self.handle_goal_state(current_state, sim_obs)

    def handle_init_state(self, current_state, sim_obs):
        """
        This method is called during the first simulation step to determine whether the goal lane
        is to the left or right of the current vehicle position.
        In addition, it establishes the lane orientation.
        """

        self.lane_orientation = sim_obs.players[
            "Ego"
        ].state.psi  # assuming that lane orientation == initial orientation vehicle
        # check whether goal lane is left or right of current lane
        lanelet_id = self.lanelet_network.find_lanelet_by_position([np.array([current_state.x, current_state.y])])[0][0]
        lanelet = self.lanelet_network.find_lanelet_by_id(lanelet_id)
        if lanelet.adj_left in self.goal_lane_ids:
            self.goal_lane = "left"
        else:
            self.goal_lane = "right"
        self.state = StateMachine.ATTEMPT_LANE_CHANGE

    def handle_attempt_lane_change_state(self, current_state, sim_obs) -> VehicleCommands:
        """
        Handle the ATTEMPT_LANE_CHANGE state of the agent.
        In this state, the agent attempts to change lanes by calculating the shortest path
        to the goal lane using Dijkstra's algorithm.
        The agent generates motion primitives for the lane change and checks for collisions with dynamic obstacles.
        If the shortest path is found, the agent switches to the EXECUTE_LANE_CHANGE state.
        If no path is found, the agent switches to the HOLD_LANE state.
        """

        # need 4 motion primitives to change lanes - calculate the time horizon for one motion primitive
        self.time_horizon, ddelta = self.lane_change_planner.calc_time_horizon_and_ddelta(current_state=current_state)
        self.n_steps = int(self.time_horizon / self.dt_integral)
        self.delta_steer = ddelta * self.time_horizon

        # calculate control inputs and end states for motion primitives
        end_states, control_inputs = self.motion_primitive_manager.generate_control_inputs(
            current_state=current_state,
            goal_lane=self.goal_lane,
            lane_orientation=self.lane_orientation,
            delta_steer=self.delta_steer,
            steps_lane_change=self.steps_lane_change,
            n_steps=self.n_steps,
        )

        # build graph for djikstra's algorithm
        weighted_graph = generate_graph(
            current_state=current_state,
            end_states_traj=end_states,
            controls_traj=control_inputs,
            depth=self.depth,
            lanelet_network=self.lanelet_network,
            goal_id=self.goal_lane_ids,
            steps_lane_change=self.steps_lane_change,
        )

        # retrieve dynamic obstacles from simulation observations
        dyn_obs_current = self.process_dyn_obs(sim_obs)

        # create rtree for dynamic obstacles at each level of the graph (i.e. prepare efficient search for collision checking)
        states_dyn_obs = self.collision_checker.states_other_cars(
            dyn_obs_current,
            self.lane_orientation,
            self.time_horizon,
            self.depth,
            self.steps_lane_change,
        )

        # find shortest path using djikstra's algorithm
        self.shortest_path_finder(sim_obs, weighted_graph, states_dyn_obs)

        if self.shortest_path and self.count_removed_branches < self.depth:
            self.state = StateMachine.EXECUTE_LANE_CHANGE
            self.num_steps_path = len(self.shortest_path) - 1  # exclude virtual goal node
            self.path_node = 0

        else:
            self.state = StateMachine.HOLD_LANE

    def handle_execute_lane_change_state(self, current_state, sim_obs) -> VehicleCommands:
        """
        Handle the EXECUTE_LANE_CHANGE state of the agent.
        In this state, the agent follows the shortest path computed by Dijkstra's algorithm.
        The agent executes the lane change by following the control inputs of the shortest path.
        The agent remains in this state until the next graph node is reached.
        """

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

    def handle_hold_lane_state(self, current_state, sim_obs) -> VehicleCommands:
        """ "
        Handle the HOLD_LANE state of the agent.
        In this state, the agent is holding its lane.
        If the agent is on the goal lane, it switches to the GOAL_STATE.
        If the agent is not on the goal lane, it attempts to change lanes again after 0.5 seconds.
        The agent uses the velocity controller to maintain a safe distance to the vehicles in front
        and the custom steering controller to stay in the lane.
        """

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

    def handle_goal_state(self, current_state, sim_obs) -> VehicleCommands:
        """
        Handle the goal state of the agent.
        In this state, the agent has reached the goal lane and the controllers ensure that
        the agent stays in the lane and keeps a safe distance to the vehicles in front.
        """

        # In the goal state, we can simply return the commands to maintain the current speed and direction.
        acc = self.velocity_controller.abstandhalteassistent(current_state, sim_obs.players)
        ddelta = self.custom_steer_controller.spurhalteassistent(current_state, float(sim_obs.time))
        return VehicleCommands(acc, ddelta)

    def shortest_path_finder(self, sim_obs, weighted_graph, states_dyn_obs):
        """
        Find the shortest path using Dijkstra's algorithm.
        If a collision is detected, the algorithm will remove the edges and nodes of the shortest path
        and recompute the shortest path until a valid path is found or the maximum depth is reached.
        """

        current_occupancy = sim_obs.players["Ego"].occupancy
        self.count_removed_branches = 0  # note: assume one branch per lane change
        for k in range(weighted_graph.num_goal_nodes):
            djikstra_solver = Dijkstra(weighted_graph)
            self.shortest_path = djikstra_solver.path(
                start_node=weighted_graph.start_node, goal_node=weighted_graph.goal_node
            )

            # collision checking on shortest path with rtree for every time step
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
                weighted_graph.graph.remove_nodes_from(self.shortest_path[(k + 1) : -1])  # exclude start and goal node

                # update adjacency list and weights
                for u, v in edges_to_remove:
                    if u in weighted_graph.adj_list:
                        weighted_graph.adj_list[u].discard(v)

                self.count_removed_branches += 1
                self.shortest_path = []
                continue

            else:
                print("Found path without collision.")
                break

    def process_dyn_obs(self, sim_obs) -> list:
        """
        Process dynamic obstacles from the simulation observations.
        If the speed of a player is above a certain threshold, we calculate a bigger occupancy
        to account for the larger area they might cover due to their speed.
        Otherwise, we use the standard occupancy.
        This is to ensure that the collision checker is more conservative.
        """

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

        return dyn_obs_current

    def retrieve_goal_lane_ids(self) -> list:
        """
        Retrieve the lane IDs of the goal lane.
        The goal lane is a concatenation of multiple lanelets leading to the goal.
        """

        goal_lane_ids = [self.goal_id]
        lanelet = self.lanelet_network.find_lanelet_by_id(goal_lane_ids[-1])
        predecessor_id = lanelet.predecessor
        while predecessor_id:
            goal_lane_ids.append(predecessor_id[0])
            predecessor_id = self.lanelet_network.find_lanelet_by_id(predecessor_id[0]).predecessor

        return goal_lane_ids
