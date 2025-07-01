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
import frozendict
from dataclasses import dataclass
from decimal import Decimal
from shapely.geometry import Polygon
from matplotlib.collections import LineCollection
from dg_commons.controllers.steer import SteerController
from dg_commons.controllers.pure_pursuit import PurePursuit
import numpy as np
import math
import enum
from shapely.affinity import translate, rotate
from shapely.strtree import STRtree

from pdm4ar.exercises.ex12.graph import generate_graph
from pdm4ar.exercises.ex12.motion_primitves import calc_next_state, generate_primat
from pdm4ar.exercises.ex12.dijkstra import Dijkstra

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
    d_ref: float = 2
    d_ref_K = 0.5
    T: float = 8
    init_abstand: bool = False
    last_e: float

    # parameters for the steering controller:
    K_psi: float = 1
    K_dist: float = 0.1
    K_delta: float = 2
    K_d_psi: float = 0.5
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
        self.state = StateMachine.INITIALIZATION
        self.goal_lane_ids = self.retrieve_goal_lane_ids()
        self.steer_controller = SteerController.from_vehicle_params(vehicle_param=self.vp)
        self.freq_counter = 0
        self.shortest_path = []
        self.idx_previous_center_point = 0

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
            self.time_horizon, ddelta = self.calc_time_horizon_and_ddelta(current_state=current_state)
            self.n_steps = int(self.time_horizon / self.dt_integral)
            self.delta_steer = ddelta * self.time_horizon

            end_states = []
            control_inputs = []
            state = current_state

            # only use 3 motion primitives to have abstandhalteassistent active again as soon as possible
            self.steps_lane_change = 3
            for i in range(self.steps_lane_change):
                # case 3 motion primitives
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

                if i == self.steps_lane_change - 1:  # to avoid unncecessary computations
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
            self.depth = 1  # note: has to be greater than steps_lane_change!
            weighted_graph = generate_graph(
                current_state=current_state,
                end_states_traj=end_states,
                controls_traj=control_inputs,
                depth=self.depth,
                lanelet_network=self.lanelet_network,
                goal_id=self.goal_lane_ids,
                steps_lane_change=self.steps_lane_change,
            )

            current_occupancy = sim_obs.players["Ego"].occupancy  # type: ignore
            dyn_obs_current = []
            magic_speed = 6
            for player in sim_obs.players:
                if player != "Ego":
                    if sim_obs.players[player].state.vx > magic_speed:
                        new_occupancy = self.calc_big_occupacy(
                            sim_obs.players[player].occupancy,
                            sim_obs.players[player].state.x,
                            sim_obs.players[player].state.y,
                        )  # type: ignore

                        dyn_obs_current.append(
                            (
                                sim_obs.players[player].state.x,  # type: ignore
                                sim_obs.players[player].state.y,  # type: ignore
                                sim_obs.players[player].state.vx,  # type: ignore
                                new_occupancy,  # sim_obs.players[player].occupancy
                            )
                        )

                    else:
                        dyn_obs_current.append(
                            (
                                sim_obs.players[player].state.x,  # type: ignore
                                sim_obs.players[player].state.y,  # type: ignore
                                sim_obs.players[player].state.vx,  # type: ignore
                                sim_obs.players[player].occupancy,  # type: ignore
                            )
                        )

            # create rtree for dynamic obstacles at each level of the graph (i.e. prepare efficient search for collision checking)
            states_dyn_obs = self.states_other_cars(dyn_obs_current)
            self.count_removed_branches = 0  # note: assume one branch per lane change
            for k in range(weighted_graph.num_goal_nodes):
                djikstra_solver = Dijkstra(weighted_graph)
                self.shortest_path = djikstra_solver.path(
                    start_node=weighted_graph.start_node, goal_node=weighted_graph.goal_node
                )
                # collision checking on shortest path with RTree for every time step
                if self.has_collision(
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
                    acc = self.abstandhalteassistent(current_state, sim_obs.players)  # type: ignore
                    ddelta = self.spurhalteassistent(current_state, float(sim_obs.time))  # type: ignore
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

            acc = self.abstandhalteassistent(current_state, sim_obs.players)  # type: ignore
            ddelta = self.spurhalteassistent(current_state, float(sim_obs.time))  # type: ignore
            return VehicleCommands(acc, ddelta)

        if self.state == StateMachine.GOAL_STATE:
            acc = self.abstandhalteassistent(current_state, sim_obs.players)  # type: ignore
            ddelta = self.spurhalteassistent(current_state, float(sim_obs.time))  # type: ignore
            return VehicleCommands(acc, ddelta)

    def retrieve_goal_lane_ids(self) -> list:
        goal_lane_ids = [self.goal_id]
        lanelet = self.lanelet_network.find_lanelet_by_id(goal_lane_ids[-1])
        predecessor_id = lanelet.predecessor
        while predecessor_id:
            goal_lane_ids.append(predecessor_id[0])
            predecessor_id = self.lanelet_network.find_lanelet_by_id(predecessor_id[0]).predecessor

        return goal_lane_ids

    def calc_time_horizon_and_ddelta(self, current_state: VehicleState) -> tuple[float, float]:
        # heurisitic approach: in first motion primitive with delta_max, the car drives approximately a sixth of the lane width in y' direction
        delta_y = self.half_lane_width * (1 / 3)
        n_steps = 0
        init_state_y = current_state.y
        next_state = VehicleState(
            x=current_state.x,
            y=current_state.y,
            psi=0,  # to make calculation easier - it does not matter for this calculation whether the road is at an angle
            vx=current_state.vx,
            delta=current_state.delta,
        )

        while (np.abs(next_state.y - init_state_y)) < delta_y:
            n_steps += 1
            next_state = calc_next_state(
                next_state, VehicleCommands(acc=0, ddelta=self.vp.ddelta_max), self.dt_integral, self.vp, self.vg
            )

        goal_state_y = next_state.y
        time_horizon = np.ceil(n_steps * self.dt_integral * 10) / 10  # round to nearest multiple of 0.1

        # calculate ddelta with new time horizon
        ddelta = self.vp.ddelta_max / 2  # start with ddelta_max / 2 as init guess for correct ddelta
        ddelta_interval = np.array([0, self.vp.ddelta_max])
        n_steps = int(time_horizon / self.dt_integral)
        init_state_y = current_state.y
        next_state = VehicleState(
            x=current_state.x,
            y=current_state.y,
            psi=0,  # to make calculation easier - it does not matter for this calculation whether the road is at an angle
            vx=current_state.vx,
            delta=current_state.delta,
        )

        calc_goal_state_y = current_state.y
        while np.abs(goal_state_y - calc_goal_state_y) > 1e-2:
            for _ in range(n_steps):
                next_state = calc_next_state(
                    next_state, VehicleCommands(acc=0, ddelta=ddelta), self.dt_integral, self.vp, self.vg
                )

            if goal_state_y > 0:
                if goal_state_y - next_state.y > 0:
                    ddelta_interval = np.array([ddelta, ddelta_interval[1]])
                    ddelta = np.mean(ddelta_interval)
                else:
                    ddelta_interval = np.array([ddelta_interval[0], ddelta])
                    ddelta = np.mean(ddelta_interval)
            else:
                if np.abs(goal_state_y) - np.abs(next_state.y) > 0:
                    ddelta_interval = np.array([ddelta_interval[0], ddelta])
                    ddelta = np.mean(ddelta_interval)
                else:
                    ddelta_interval = np.array([ddelta, ddelta_interval[1]])
                    ddelta = np.mean(ddelta_interval)

            calc_goal_state_y = next_state.y
            next_state = VehicleState(
                x=current_state.x,
                y=current_state.y,
                psi=0,  # to make calculation easier - it does not matter for this calculation whether the road is at an angle
                vx=current_state.vx,
                delta=current_state.delta,
            )
        return (time_horizon, ddelta)

    def states_other_cars(self, dyn_obs_current: list) -> list:
        states_other_cars = []
        for i in range(
            1, self.depth + self.steps_lane_change
        ):  # note: start at 1 because car does not collide in init state
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
        for i in range(1, len(shortest_path) - 1):  # exclude start and goal node, only check
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
        )
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
            return -1 * dpsi - 0.1 * dist - 2 * current_state.delta

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

        return ddelta

    def retrieve_current_lane_ids(self, lane_id) -> list:
        current_lane_ids = [lane_id]
        lanelet = self.lanelet_network.find_lanelet_by_id(current_lane_ids[-1])
        successor_id = lanelet.successor

        while successor_id:
            current_lane_ids.append(successor_id[0])
            successor_id = self.lanelet_network.find_lanelet_by_id(successor_id[0]).successor

        return current_lane_ids

    def abstandhalteassistent(self, current_state: VehicleState, players: frozendict) -> float:
        player_ahead = None
        goal_x = self.goal.goal_polygon.centroid.x
        my_lanelet_id = self.lanelet_network.find_lanelet_by_position([np.array([current_state.x, current_state.y])])

        if my_lanelet_id[0]:
            current_lane_ids = self.retrieve_current_lane_ids(my_lanelet_id[0][0])
            for player in players:
                if player != self.name:
                    player_lanelet_id = self.lanelet_network.find_lanelet_by_position(
                        [np.array([players[player].state.x, players[player].state.y])]
                    )
                    if not player_lanelet_id[0]:
                        continue
                    if player_lanelet_id[0][0] in current_lane_ids:
                        if np.abs(goal_x - players[player].state.x) < np.abs(
                            goal_x - current_state.x
                        ):  # note: has singularity at lane heading of 90 deg
                            player_ahead = players[player]

        if player_ahead:
            dist_to_player = math.sqrt(
                (player_ahead.state.x - current_state.x) ** 2 + (player_ahead.state.y - current_state.y) ** 2
            )
            d_ref = self.d_ref_K * player_ahead.state.vx
            poly_edges = list(player_ahead.occupancy.exterior.coords)
            l_half_ahead = math.sqrt(
                (poly_edges[0][0] - player_ahead.state.x) ** 2 + (poly_edges[0][1] - player_ahead.state.y) ** 2
            )
            if d_ref < l_half_ahead + 1.5:
                d_ref = l_half_ahead + 1.5

            e = dist_to_player - d_ref
            if not self.init_abstand:
                self.init_abstand = True
                self.last_e = e
                return (player_ahead.state.vx / self.T + self.d_ref) * (
                    self.last_e
                )  # P-controler for first time step, here last_e is also the current error

            de = (e - self.last_e) / float(self.dt)
            self.last_e = e
            K_p = player_ahead.state.vx / self.T + self.d_ref

            return K_p * (self.T * de + e)

        else:
            # increase/decrease speed if car is too slow/fast (avoid velocity penalty) - else keep speed
            if current_state.vx > 25:  #
                return self.vp.acc_limits[0]
            elif current_state.vx < 5:
                return self.vp.acc_limits[1]
            else:
                return 0.0

    def calc_big_occupacy(self, occupancy: Polygon, x: float, y: float) -> Polygon:
        # enlarge occupancy polygon to avoid collisions
        coordinates = list(occupancy.exterior.coords)[:-1]
        new_coords = []
        c_coords = [x, y]

        for coords in coordinates:
            r_ce = [coords[0] - x, coords[1] - y]
            m = 1.5
            new_e = [c_coords[0] + m * r_ce[0], c_coords[1] + m * r_ce[1]]
            new_coords.append(new_e)

        new_occupancy = Polygon(new_coords)

        return new_occupancy

    def plot_collisions(self, rtree, lanelet_polygons, shortest_path, states_other_cars, occupancy, graph):
        """
        Plot the shapely.Polygon objects stored in the R-Tree.
        :param rtree_idx: R-Tree index containing the polygons.
        :param title: Title of the plot.
        """
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
