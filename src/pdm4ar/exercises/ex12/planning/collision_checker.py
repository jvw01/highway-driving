from shapely.geometry import Polygon
from shapely.strtree import STRtree
from shapely.affinity import translate, rotate
import numpy as np
import math

class CollisionChecker:

    def states_other_cars(
        self,
        dyn_obs_current: list,
        lane_orientation: float,
        time_horizon: float,
        depth: int,
        steps_lane_change: int,
    ) -> list:
        states_other_cars = []
        for i in range(
            1, depth + steps_lane_change
        ):  # note: start at 1 because car does not collide in init state
            states_i = []
            for dyn_obs in dyn_obs_current:
                vx = dyn_obs[2]
                s = vx * time_horizon  # note: other cars do not change lanes
                x_off = i * s * math.cos(lane_orientation)
                y_off = i * s * math.sin(lane_orientation)
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
