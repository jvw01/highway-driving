import matplotlib.pyplot as plt
from pdm4ar.exercises.ex12.planning.collision_checker import CollisionChecker
from matplotlib.collections import LineCollection
import numpy as np
import os

class VisualizationUtils:

    def __init__(self, collision_checker: CollisionChecker):
        """
        Initialize the VisualizationUtils with a collision checker.
        """

        self.collision_checker = collision_checker

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
            occupancy = self.collision_checker.calc_new_occupancy(
                current_occupancy=occupancy, delta_pos=delta_pos, dpsi=dpsi
            )
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
