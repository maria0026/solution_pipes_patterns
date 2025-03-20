from voronoi_analysis import VoronoiAnalyser
from scipy.spatial import Voronoi,  voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from voronoi import BaseVoronoi
import matplotlib.cm as cm

class Voronoi_Plotter(VoronoiAnalyser):
    def __init__(self, df):
        super().__init__(df)
        #self.voronoi_points = df["Point of Voronoi"]

    def all_voronoi_diagram(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        voronoi_plot_2d(self.voronoi, ax=ax, show_vertices=False, line_width=2, line_colors='blue')
        ax.scatter(self.points[:, 0], self.points[:, 1], s=1, color='red', zorder=5, label="Points")

        for region in self.regions:
            if not -1 in region:
                polygon = [self.vertices[i] for i in region]
                plt.fill(*zip(*polygon))

        ax.set_title("Voronoi Diagram", fontsize=16)
        ax.set_xlabel("X Coordinate", fontsize=14)
        ax.set_ylabel("Y Coordinate", fontsize=14)
        ax.grid(True)
        #ax.legend()
        #Zooming in:

        plt.show()

    def all_voronoi_diagram_area_filtered(self, x_lim_min=20, x_lim_max=60, y_lim_min=30, y_lim_max=50):
        fig, ax = plt.subplots(figsize=(8, 6))
        voronoi_plot_2d(self.voronoi, ax=ax, show_vertices=False, line_width=2, line_colors='blue')

        for i, point in enumerate(self.points):
            color = 'blue' if self.voronoi_points.iloc[i] else 'red'  # Assuming True/False values in 'Point of Voronoi'
            ax.scatter(point[0], point[1], s=3, color=color, zorder=5)

        ax.set_title("Voronoi Diagram", fontsize=16)
        ax.set_xlabel("X Coordinate", fontsize=14)
        ax.set_ylabel("Y Coordinate", fontsize=14)
        ax.grid(True)
        ax.set_xlim(x_lim_min, x_lim_max)
        ax.set_ylim(y_lim_min, y_lim_max)

        for j, region_index in enumerate(self.point_to_region):
            if (self.voronoi_points.iloc[j]) and (-1 not in self.regions[int(region_index)]):
                polygon = [self.vertices[i] for i in self.regions[int(region_index)]]
                plt.fill(*zip(*polygon))
                

        plt.show()


    def areas_hist(self,areas):
        plt.hist(areas, bins=20, range=(0,13), edgecolor='black', label=f'Mean {np.mean(areas):.2f}, std {np.std(areas):.2f}')
        plt.xticks(range(13))
        plt.xlabel("Area")
        plt.ylabel("Number of regions")
        plt.legend()
        plt.xlim(0,13)
        plt.show()

    def sides_number_hist(self, sides):
        number_of_bins = np.max(sides)-np.min(sides)+1
        plt.hist(sides, bins=number_of_bins, edgecolor='black', label=f'Mean {np.mean(sides):.2f}, std {np.std(sides):.2f}')
        plt.xticks(range(12))
        plt.locator_params(axis='y', integer=True)
        plt.xlim(0, 12)
        plt.legend()
        plt.xlabel("Number of sides")
        plt.ylabel("Number of regions")
        plt.show()

    def distance_between_neighbours_hist(self, distances):
        number_of_bins = int(np.max(distances)-np.min(distances))
        plt.hist(distances, bins=number_of_bins, edgecolor='black', label=f'Mean {np.mean(distances):.2f}, std {np.std(distances):.2f}')
        plt.locator_params(axis='y', integer=True)
        plt.xlim(0,30)
        #plt.xlim(0, math.ceil(np.max(distances))) #250
        plt.legend()
        plt.xlabel("Distance")
        plt.ylabel("Number of neighbours in voronoi")
        plt.show()

    def order_hist(self, order):
        plt.hist(order, bins = 20,  edgecolor='black', label=f'Mean {np.mean(order):.2f}, std {np.std(order):.2f}')
        plt.title("Orientational order")
        plt.legend()
        plt.xlabel("Order parameter")
        plt.ylabel("Number of points")
        plt.show()


    def hexatic_order(self, hexatic_order, x_lim_min=20, x_lim_max=60, y_lim_min=30, y_lim_max=50):
        fig, ax = plt.subplots(figsize=(8, 6))
        voronoi_plot_2d(self.voronoi, ax=ax, show_points=False, show_vertices=False, line_width=0.5, line_colors='blue')

        for j, region_index in enumerate(self.point_to_region):
            if (self.voronoi_points.iloc[j]) and (-1 not in self.regions[int(region_index)]):
                polygon = [self.vertices[i] for i in self.regions[int(region_index)]]
                print(hexatic_order[j])
                color = cm.viridis(hexatic_order[j])  # Assign color based on hexatic order
                plt.fill(*zip(*polygon), color=color, alpha=0.7, edgecolor="black")  # Adjust transparency if needed

        sm = cm.ScalarMappable(cmap=cm.viridis)
        sm.set_array(hexatic_order[hexatic_order<19])
        plt.colorbar(sm, ax=ax, label="Hexatic Order") 
        #plt.colorbar(cm.ScalarMappable(cmap=cm.viridis), label="Hexatic Order")  # Add color legend

        ax.set_title("Voronoi Diagram", fontsize=16)
        ax.set_xlabel("X Coordinate", fontsize=14)
        ax.set_ylabel("Y Coordinate", fontsize=14)
        ax.grid(True)
        ax.set_xlim(x_lim_min, x_lim_max)
        ax.set_ylim(y_lim_min, y_lim_max)

        plt.show()

