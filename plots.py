from voronoi_analysis import VoronoiAnalyser
from scipy.spatial import Voronoi,  voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from voronoi import BaseVoronoi

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
        #ax.set_xlim(531925, 531975)
        #ax.set_ylim(5752275, 5752325)
        plt.show()

    def all_voronoi_diagram_area_filtered(self, area_limit):
        fig, ax = plt.subplots(figsize=(8, 6))
        voronoi_plot_2d(self.voronoi, ax=ax, show_vertices=False, line_width=2, line_colors='blue')
        for i, point in enumerate(self.points):
            color = 'blue' if self.voronoi_points.iloc[i] else 'red'  # Assuming True/False values in 'Point of Voronoi'
            ax.scatter(point[0], point[1], s=20, color=color, zorder=5)

        ax.set_title("Voronoi Diagram", fontsize=16)
        ax.set_xlabel("X Coordinate", fontsize=14)
        ax.set_ylabel("Y Coordinate", fontsize=14)
        ax.grid(True)
        #ax.legend()
        #ax.set_xlim(531925, 531975)
        #ax.set_ylim(5752275, 5752325)

        for j, area in enumerate(self.areas):
            if not area>area_limit:
                polygon = [self.vertices[i] for i in self.regions[int(self.regions_indices[j])]]
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
