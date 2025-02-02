from voronoi_analysis import VoronoiAnalyser
from scipy.spatial import Voronoi,  voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt

class Voronoi_Plotter(VoronoiAnalyser):
    def __init__(self, df):
        super().__init__(df)
        print("Voronoi plotter initialized")

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
        ax.legend()
        #Zooming in:
        #ax.set_xlim(531925, 531975)
        #ax.set_ylim(5752275, 5752325)
        plt.show()

    def all_voronoi_diagram_area_filtered(self):
        areas=self.calculate_areas()
        fig, ax = plt.subplots(figsize=(8, 6))
        voronoi_plot_2d(self.voronoi, ax=ax, show_vertices=False, line_width=2, line_colors='blue')
        ax.scatter(self.points[:, 0], self.points[:, 1], s=1, color='red', zorder=5, label="Points")
        ax.set_title("Voronoi Diagram", fontsize=16)
        ax.set_xlabel("X Coordinate", fontsize=14)
        ax.set_ylabel("Y Coordinate", fontsize=14)
        ax.grid(True)
        ax.legend()
        ax.set_xlim(531925, 531975)
        ax.set_ylim(5752275, 5752325)

        for j, area in enumerate(areas):
            if not area>14:
                polygon = [self.vertices[i] for i in self.regions[j]]
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
        plt.ylabel("Number of neughbours in voronoi")
        plt.show()
