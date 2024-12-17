import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi,  voronoi_plot_2d


def voronoi_diagrams(df):
    points = np.column_stack((df["Center x coordinate"], df["Center y coordinate"]))
    vor = Voronoi(points)


    fig, ax = plt.subplots(figsize=(8, 6))
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_width=2, line_colors='blue')
    ax.scatter(points[:, 0], points[:, 1], color='red', zorder=5, label="Points")
    ax.set_title("Voronoi Diagram", fontsize=16)
    ax.set_xlabel("X Coordinate", fontsize=14)
    ax.set_ylabel("Y Coordinate", fontsize=14)
    ax.grid(True)
    ax.legend()
    #Zooming in:
    ax.set_xlim(531925, 531975)
    ax.set_ylim(5752275, 5752325)
    plt.show()




