import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi,  voronoi_plot_2d

class BaseVoronoi:
    def __init__(self, df):
        self.df = df.dropna(subset=['Center x coordinate', 'Center y coordinate']).drop_duplicates()
        self.points = np.column_stack((self.df["Center x coordinate"], self.df["Center y coordinate"]))
        self.voronoi = Voronoi(self.points)
        self.vertices = self.voronoi.vertices
        self.regions = self.voronoi.regions
        self.point_to_region = self.voronoi.point_region
        print("BaseVoronoi initialized")
