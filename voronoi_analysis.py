import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi,  voronoi_plot_2d
from voronoi import BaseVoronoi

class VoronoiAnalyser(BaseVoronoi):
    def __init__(self, df):
        super().__init__(df)
        self.ridge_points = self.voronoi.ridge_points
        self.voronoi_points = df["Point of Voronoi"]
        self.areas=list(self.df['Area'])
        self.regions_indices=list(self.df['Region index'])

    def calculate_sides(self):
        sides=[]
        for j, region_idx in enumerate(self.regions_indices):
            if self.voronoi_points[j]==1: #only if region is good voronoi cell
                sides.append(len(self.regions[int(region_idx)]))
        return sides

    def calculate_distance_between_neighbours(self):
        distances = []
        checked_pairs = set()  
        for ridge in self.ridge_points:
            point1_index, point2_index = ridge  
            if (point1_index, point2_index) in checked_pairs or (point2_index, point1_index) in checked_pairs:
                continue 
            if (self.voronoi_points[point1_index]==1 and self.voronoi_points[point2_index]==1):
                point1 = self.points[point1_index]
                point2 = self.points[point2_index]
                distance = np.linalg.norm(point1 - point2)
                distances.append(distance)
                checked_pairs.add((point1_index, point2_index))

        distances = np.array(distances)
        return distances


    def calculate_orientational_order(self):
        psi = np.zeros(len(self.points), dtype=complex)
        for j, region_idx in enumerate(self.regions_indices): 
            if self.voronoi_points[j] == 1: 
                region = self.regions[int(region_idx)]

                # Find neighboring points from Voronoi ridges
                neighbors = set()
                for (p1, p2) in self.ridge_points:
                    if p1 == j:
                        neighbors.add(p2)
                    elif p2 == j:
                        neighbors.add(p1)

                if not neighbors:
                    continue  # Skip if no neighbors

                N_j = len(neighbors)  # Number of neighbors
                sum_theta = 0  # Sum of angles for complex exponential
                
                for k in neighbors:
                    dx, dy = self.points[k] - self.points[j]
                    theta_jk = np.arctan2(dy, dx)  # arctan2 gives the correct angle in all four quadrants
                    sum_theta += np.exp(1j * 6 * theta_jk)  # Apply 6-fold symmetry
                
                psi[j] = sum_theta / N_j 
        self.df['Hexatic order']=np.abs(psi)

        return self.df





