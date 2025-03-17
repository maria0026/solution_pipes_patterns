import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi,  voronoi_plot_2d
from voronoi import BaseVoronoi
import pandas as pd
from scipy.spatial import distance_matrix
import math

class VoronoiAnalyser(BaseVoronoi):
    def __init__(self, df):
        super().__init__(df)
        self.ridge_points = self.voronoi.ridge_points
        self.voronoi_points = df["Point of Voronoi"]
        self.areas=list(self.df['Area'])
        #self.regions_indices=list(self.df['Region index'])

    def calculate_sides(self):
        sides=[]
        for j, region_idx in enumerate(self.point_to_region):
            if self.voronoi_points[j]==1 and (-1 not in self.regions[int(region_idx)]): #only if region is good voronoi cell
                sides.append(len(self.regions[int(region_idx)]))
        return sides

    def calculate_distance_between_neighbours(self):
        distances = []
        checked_pairs = set()  
        for ridge in self.ridge_points:
            point1_index, point2_index = ridge  
            if (point1_index, point2_index) in checked_pairs or (point2_index, point1_index) in checked_pairs:
                continue 
            if (self.voronoi_points[point1_index]==1 and self.voronoi_points[point2_index]==1) and (-1 not in self.regions[int(point1_index)]) and (-1 not in self.regions[int(point2_index)]):
                point1 = self.points[point1_index]
                point2 = self.points[point2_index]
                distance = np.linalg.norm(point1 - point2)
                distances.append(distance)
                checked_pairs.add((point1_index, point2_index))

        distances = np.array(distances)
        return distances


    def calculate_orientational_order(self, absolute = False):
        psi = np.zeros(len(self.points), dtype=complex)
        for j, region_idx in enumerate(self.point_to_region): 
            if self.voronoi_points[j] == 1 and (-1 not in self.regions[int(region_idx)]): 
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
                if absolute:
                    self.df['Hexatic order']=np.abs(psi)
                else:
                    self.df['Hexatic order']=np.sqrt(np.real(psi)**2 + np.imag(psi)**2)

        return self.df
    

    def calculate_ripleys_k(self, r_vals, area):
        """
        Compute Ripley's K function for the Voronoi points.

        Parameters:
            r_vals (array-like): List of distances at which to compute K(r).

        Returns:
            DataFrame with 'r' values and 'K(r)' values.
        """
        distances=self.calculate_distance_between_neighbours()
        print(np.shape(distances))
        valid_points = self.df[self.df['Point of Voronoi'] == 1][["Center x coordinate", "Center y coordinate"]].values
        n = len(valid_points)  # Number of valid Voronoi points
        if n < 2:
            print("Not enough points for Ripley's K function.")
            return None
        
        lambda_hat = n / area  # Point intensity (density)
        dist_matrix = distance_matrix(valid_points, valid_points)  # Compute distances
        print(np.shape(dist_matrix))
        k_values = []
        for r in r_vals:
            count = np.sum((dist_matrix > 0) & (dist_matrix <= r))  # Exclude self-distances
            K_r = (area / (n * (n - 1))) * count  # Normalize
            k_values.append(K_r)

        # Store results
        ripley_df = pd.DataFrame({'r': r_vals, 'K(r)': k_values})
        self.df['Ripley K'] = np.interp(self.df['Area'], r_vals, k_values)  # Interpolate K values into df

        return ripley_df

    def find_points(self, x_min, y_min, x_max, y_max):
        N=0
        for point in self.points:
            if point[0]< x_max and point[0]> x_min and point[1]< y_max and point[1]> y_min:
                N+=1
        return N

    def calculate_mean_density(self):
        R = int(np.min(np.ptp(self.points, axis=0))/2)
        max_area = np.pi*R**2
        N=self.find_points(-R*2, -R*2,  R*2, R*2)
        n=N/max_area
        return n
        
    def radial_distribution(self, x0, y0, dr):

        #R= int(np.min(np.ptp(self.points, axis=0))/2)
        x_bounds = np.abs(self.points[0, :].min()), np.abs(self.points[0, :].max())
        y_bounds = np.abs(self.points[1, :].min()), np.abs(self.points[1, :].max())
        x_min = np.min(np.abs(x0) - np.array(x_bounds))
        y_min = np.min(np.abs(y0) - np.array(y_bounds))
        R = int(min(abs(x_min), abs(y_min)))
        g= np.zeros(math.ceil(R/dr), dtype=float)
        n=self.calculate_mean_density()

        for i, _ in enumerate(np.arange(0, R, dr)):
            r_i=(i+0.5)*dr
            x_min_i = x0 + r_i-dr/2
            x_max_i = x0 + r_i+dr/2
            y_min_i = y0 + r_i-dr/2
            y_max_i = y0 + r_i+dr/2
            area=2*np.pi*r_i*dr
            N_i=self.find_points(x_min_i, y_min_i, x_max_i, y_max_i)
            #print(r_i, area)
            g[i]=N_i/(area*n)
        return g
    
    def mean_radial_distribution(self, dr):
        mean=np.zeros(len(self.points), dtype=float)
        for i, point in enumerate(self.points):
            g=self.radial_distribution(point[0], point[1], dr)
            mean[i]=np.mean(g)
        return mean
