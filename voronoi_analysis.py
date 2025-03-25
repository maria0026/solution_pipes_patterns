import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi,  voronoi_plot_2d
from voronoi import BaseVoronoi
import pandas as pd
from scipy.spatial import distance_matrix
import math
from shapely.geometry import Point, Polygon

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

    def find_points(self, x_center, y_center, radius_min, radius_max):
        N=0
        for point in self.points:
            if point[0]!=x_center and point[1]!=y_center:
                distance = math.sqrt((point[0] - x_center) ** 2 + (point[1] - y_center) ** 2)
                if distance <= radius_max and distance>=radius_min:
                    N += 1
        return N

    def calculate_mean_density(self):
        R = int(np.min(np.ptp(self.points, axis=0))/(2))
        max_area = np.pi*R**2
        N=self.find_points(0, 0, 0, R)
        print(N)
        n=N/max_area
        return n
        
    def intersection_area(self, circle_center, circle_radius, polygon_points):
        circle = Point(circle_center).buffer(circle_radius)
        
        polygon = Polygon(polygon_points)
        intersection = circle.intersection(polygon)
        return intersection.area
        
    def convex_hull_creation(self, points):
        points_sorted = sorted(points)
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            
        lower = []
        for p in points_sorted:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        
        upper = []
        for p in reversed(points_sorted):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        convex_hull = lower[:-1] + upper[:-1]
        
        return convex_hull
    def radial_distribution(self, dr):

        R= int(np.min(np.ptp(self.points, axis=0))/(2))
        #x_bounds = np.abs(self.points[0, :].min()), np.abs(self.points[0, :].max())
        #y_bounds = np.abs(self.points[1, :].min()), np.abs(self.points[1, :].max())
        #x_min = np.min(np.abs(x0) - np.array(x_bounds))
        #y_min = np.min(np.abs(y0) - np.array(y_bounds))
        
        #R = int(min(abs(x_min), abs(y_min)))
        g= np.zeros(math.ceil(R/dr), dtype=float)
        #g= np.zeros(dr, dtype=float)
        n=self.calculate_mean_density()

        for i, _ in enumerate(np.arange(0, R, dr)):
            g_r =np.zeros((len(self.points)))
            for j, point in enumerate(self.points):
                r_i=(i+0.5)*dr
                radius_min = r_i-dr/2
                radius_max = r_i+dr/2
                area=2*np.pi*r_i*dr
                N_i=self.find_points(point[0], point[1], radius_min, radius_max)
                #print(r_i, area)
                # adding weights
                 intersection=self.intersection_area(point,radius_max, self.convex_hull_creation(self.points))-self.intersection_area(point,radius_min, self.convex_hull_creation(self.points))
                weight= area/intersection
                #
                #g_r[j]=N_i/(area*n)
                g_r[j]= N_i*weight/(area*n) 
            g[i]= np.mean(g_r)#N_i/(area*n)
        return g
