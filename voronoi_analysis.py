import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi,  voronoi_plot_2d
from voronoi import BaseVoronoi
import pandas as pd
from scipy.spatial import distance_matrix
import math
from typing import Tuple
from shapely.geometry import Point, Polygon
from scipy.spatial.distance import cdist

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
                print(psi[j])
                if absolute:
                    self.df['Hexatic order']=np.abs(psi)
                else:
                    self.df['Hexatic order']=np.sqrt(np.real(psi)**2 + np.imag(psi)**2)

        return self.df
    

    def find_neighbors(self, x_center, y_center, radius):
        neighbors = set()
        for point in self.points:
            if (point[0]== x_center) and (point[1] ==y_center):
                continue
            new_radius = ((point[0] - x_center)**2 + (point[1] - y_center)**2)**0.5
            if new_radius <= radius:
                neighbors.add((point[0], point[1]))
        return neighbors
    

    def calculate_orientational_order_circles(self, radius, absolute = False):
        psi = np.zeros(len(self.points), dtype=complex)
        for i, point in enumerate(self.points):
            x_center, y_center = point
            
            #neighbors = set()
            neighbors = self.find_neighbors(x_center, y_center, radius)

            N_i = len(neighbors)  # Number of neighbors
            sum_theta = 0  # Sum of angles for complex exponential
            
            if N_i == 0 or N_i ==1:
                psi[i] = np.nan
                continue
            for k in neighbors:
                dx, dy = k - self.points[i]
                theta_jk = np.arctan2(dy, dx)  # arctan2 gives the correct angle in all four quadrants
                sum_theta += np.exp(1j * 6 * theta_jk)  # Apply 6-fold symmetry
            
            psi[i] = sum_theta / N_i 
            
            if absolute:
                self.df['Hexatic order']=np.abs(psi)
            else:
                self.df['Hexatic order']=np.sqrt(np.real(psi)**2 + np.imag(psi)**2)

        return self.df


    def distances_matrix(self):
        return cdist(self.points, self.points)


    def intersection_area_with_unit_square(self, x, y, r):
        """
        Calculates the intersection area between a circle centered at (x, y) with radius r
        and a square from (-0.5, -0.5) to (0.5, 0.5).
        """
        # Define the unit square
        square = Polygon([(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)])

        # Create the circle using a buffered point
        circle = Point(x, y).buffer(r, resolution=256)  # resolution controls smoothness

        # Compute the intersection
        intersection = circle.intersection(square)

        return intersection.area

    def ripley_2(self, r_values):

        hull=self.convex_hull_creation()
        max_area = self.hull_area(hull)
        distances = self.distances_matrix()
        #print(distances)
        N = len(self.points) #distances.shape[0]  # chcemy wszystkie, nie tylko voronoja
        
        K_values = np.zeros_like(r_values, dtype=float)
        L_values = np.zeros_like(r_values, dtype=float)

        for i, r in enumerate(r_values):
            k_r =np.zeros((len(self.points)))
            for j, point in enumerate(self.points):
                area=np.pi*r**2
                intersection=self.intersection_area(point, r, hull)
                #intersection = self.intersection_area_with_unit_square(point[0], point[1], r)
                weight= area/intersection
                neighbors_within_r = np.sum(distances[j] <= r) 
                k_r[j] = neighbors_within_r * weight

            K_values[i] = (max_area / (N * (N - 1))) * np.sum(k_r)
            #K_values[i]= (max_area) / (N*(N- 1))  * np.mean(k_r)
            L_values[i] = math.sqrt(K_values[i] / np.pi) - r

        return K_values, L_values
    

    def find_points(self, x_center, y_center, radius_min, radius_max):
        N=0
        for point in self.points:
            if point[0]!=x_center and point[1]!=y_center:
                distance = math.sqrt((point[0] - x_center) ** 2 + (point[1] - y_center) ** 2)
                if distance <= radius_max and distance>=radius_min:
                    N += 1
        return N

    def calculate_mean_density(self):
        #R = int(np.min(np.ptp(self.points, axis=0))/(2))
        '''
        x_bounds = (np.abs(self.points[0, :].max())-np.abs(self.points[0, :].min()))/2
        y_bounds = (np.abs(self.points[1, :].max())-np.abs(self.points[1, :].min()))/2
        R=np.sqrt(x_bounds**2+y_bounds**2)
        max_area = np.pi*R**2
        N=self.find_points(0, 0, 0, R)
        print(N)
        '''
        N=len(self.points)
        hull=self.convex_hull_creation()
        max_area=self.hull_area(hull)
        n=N/max_area
        return n
        
    def intersection_area(self, circle_center, circle_radius, polygon_points):
        circle = Point(circle_center).buffer(circle_radius)
        polygon = Polygon(polygon_points)
        intersection = circle.intersection(polygon)
        return intersection.area
    
    def convex_hull_creation(self):
        points_sorted = sorted(self.points, key=lambda x: x[0])
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
    
    def hull_area(self, polygon_points):
        polygon=Polygon(polygon_points)
        return polygon.area
    
    def radial_distribution(self, dr):
        hull_max=self.convex_hull_creation()
        weights=np.zeros(len(self.points), dtype=int)
        x=np.zeros(len(self.points), dtype=int)
        y=np.zeros(len(self.points), dtype=int)
        weight=0
        #R= int(np.min(np.ptp(self.points, axis=0))/(2))
        x_bounds = np.abs((self.points[0, :].max())-(self.points[0, :].min()))/2
        y_bounds = np.abs(self.points[1, :].max())-(self.points[1, :].min())/2
        R=np.sqrt(x_bounds**2 + y_bounds**2)
        #x_min = np.min(np.abs(x0) - np.array(x_bounds))
        #y_min = np.min(np.abs(y0) - np.array(y_bounds))
        
        #R = int(min(abs(x_min), abs(y_min)))
        g= np.zeros(math.ceil(R/dr), dtype=float)
        #g= np.zeros(dr, dtype=float)
        n=self.calculate_mean_density()
        print("R", R)
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
                intersection=self.intersection_area(point,radius_max, hull_max)-self.intersection_area(point,radius_min, hull_max)
                weight= area/intersection
                #g_r[j]=N_i/(area*n)
                g_r[j]= N_i*weight/(area*n)
                weight=math.floor(1000*weight)
                if np.abs(i-7)<=dr:
                    weights[j]=weight
                    x[j]=point[0]
                    y[j]=point[1]
            g[i]= np.mean(g_r)#N_i/(area*n)
        return g, weights, x, y, R