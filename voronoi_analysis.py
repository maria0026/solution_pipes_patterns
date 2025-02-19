import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi,  voronoi_plot_2d
import math

class VoronoiAnalyser:
    def __init__(self, df):
        print("Voronoi analyser initialized")
        self.df=df
        self.df = self.df.dropna(subset=['Center x coordinate', 'Center y coordinate', 'Point of Voronoi']).drop_duplicates()
        self.points = np.column_stack((df["Center x coordinate"], df["Center y coordinate"]))
        self.good_point=df["Point of Voronoi"]
        self.voronoi = Voronoi(self.points)
        self.vertices = self.voronoi.vertices
        self.regions = self.voronoi.regions
        self.point_to_region = self.voronoi.point_region
        self.ridge_points = self.voronoi.ridge_points
        self.voronoi_points = df["Point of Voronoi"]
        
    def calculate_areas(self):
        areas=[]
        for region in self.regions:
            area=self.calculate_polygon_area(region)
            areas.append(area)
        return areas

    def calculate_polygon_area(self, region):
        points = []
        for point in region:
            x=self.vertices[point, 0]
            y=self.vertices[point, 1]
            points.append([x, y])
        lines = np.hstack([points,np.roll(points,-1,axis=0)])
        area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
        return area

    def calculate_sides(self):
        sides=[]
        for i, region in enumerate(self.regions):
            if self.df.loc[i,'good_point']==1:
                sides.append(len(region))
        return sides

    def calculate_distance_between_neighbours(self):
        distances = []
        checked_pairs = set()  
        for ridge in self.ridge_points:
            point1_index, point2_index = ridge  
            if (point1_index, point2_index) in checked_pairs or (point2_index, point1_index) in checked_pairs:
                continue 
            if (self.df.loc[point1_index, 'good_point']==1 and self.df.loc[point2_index, 'good_point']==1):
                point1 = self.points[point1_index]
                point2 = self.points[point2_index]
                distance = np.linalg.norm(point1 - point2)
                distances.append(distance)
                checked_pairs.add((point1_index, point2_index))

        distances = np.array(distances)
        return distances


    def calculate_orientational_order(self):

        psi = np.zeros(len(self.points), dtype=complex)
        
        for j, region_idx in enumerate(self.point_to_region):  
            region = self.regions[region_idx]
            if -1 in region or len(region) == 0:
                continue 

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

        return np.abs(psi)





