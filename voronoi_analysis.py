import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi,  voronoi_plot_2d
import math

class VoronoiAnalyser:
    def __init__(self, df):
        print("Voronoi analyser initialized")
        self.df=df
        self.points = np.column_stack((df["Center x coordinate"], df["Center y coordinate"]))
        self.voronoi = Voronoi(self.points)
        self.vertices = self.voronoi.vertices
        self.regions = self.voronoi.regions
        self.point_to_region = self.voronoi.point_region
        self.ridge_points = self.voronoi.ridge_points

    def all_voronoi_diagram(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        voronoi_plot_2d(self.voronoi, ax=ax, show_vertices=False, line_width=2, line_colors='blue')
        ax.scatter(self.points[:, 0], self.points[:, 1], color='red', zorder=5, label="Points")
        ax.set_title("Voronoi Diagram", fontsize=16)
        ax.set_xlabel("X Coordinate", fontsize=14)
        ax.set_ylabel("Y Coordinate", fontsize=14)
        ax.grid(True)
        ax.legend()
        #Zooming in:
        ax.set_xlim(531925, 531975)
        ax.set_ylim(5752275, 5752325)
        plt.show()

    def count_sides(self):
        sides= [len(region) for region in self.regions]
        number_of_bins = np.max(sides)-np.min(sides)+1
        plt.hist(sides, bins=number_of_bins, edgecolor='black')
        plt.xticks(range(12))
        plt.locator_params(axis='y', integer=True)
        plt.xlim(0, 12)
        plt.xlabel("Number of sides")
        plt.ylabel("Number of regions")
        plt.show()
        return sides

    def calculate_areas(self):
        areas=[]
        for region in self.regions:
            if -1 in region: #-1 is the index of the region that is unbounded
                pass
            else:
                area=self.calculate_polygon_area(region)
                areas.append(area)
                
        plt.hist(areas, bins=20, range=(0,13), edgecolor='black')
        plt.xticks(range(13))
        plt.xlabel("Area")
        plt.ylabel("Number of regions")
        plt.xlim(0,13)
        plt.show()
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
    

    def calculate_distance_between_neighbours(self):
        distances = []
        checked_pairs = set()  
        for ridge in self.ridge_points:
            point1_index, point2_index = ridge  
            if (point1_index, point2_index) in checked_pairs or (point2_index, point1_index) in checked_pairs:
                continue  
            point1 = self.points[point1_index]
            point2 = self.points[point2_index]
            distance = np.linalg.norm(point1 - point2)
            distances.append(distance)
            checked_pairs.add((point1_index, point2_index))

        distances = np.array(distances)
        number_of_bins = int(np.max(distances)-np.min(distances))
        plt.hist(distances, bins=number_of_bins, edgecolor='black')
        plt.locator_params(axis='y', integer=True)
        plt.xlim(0,30)
        #plt.xlim(0, math.ceil(np.max(distances))) #250
        plt.xlabel("Distance")
        #plt.ylabel("Number of ?")
        plt.show()
        return distances

        
        
                    

