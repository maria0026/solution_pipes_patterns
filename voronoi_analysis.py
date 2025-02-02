import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi,  voronoi_plot_2d
import math

class VoronoiAnalyser:
    def __init__(self, df):
        print("Voronoi analyser initialized")
        self.df=df
        self.df = self.df.dropna(subset=['Center x coordinate', 'Center y coordinate']).drop_duplicates()
        self.points = np.column_stack((df["Center x coordinate"], df["Center y coordinate"]))
        self.voronoi = Voronoi(self.points)
        self.vertices = self.voronoi.vertices
        self.regions = self.voronoi.regions
        self.point_to_region = self.voronoi.point_region
        self.ridge_points = self.voronoi.ridge_points
        print(len(self.regions))

    def is_point_inside_voronoi(self, point):
        """ Sprawdzenie, czy punkt znajduje się wewnątrz diagramu Voronoja """
        region_idx = self.voronoi.point_region[point]
        region = self.voronoi.regions[region_idx]
        
        if -1 in region:  # Jeśli region zawiera -1, oznacza to, że komórka jest otwarta (nieskończona)
            return False
        return True


    def calculate_areas(self):
        areas=[]
        self.df['good_point']={}
        for region in self.regions:
            #if -1 in region: #-1 is the index of the region that is unbounded
                #pass
            #else:
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
    
    def filter_by_area(self, areas, area_limit):
        for i, area in enumerate(areas):
            if area>area_limit:
                self.df.loc[i, 'good_point']=0
            else:
                self.df.loc[i, 'good_point']=1
        areas=[area for area in areas if area<=area_limit]
        print(self.df)
        return areas

    def calculate_sides(self):
        sides=[]
        for i, region in enumerate(self.regions):
            area=self.calculate_polygon_area(region)
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




