import numpy as np
import math
from shapely.geometry import Point, Polygon
import voronoi_analysis
def find_points(x_list, y_list, x_center, y_center, radius_min, radius_max):
        N=0
        for x,y in zip(x_list, y_list):
            if x!=x_center and y!=y_center:
                distance = math.sqrt((x- x_center) ** 2 + (y - y_center) ** 2)
                if distance <= radius_max and distance>=radius_min:
                    N += 1
        return N
def intersection_area(circle_center, circle_radius, polygon_points):
        circle = Point(circle_center).buffer(circle_radius)
        polygon = Polygon(polygon_points)
        intersection = circle.intersection(polygon)
        return intersection.area
def radial_distribution_noise( dr, x_list,y_list):
        hull_max=[(0,0), (1,0),(1,1),(0,1)]
        weights=np.zeros(len(x_list), dtype=float)
        weight=0
        #R= int(np.min(np.ptp(self.points, axis=0))/(2))
        x_bounds = (np.abs(x_list.max())-np.abs(x_list.min()))/2
        y_bounds = (np.abs(y_list.max())-np.abs(y_list.min()))/2
        R=np.sqrt(x_bounds**2+y_bounds**2)-dr
        #x_min = np.min(np.abs(x0) - np.array(x_bounds))
        #y_min = np.min(np.abs(y0) - np.array(y_bounds))
        
        #R = int(min(abs(x_min), abs(y_min)))
        g= np.zeros(math.ceil(R/dr), dtype=float)
        #g= np.zeros(dr, dtype=float)
        n=1024
        for i, _ in enumerate(np.arange(0, R, dr)):
            g_r =np.zeros((len(x_list)))
            j=0
            for x,y in zip(x_list, y_list):
                r_i=(i+0.5)*dr
                radius_min = r_i-dr/2
                radius_max = r_i+dr/2
                area=2*np.pi*r_i*dr
                N_i= find_points(x_list,y_list, x, y, radius_min, radius_max)
                #print(r_i, area)
                # adding weights
                intersection=intersection_area((x,y),radius_max, hull_max)-intersection_area((x,y),radius_min, hull_max)
                if intersection<=area/100:
                    return g, weights, x_list, y_list
                weight= area/intersection
                #g_r[j]=N_i/(area*n)
                g_r[j]= N_i*weight/(area*n)
                if  abs(i-13)<=(0.4):
                     weights[j]=(1/weight)
                j+=1
            g[i]= np.mean(g_r)#N_i/(area*n)

        return g, weights, x_list, y_list
