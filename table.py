import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
import sys
import prepare_data
import voronoi_analysis


class Area:
    
    '''Class aiming to calculate statistics for data set of solution pipes'''
    
    def __init__(self, path):
        
        self.data = pd.read_csv(path, sep=r'\s+')
        self.data.drop_duplicates()
        self.data['area'] = self.data['R'] ** 2 * np.pi 
        
        self.data['X'] = self.data['X'] -\
            np.sum(self.data['X']) / len(self.data['X'])
        self.data['Y'] = self.data['Y'] - \
            np.sum(self.data['Y']) / len(self.data['Y'])
        
        
    def calculate_radius_stat(self, path = 'promienie/'):
        
        # average radius
        self.R = np.average(self.data['R']) 
        # max and min radius
        self.maxR = np.max(self.data['R']) 
        self.minR = np.min(self.data['R'])
        
        # standard deviation, kurtosis and skewness
        self.kurtosis = kurtosis(self.data['R'])
        self.R_stdev = np.std(self.data['R'])
        self.skewness = skew(self.data['R'])
        
        
        
    def density(self):
        # calculate pipe density in whole area
        
        points = self.data[['X', 'Y']].values
        hull = ConvexHull(points)
        area = hull.volume
        
        self.density1 = len(points) / area # N / area
        self.density2 = np.sum(self.data['area']) / area # area percentage
        
    
    def calculate_distance_between_neighbours(self):
        # average distance between neighbours in the Voronoi sense
        
        points = self.data[['X', 'Y']].to_numpy() # dataframe to array
        vor = Voronoi(points)
        
        self.d = 0
        for p1, p2 in vor.ridge_points:
           # euclidean distance 
           self.d += np.linalg.norm(vor.points[p1] - vor.points[p2])
           
        self.d /= len(vor.ridge_points) # average distance
    
    def orientations(self, path = 'orientacje/'):
        # create histogram of angles between neighbours in Voronoi sense
        # path - save path
        
        points = self.data[['X', 'Y']].to_numpy()
        vor = Voronoi(points)
        angles = []  # List to store angles

        for p1, p2 in vor.ridge_points:
            
            p1p, p2p = vor.points[p1], vor.points[p2] 
            x1, y1 = p1p[0], p1p[1]
            x2, y2 = p2p[0], p2p[1]
            angle = np.arctan2(y2 - y1, x2 - x1)  # Compute the angle in rad
            angles.append(angle)  # Store the angle

        # Plot the histogram
        plt.hist(angles, bins=30, edgecolor='black')  
        plt.xlabel("Angle (radians)")
        plt.ylabel("Count")
        plt.title("Histogram of Angles")
        plt.savefig(path + self.name + '.png')
        plt.close()
    
    def R_dist(self, path = 'promienie/'):
        # Plot histograms of radii
        # path - save path
        
        points = self.data['R'].to_numpy()
        plt.hist(points, bins=30, edgecolor='black')
        plt.xlabel("Radius [m]")
        plt.ylabel("Count")
        plt.title("Histogram of Radii")
        plt.savefig(path + self.name + '.png')
        plt.close()
        
    
    def max_density(self, side = 10):
        # calculate max density within squares looping over whole area
        # side - side of each square
        
        self.maxdensity1, self.maxdensity2 = 0, 0
        x_max = np.max(self.data['X'])
        x_min = np.min(self.data['X'])
        y_max = np.max(self.data['Y'])
        y_min = np.min(self.data['Y'])
        
        size_x = x_max - x_min + 2 * side
        size_y = y_max - y_min + 2 * side
        
        
        start = np.array([x_min - 5, y_min - 5])

        # loops over the whole area
        for x in range(int(size_x) // side + 1):
            for y in range(int(size_y) // side + 1):
                
                new_start = start + np.array([x * side, y * side])
                
                x0, y0 = new_start[0], new_start[1]
                x1, y1 = x0 + side, y0 + side
                
                # pipes inside the square
                filtered_data = self.data[(self.data["X"] >= x0) & 
                                          (self.data["X"] <= x1) & 
                                          (self.data["Y"] >= y0) & 
                                          (self.data["Y"] <= y1)]
                
                
                points = filtered_data[['X', 'Y']].values
     
                area = side ** 2
                new_density1 = len(points) / area
                new_density2 = np.sum(filtered_data['area']) / area
                
                if new_density1 > self.maxdensity1:
                    self.maxdensity1 = new_density1 # N / m^2
                    
                if new_density2 > self.maxdensity2:
                    self.maxdensity2 = new_density2 # percentage of the area
                     
    def calculate_hexatic_order(self, path):
        df = prepare_data.read_data(path, preprocessed=True)
        voronoi_analyser = voronoi_analysis.VoronoiAnalyser(df)
        updated_df = voronoi_analyser.calculate_orientational_order(absolute = 
                                                                    False)
        order=updated_df.loc[df['Point of Voronoi']==1, 'Hexatic order']
        self.hexatic = np.average(order)
                
        
NAMES = [
'Apulia_Novaglie.dat',
'Apulia_Santa_Sabina.dat',
'Apulia_Santa_Sabina_Area1.dat',
'Apulia_Santa_Sabina_Area2.dat',
'Apulia_Santa_Sabina_Area3.dat',
'Apulia_Santa_Sabina_Area4.dat',
'Apulia_day1.dat',
'Apulia_day1_Area1.dat',
'Apulia_day1_Area2.dat',
'Apulia_day2.dat',
'Apulia_day2_Area1.dat',
'Apulia_day3.dat',
'Apulia_day3_Area1.dat',
'Apulia_day4.dat',
'Australia_Portland_01.dat',
'Australia_Portland_01_Area1.dat',
'Australia_Portland_01_Area2.dat',
'Australia_Portland_01_Area3.dat',
'Australia_Portland_01_Area4.dat',
'Australia_Portland_01_Area5.dat',
'Australia_Portland_01_Area6.dat',
'Australia_Portland_02_Springs.dat',
'Australia_Portland_03_MIS5.dat',
'Australia_Portland_04_West.dat',
'Australia_Portland_05_West.dat',
'Bermuda_Church_Bay.dat',
'Bermuda_Devonshire_Bay_East.dat',
'Bermuda_Devonshire_Bay_East_1.dat',
'Bermuda_Devonshire_Bay_East_2.dat',
'Bermuda_Devonshire_Bay_West.dat',
'Bermuda_Devonshire_Bay_West_1.dat',
'Bermuda_Whalebone_Bay.dat',
'Crete.dat',
'Crete_Area1.dat',
'Sicily.dat',
'Turkey_Region1.dat',
'Turkey_Region1_Area1.dat',
'Turkey_Region1_Area2.dat',
'Turkey_Region2.dat'
]

data = {}
    
for name in NAMES:
    
    print('Currently processing:', name)
    
    # creating object
    path = f'processed/{name}'
    data[name] = Area(path)
    data[name].name = name[:-4]
    
    # calculating statistics
    data[name].calculate_radius_stat()
    data[name].density()
    data[name].max_density(side = 5)
    data[name].calculate_distance_between_neighbours()
    data[name].calculate_hexatic_order(path = f'new_processed/{name}')
    
    # generating histograms
    data[name].orientations()
    data[name].R_dist()


   
import csv
csv_file = 'table.csv'

# Open the CSV file for writing
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header row
    writer.writerow(['name', 'average pipe R [m]', 'max R', 'min R', 'R stdev',
                     'R kurtosis', 'R skewness', 'density1 [N / m^2]', 
                     'density2 [area percentage]', 
                     'average distance between neighbours', 
                     'maxdensity1 (N/m^2 max value within squares)', 
                     'maxdensity2 (percentage max value within squares)',
                     'hexatic order'])

#     # Write data from each object in the dictionary
    for obj in data.values():
        writer.writerow([obj.name, obj.R, obj.maxR, obj.minR, obj.R_stdev, 
                         obj.kurtosis, obj.skewness,
                         obj.density1, obj.density2, obj.d,
                         obj.maxdensity1, obj.maxdensity2, obj.hexatic])
        