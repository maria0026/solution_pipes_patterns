import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor



class Area:
    def __init__(self, path):
        self.data = pd.read_csv(path, sep=r'\s+')
        self.data.drop_duplicates()
        self.data['area'] = self.data['R'] ** 2 * np.pi 
        
        self.data['X'] = self.data['X'] -\
            np.sum(self.data['X']) / len(self.data['X'])
        self.data['Y'] = self.data['Y'] - \
            np.sum(self.data['Y']) / len(self.data['Y'])
        
        
        
    def avg_R(self):
        self.R = np.average(self.data['R'])
        
    def density(self):
        points = self.data[['X', 'Y']].values
        
        hull = ConvexHull(points)
        area = hull.volume
        self.density1 = len(points) / area
        self.density2 = np.sum(self.data['area']) / area
        
    # def plot_voronoi(self):
        
    #     # Assuming the points are in 'x' and 'y' columns in the data
    #     points = self.data[['X', 'Y']].to_numpy()
        
    #     # Compute Voronoi diagram
    #     vor = Voronoi(points)
    #     # print(self.name, len(vor.regions), len(points)
        
    #     # Plot Voronoi diagram
    #     # fig, ax = plt.subplots()
    #     # voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_width=2, 
    #     #                 line_colors='blue')
        
        
    #     # # Plot points
    #     # ax.plot(points[:, 0], points[:, 1], 'r.')  # red points
        
    #     # # Set plot labels and title
    #     # ax.set_xlabel('X Coordinate')
    #     # ax.set_ylabel('Y Coordinate')
    #     # ax.set_title('Voronoi Diagram')

    #     # # Show the plot
    #     # # plt.show()
    #     # plt.savefig('output.jpg', dpi = 500)
    #     # finite_regions = []
    #     # for region in vor.regions:
    #     #     if len(region) > 0 and -1 not in region:  # Check if region does not contain -1
    #     #         finite_regions.append(region)

    #     # # Optionally, plot the Voronoi diagram excluding infinite regions
    #     # fig, ax = plt.subplots()
    #     # voronoi_plot_2d(vor, ax=ax, show_vertices=False)

    #     # # To plot only the finite regions, we can draw them manually
    #     # for region in finite_regions:
    #     #     polygon = [vor.vertices[i] for i in region]
    #     #     ax.fill(*zip(*polygon), alpha=0.4)

    #     # plt.show()
        
    #     distances = []

    #     # Iterate through each point and its neighbors
    #     for i, region in enumerate(vor.regions):
    #         print(i)
    #         # Ensure the region is not infinite
    #         if len(region) > 0 and -1 not in region:  
    #             # Get the neighbors of the current point
    #             for neighbor_index in vor.neighbors[i]:
    #                 # Calculate the Euclidean distance between the points
    #                 distance = np.linalg.norm(vor.points[i] -\
    #                     vor.points[neighbor_index])
    #                 distances.append(distance)

    #     # Calculate the average distance
    #     average_distance = np.mean(distances)
    
    def calculate_distance_between_neighbours(self):
        
        points = self.data[['X', 'Y']].to_numpy()
        vor = Voronoi(points)
        # print(len(vor.ridge_points))
        
        self.d = 0
        
        for p1, p2 in vor.ridge_points:
        #    print(p1, p2)
           self.d += np.linalg.norm(vor.points[p1] - vor.points[p2])
           
        self.d /= len(vor.ridge_points)
    
    def orientations(self, path = 'orientacje/'):
        points = self.data[['X', 'Y']].to_numpy()
        vor = Voronoi(points)
        angles = []  # List to store angles

        # for (x1, y1, x2, y2) in some_iterator:  # Replace with your loop logic
        for p1, p2 in vor.ridge_points:
            
            p1p, p2p = vor.points[p1], vor.points[p2] 
            x1, y1 = p1p[0], p1p[1]
            x2, y2 = p2p[0], p2p[1]
            angle = np.arctan2(y2 - y1, x2 - x1)  # Compute the angle in radians
            angles.append(angle)  # Store the angle

        # Plot the histogram
        plt.hist(angles, bins=30, edgecolor='black')  
        plt.xlabel("Angle (radians)")
        plt.ylabel("Count")
        plt.title("Histogram of Angles")
        # plt.show()
        plt.savefig(path + self.name + '.png')
        plt.close()
        
    
    def max_density(self, side = 10):
        
        self.maxdensity1, self.maxdensity2 = 0, 0
        x_max = np.max(self.data['X'])
        x_min = np.min(self.data['X'])
        y_max = np.max(self.data['Y'])
        y_min = np.min(self.data['Y'])
        
        # print(x_max, x_min, y_max, y_min)
        # return 0
        
        size_x = x_max - x_min + 2 * side
        size_y = y_max - y_min + 2 * side
        
        
        start = np.array([x_min - 5, y_min - 5])
        count = 0
        for x in range(int(size_x) // side + 1):
            for y in range(int(size_y) // side + 1):
                count +=1
                new_start = start + np.array([x * side, y * side])
                
                x0, y0 = new_start[0], new_start[1]
                x1, y1 = x0 + side, y0 + side
                # print(x0, x1, y0, y1)
                # print(self.data[(self.data["X"] >= x0) & 
                #                           (self.data["X"] <= x1) & 
                #                           (self.data["Y"] >= y0) & 
                #                           (self.data["Y"] <= y1)])
                
                filtered_data = self.data[(self.data["X"] >= x0) & 
                                          (self.data["X"] <= x1) & 
                                          (self.data["Y"] >= y0) & 
                                          (self.data["Y"] <= y1)]
                
                # print(filtered_data)
                
                # plt.plot(self.data[['X']].values, self.data[['Y']].values, 'b.')
                # plt.plot(x0, y0, 'r.')
                # plt.plot(x0, y1, 'r.')
                # plt.plot(x1, y1, 'r.')
                # plt.plot(x1, y0, 'r.')
                
                # plt.show()
                
                
                # continue
                
                # print(x0, x1, y0, y1)
                # print(self.data[(self.data["X"] >= x0) & 
                #                           (self.data["X"] <= x1) & 
                #                           (self.data["Y"] >= y0) & 
                #                           (self.data["Y"] <= y1)])
                # break
                
                # print(x0, x1, y0, y1)
                # continue
                
               
                # if len(filtered_data)<3:
                #     # print("empty", x0, x1, y0, y1)
                #     continue
                
                points = filtered_data[['X', 'Y']].values
        
                # hull = ConvexHull(points)
                # area = hull.volume
                area = side ** 2
                new_density1 = len(points) / area
                new_density2 = np.sum(filtered_data['area']) / area
                
                if new_density1 > self.maxdensity1:
                    self.maxdensity1 = new_density1
                    
                if new_density2 > self.maxdensity2:
                    self.maxdensity2 = new_density2
        # print(count)
                
        


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
    
    print(name)
    
    
    path = f'processed/{name}'
    data[name] = Area(path)
    data[name].name = name[:-4]

    
    data[name].orientations()
    continue
    
    data[name].avg_R()
    data[name].density()
    
    data[name].max_density(side = 5)
    
    # print(data[name].data.head) 
    # print(data[name].name, data[name].density2)
    # print(name[:-4])
    
   
    data[name].calculate_distance_between_neighbours()
import csv
csv_file = 'tabelka.csv'

# Open the CSV file for writing
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header row
    writer.writerow(['name', 'average R', 'density1', 'density2', 
                     'average distance', 'maxdensity1', 'maxdensity2'])

#     # Write data from each object in the dictionary
    for obj in data.values():
        writer.writerow([obj.name, obj.R, obj.density1, obj.density2, obj.d,
                         obj.maxdensity1, obj.maxdensity2])
        
    # if(name == 'Australia_Portland_01_Area1.dat'):
        # data[name].plot_voronoi()
        # data[name].detect_lines_from_points()
        # data[name].max_density(side = 5)
        # print(data[name].maxdensity1, data[name].maxdensity2)
        
        # plt.plot(data[name].data[['X']].values, data[name].data[['Y']].values, '.')
        # plt.show()
        
        
