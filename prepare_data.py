import pandas as pd
import numpy as np
from scipy.spatial import Voronoi
from voronoi_analysis import VoronoiAnalyser

def read_data(path, preprocessed=False):
    """
    Reads data from a file and converts it to a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the data file.
    preprocessed : bool, optional
        Whether the data is preprocessed (includes "Point of Voronoi" column).
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded data.
    """
    data = []
    with open(path, 'r') as file:
        for line in file:
            value = line.strip().split()
            data.append(value)

    if preprocessed:
        columns = ["Pipe radius", "Center x coordinate", "Center y coordinate", "Point of Voronoi", "Area"]
    else:
        columns = ["Pipe radius", "Center x coordinate", "Center y coordinate"]

    df = pd.DataFrame(data, columns=columns)
    
    df["Pipe radius"] = pd.to_numeric(df["Pipe radius"], errors="coerce")
    df["Center x coordinate"] = pd.to_numeric(df["Center x coordinate"], errors="coerce")
    df["Center y coordinate"] = pd.to_numeric(df["Center y coordinate"], errors="coerce")

    if preprocessed:
        df["Point of Voronoi"] = pd.to_numeric(df["Point of Voronoi"], errors="coerce")
        df["Area"] = pd.to_numeric(df["Area"], errors="coerce")

    return df


class VoronoiPreprocess():
    def __init__(self, df):
        print("Voronoi preprocessor initialized")
        self.df=df
        self.df = self.df.dropna(subset=['Center x coordinate', 'Center y coordinate']).drop_duplicates()
        self.points = np.column_stack((df["Center x coordinate"], df["Center y coordinate"]))
        self.voronoi = Voronoi(self.points)
        self.vertices = self.voronoi.vertices
        self.regions= self.voronoi.regions
        self.point_to_region = self.voronoi.point_region
        print("Number of points: ", len(self.points))
        print("Number of regions: ", len(self.regions))
        print("Number of connections :", len(self.point_to_region))

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

    def mark_points_without_regions(self):

        region_list = np.array([])
        valid_points = np.array([])
        for i in range(len(self.point_to_region)):
            if self.point_to_region[i] in region_list:
                valid_points = np.append(valid_points, 0)
            else:
                valid_points = np.append(valid_points, 1)
                region_list = np.append(region_list, self.point_to_region[i])

        new_column = np.reshape(valid_points, (len(self.point_to_region), 1))
        self.df["Point of Voronoi"]=new_column
        #updated_data = np.hstack((self.df, new_column))
       
        return self.df
    
    def mark_points_with_big_area(self, areas, area_limit):

        df_copy = self.df.copy()
        for i, region_index in enumerate(self.point_to_region):
            if region_index == -1 or region_index >= len(areas):
                df_copy.loc[i, 'Point of Voronoi'] = 0
                df_copy.loc[i, 'Area']= 1000
                continue
            df_copy.loc[i, 'Area']= areas[region_index]
            if areas[region_index] > area_limit:
                df_copy.loc[i, 'Point of Voronoi'] = 0 

        return df_copy

