import pandas as pd
import numpy as np
from voronoi import BaseVoronoi

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
        columns = ["Pipe radius", "Center x coordinate", "Center y coordinate", "Point of Voronoi", "Area", "Region index"]
    else:
        columns = ["Pipe radius", "Center x coordinate", "Center y coordinate"]

    df = pd.DataFrame(data, columns=columns)
    
    df["Pipe radius"] = pd.to_numeric(df["Pipe radius"], errors="coerce")
    df["Center x coordinate"] = pd.to_numeric(df["Center x coordinate"], errors="coerce")
    df["Center y coordinate"] = pd.to_numeric(df["Center y coordinate"], errors="coerce")

    if preprocessed:
        df["Point of Voronoi"] = pd.to_numeric(df["Point of Voronoi"], errors="coerce")
        df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
        df["Region index"] = pd.to_numeric(df["Region index"], errors="coerce")

    return df

#function for preparing random data
def prepare_mock_data():
    x_min = 531840.694
    x_max = 532015.134
    y_min = 5752195.773
    y_max = 5752549.147

    pipe_radius = np.ones(5356).reshape(-1, 1)
    x_random_points = np.random.uniform(x_min, x_max, size = 5356).reshape(-1, 1)
    y_random_points = np.random.uniform(y_min, y_max, size = 5356).reshape(-1, 1)
    
    points = np.hstack((pipe_radius, x_random_points, y_random_points))
    print(np.shape(points))
    np.savetxt("points_random.dat", points, fmt="%.3f")
    
    df = read_data("points_random.dat")
    voronoi_preprocessor = VoronoiPreprocess(df)
    areas=voronoi_preprocessor.calculate_areas()
    updated_data = voronoi_preprocessor.mark_points_without_regions()
    updated_data = voronoi_preprocessor.mark_points_with_big_area(areas, area_limit = 14)
    np.savetxt("new_points_random.dat", updated_data, fmt="%.3f")


class VoronoiPreprocess(BaseVoronoi):
    def __init__(self, df):
        super().__init__(df)
        print("Voronoi preprocessor initialized")
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
            df_copy.loc[i, 'Region index']= region_index
        return df_copy

