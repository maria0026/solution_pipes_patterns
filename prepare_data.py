import pandas as pd
import numpy as np
from voronoi import BaseVoronoi

def read_data(path, preprocessed=False, add_geometric_center = False):
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



    if add_geometric_center:
        x_center= df["Center x coordinate"].mean()
        y_center= df["Center y coordinate"].mean()
        df["Center x coordinate"] = df["Center x coordinate"] - x_center
        df["Center y coordinate"] = df["Center y coordinate"] - y_center


    if preprocessed:
        df["Point of Voronoi"] = pd.to_numeric(df["Point of Voronoi"], errors="coerce")
        df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
        df["Region index"] = pd.to_numeric(df["Region index"], errors="coerce")

    return df

#function for preparing random data
def prepare_mock_data(x_min, x_max, y_min, y_max, num_poins):

    pipe_radius = np.ones(num_poins).reshape(-1, 1)
    x_random_points = np.random.uniform(x_min, x_max, size = num_poins).reshape(-1, 1)
    y_random_points = np.random.uniform(y_min, y_max, size = num_poins).reshape(-1, 1)
    
    x_center = np.mean(x_random_points)
    y_center = np.mean(y_random_points)
    x_random_points = x_random_points - x_center
    y_random_points = y_random_points - y_center

    points = np.hstack((pipe_radius, x_random_points, y_random_points))
    random_df = pd.DataFrame(points, columns=["Pipe radius", "Center x coordinate", "Center y coordinate"])

    return random_df

def generate_hexagonal_grid(rows, cols, spacing=1.0):
        points = []
        for row in range(rows):
            for col in range(cols):
                x = col * spacing * np.sqrt(3)/2  # Horizontal spacing
                y = row * spacing   # Vertical spacing
                if col % 2 == 1:  # Offset every other column
                    y += spacing * 0.5
                points.append((1, x, y))  # First column filled with ones
        return np.array(points)


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
    
    def calculate_area_limit(self, areas, per=95):

        return np.percentile(areas, per)

    def mark_points_with_big_area(self, areas, area_limit):

        df_copy = self.df.copy()
        for i, region_index in enumerate(self.point_to_region):

            df_copy.loc[i, 'Area']= areas[region_index]
            df_copy.loc[i, 'Region index']= region_index

            if areas[region_index] > area_limit:
                df_copy.loc[i, 'Point of Voronoi'] = 0 

        return df_copy


