import argparse
import prepare_data
import pandas as pd 
import numpy as np
from scipy.spatial import Voronoi,  voronoi_plot_2d

class VoronoiPreprocess:
    def __init__(self, df):
        print("Voronoi preprocessor initialized")
        self.df=df
        self.df = self.df.dropna(subset=['Center x coordinate', 'Center y coordinate']).drop_duplicates()
        self.points = np.column_stack((df["Center x coordinate"], df["Center y coordinate"]))
        self.voronoi = Voronoi(self.points)
        self.regions= self.voronoi.regions
        self.point_to_region = self.voronoi.point_region
        print("Number of points: ", len(self.points))
        print("Number of regions: ", len(self.regions))
        print("Number of connections :", len(self.point_to_region))


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
        updated_data = np.hstack((self.df, new_column))
        np.savetxt("new_points.dat", updated_data, fmt="%.3f")
        return len(region_list)



def main(args):
    df = prepare_data.read_original_data(args.data_path)
    voronoi_preprocessor = VoronoiPreprocess(df)
    print("Points that have a region: ", voronoi_preprocessor.mark_points_without_regions())



if __name__ == "__main__":
    parser = argparse.ArgumentParser("parser for solution pipes pattern analysis")
    parser.add_argument("--data_path", nargs="?", default="pipes3.dat", help="path of data", type=str)
    parser.add_argument("--area_limit", nargs="?", default=14.0, help="limit of area", type=float)
    args = parser.parse_args()
    main(args)
