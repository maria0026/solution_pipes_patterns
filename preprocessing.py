import argparse
import prepare_data
import pandas as pd 
import numpy as np


def main(args):
    #preprocessing of our data
    if args.data_path == "pipes3.dat":
        df = prepare_data.read_data(args.data_path, preprocessed=False, add_geometric_center = True)
    else:
        df = prepare_data.read_data(args.data_path, preprocessed=False, add_geometric_center = False)

    voronoi_preprocessor = prepare_data.VoronoiPreprocess(df)
    areas=voronoi_preprocessor.calculate_areas()
    updated_data = voronoi_preprocessor.mark_points_without_regions()
    updated_data = voronoi_preprocessor.mark_points_with_big_area(areas, args.area_limit)
    print(updated_data)
    
    if args.data_path.startswith("_"): 
        np.savetxt("new" + args.data_path, updated_data, fmt="%.3f")
    else:
        np.savetxt("new_" + args.data_path, updated_data, fmt="%.3f")


    #generating and preprocessing random data
    x_min, x_max = df["Center x coordinate"].min(), df["Center x coordinate"].max()
    y_min, y_max = df["Center y coordinate"].min(), df["Center y coordinate"].max()
    num_points = len(voronoi_preprocessor.points)
    random_df=prepare_data.prepare_mock_data(x_min, x_max, y_min, y_max, num_points)
    voronoi_preprocessor = prepare_data.VoronoiPreprocess(random_df)
    areas=voronoi_preprocessor.calculate_areas()
    updated_data = voronoi_preprocessor.mark_points_without_regions()
    updated_data = voronoi_preprocessor.mark_points_with_big_area(areas, area_limit = args.area_limit)

    if args.data_path.startswith("_"):
        np.savetxt("random" + args.data_path, updated_data, fmt="%.3f")
    else:
        np.savetxt("random_" + args.data_path, updated_data, fmt="%.3f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("parser for solution pipes pattern analysis")
    parser.add_argument("--data_path", nargs="?", default= "_SICILY.dat", help="path of data", type=str)
    parser.add_argument("--area_limit", nargs="?", default=14.0, help="limit of area", type=float)

    args = parser.parse_args()
    main(args)
