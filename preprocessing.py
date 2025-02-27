import argparse
import prepare_data
import pandas as pd 
import numpy as np


def main(args):
    #preprocessing of our data
    df = prepare_data.read_data(args.data_path, preprocessed=False)
    voronoi_preprocessor = prepare_data.VoronoiPreprocess(df)
    areas=voronoi_preprocessor.calculate_areas()
    updated_data = voronoi_preprocessor.mark_points_without_regions()
    updated_data = voronoi_preprocessor.mark_points_with_big_area(areas, args.area_limit)
    print(updated_data)
    np.savetxt("new_points.dat", updated_data, fmt="%.3f")


    #generating and preprocessing random data
    random_df=prepare_data.prepare_mock_data(args.x_min, args.x_max, args.y_min, args.y_max, args.num_points)
    voronoi_preprocessor = prepare_data.VoronoiPreprocess(random_df)
    areas=voronoi_preprocessor.calculate_areas()
    updated_data = voronoi_preprocessor.mark_points_without_regions()
    updated_data = voronoi_preprocessor.mark_points_with_big_area(areas, area_limit = args.area_limit)
    np.savetxt("new_points_random.dat", updated_data, fmt="%.3f")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("parser for solution pipes pattern analysis")
    parser.add_argument("--data_path", nargs="?", default="pipes3.dat", help="path of data", type=str)
    parser.add_argument("--area_limit", nargs="?", default=14.0, help="limit of area", type=float)
    parser.add_argument("--x_min", nargs="?", default=531840.694, help="Minimum X coordinate", type=float)
    parser.add_argument("--x_max", nargs="?", default=532015.134, help="Maximum X coordinate", type=float)
    parser.add_argument("--y_min", nargs="?", default=5752195.773, help="Minimum Y coordinate", type=float)
    parser.add_argument("--y_max", nargs="?", default=5752549.147, help="Maximum Y coordinate", type=float)
    parser.add_argument("--num_points", nargs="?", default=5356, help="Number of points", type=int)

    args = parser.parse_args()
    main(args)
