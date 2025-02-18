import argparse
import prepare_data
import pandas as pd 
import numpy as np


def main(args):
    df = prepare_data.read_data(args.data_path)
    voronoi_preprocessor = prepare_data.VoronoiPreprocess(df)
    areas=voronoi_preprocessor.calculate_areas()
    updated_data = voronoi_preprocessor.mark_points_without_regions()
    updated_data = voronoi_preprocessor.mark_points_with_big_area(areas, args.area_limit)
    print(updated_data)
    np.savetxt("new_points.dat", updated_data, fmt="%.3f")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("parser for solution pipes pattern analysis")
    parser.add_argument("--data_path", nargs="?", default="pipes3.dat", help="path of data", type=str)
    parser.add_argument("--area_limit", nargs="?", default=14.0, help="limit of area", type=float)
    args = parser.parse_args()
    main(args)
