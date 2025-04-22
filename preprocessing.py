import argparse
import prepare_data
import pandas as pd 
import numpy as np
import os


def main(args):
    #preprocessing of our data
    if args.data_path == "data/pipes3.dat":
        df = prepare_data.read_data(args.data_path, preprocessed=False, add_geometric_center = True)
    else:
        df = prepare_data.read_data(args.data_path, preprocessed=False, add_geometric_center = True)

    voronoi_preprocessor = prepare_data.VoronoiPreprocess(df)
    areas=voronoi_preprocessor.calculate_areas()
    updated_data = voronoi_preprocessor.mark_points_without_regions()
    area_limit= voronoi_preprocessor.calculate_area_limit(areas, per=95)
    print(area_limit)
    updated_data = voronoi_preprocessor.mark_points_with_big_area(areas, area_limit)
    print(updated_data)
    
    file=args.data_path.split('/')
    folder_name = file[0]
    filename=file[1]
    if not os.path.exists(f'new_{folder_name}'):
        os.mkdir(f'new_{folder_name}')

    np.savetxt("new_" + args.data_path, updated_data, fmt="%.3f")

    #generating and preprocessing random data
    x_min, x_max = df["Center x coordinate"].min(), df["Center x coordinate"].max()
    y_min, y_max = df["Center y coordinate"].min(), df["Center y coordinate"].max()
    num_points = len(voronoi_preprocessor.points)
    num_points=10000
    x_min, x_max, y_min, y_max=0,10,0,10
    random_df=prepare_data.prepare_mock_data(x_min, x_max, y_min, y_max, num_points)
    voronoi_preprocessor = prepare_data.VoronoiPreprocess(random_df)
    areas=voronoi_preprocessor.calculate_areas()
    updated_data = voronoi_preprocessor.mark_points_without_regions()
    updated_data = voronoi_preprocessor.mark_points_with_big_area(areas, area_limit = area_limit)

    if filename.startswith("_"):
        np.savetxt(f"new_{folder_name}/random{filename}", updated_data, fmt="%.3f")
    else:
        np.savetxt(f"new_{folder_name}/random_{filename}", updated_data, fmt="%.3f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("parser for solution pipes pattern analysis")
    parser.add_argument("--data_path", nargs="?", default= "data/Apulia_Santa_Sabina_Area4.dat", help="path of data", type=str)
    parser.add_argument("--area_limit", nargs="?", default=14.0, help="limit of area", type=float)

    args = parser.parse_args()
    main(args)
