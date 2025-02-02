import argparse
import prepare_data
import pandas as pd 
import numpy as np
import voronoi_analysis
import plots
import matplotlib.pyplot as plt

def main(args):
    df = prepare_data.read_data(args.data_path)
    voronoi_analyser = voronoi_analysis.VoronoiAnalyser(df)
    # Testowanie, które punkty mają pełne komórki
    points = df[["Center x coordinate", "Center y coordinate"]].dropna().values
    full_cells = []
    for i, point in enumerate(points):
        if voronoi_analyser.is_point_inside_voronoi(i):
            full_cells.append(i)

    print("Punkty z pełnymi komórkami Voronoja:", len(full_cells))


    voronoi_ploter = plots.Voronoi_Plotter(df)

    voronoi_ploter.all_voronoi_diagram()
    voronoi_ploter.all_voronoi_diagram_area_filtered()

    areas=voronoi_analyser.calculate_areas()
    areas=voronoi_analyser.filter_by_area(areas, args.area_limit)
    voronoi_ploter.areas_hist(areas)

    sides=voronoi_analyser.calculate_sides()
    voronoi_ploter.sides_number_hist(sides)

    distances=voronoi_analyser.calculate_distance_between_neighbours()
    voronoi_ploter.distance_between_neighbours_hist(distances)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("parser for solution pipes pattern analysis")
    parser.add_argument("--data_path", nargs="?", default="pipes3.dat", help="path of data", type=str)
    parser.add_argument("--area_limit", nargs="?", default=14.0, help="limit of area", type=float)
    args = parser.parse_args()
    main(args)
