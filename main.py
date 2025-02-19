import argparse
import prepare_data
import pandas as pd 
import numpy as np
import voronoi_analysis
import plots
import matplotlib.pyplot as plt

def main(args):
    df = prepare_data.read_data(args.data_path, preprocessed=True)
    voronoi_analyser = voronoi_analysis.VoronoiAnalyser(df)
    voronoi_plotter = plots.Voronoi_Plotter(df)

    #voronoi_plotter.all_voronoi_diagram()
    #voronoi_plotter.all_voronoi_diagram_area_filtered(args.area_limit)

    areas = df.loc[df["Point of Voronoi"] == 1, "Area"]
    #voronoi_plotter.areas_hist(areas)

    sides=voronoi_analyser.calculate_sides()
    #voronoi_plotter.sides_number_hist(sides)
    
    distances=voronoi_analyser.calculate_distance_between_neighbours()
    #voronoi_plotter.distance_between_neighbours_hist(distances)

    order = voronoi_analyser.calculate_orientational_order()
    voronoi_plotter.order_hist(order)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("parser for solution pipes pattern analysis")
    parser.add_argument("--data_path", nargs="?", default="new_points.dat", help="path of data", type=str)
    parser.add_argument("--area_limit", nargs="?", default=14.0, help="limit of area", type=float)
    args = parser.parse_args()
    main(args)
