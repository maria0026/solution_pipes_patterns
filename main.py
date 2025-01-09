import argparse
import prepare_data
import pandas as pd 
import numpy as np
import voronoi_analysis
import matplotlib.pyplot as plt

def main(args):
    df = prepare_data.read_data(args.data_path)
    voronoi_analyser = voronoi_analysis.VoronoiAnalyser(df)
    voronoi_analyser.all_voronoi_diagram()
    voronoi_analyser.count_sides()
    voronoi_analyser.calculate_areas()
    voronoi_analyser.calculate_distance_between_neighbours()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("parser for solution pipes pattern analysis")
    parser.add_argument("--data_path", nargs="?", default="pipes3.dat", help="path of data", type=str)
    args = parser.parse_args()
    main(args)
