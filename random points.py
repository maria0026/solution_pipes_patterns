import prepare_data
import pandas as pd 
import numpy as np
import voronoi_analysis
import plots
import matplotlib.pyplot as plt


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
    
    df = prepare_data.read_data("points_random.dat")
    voronoi_preprocessor = prepare_data.VoronoiPreprocess(df)
    areas=voronoi_preprocessor.calculate_areas()
    updated_data = voronoi_preprocessor.mark_points_without_regions()
    updated_data = voronoi_preprocessor.mark_points_with_big_area(areas, area_limit = 14)
    np.savetxt("new_points_random.dat", updated_data, fmt="%.3f")
    

def main():
    df = prepare_data.read_data("new_points_random.dat", preprocessed=True)
    voronoi_analyser = voronoi_analysis.VoronoiAnalyser(df)
    voronoi_plotter = plots.Voronoi_Plotter(df)

    voronoi_plotter.all_voronoi_diagram()
    voronoi_plotter.all_voronoi_diagram_area_filtered(area_limit = 14)

    areas = df.loc[df["Point of Voronoi"] == 1, "Area"]
    voronoi_plotter.areas_hist(areas)

    sides=voronoi_analyser.calculate_sides()
    voronoi_plotter.sides_number_hist(sides)
    
    distances=voronoi_analyser.calculate_distance_between_neighbours()
    voronoi_plotter.distance_between_neighbours_hist(distances)

    order = voronoi_analyser.calculate_orientational_order()
    voronoi_plotter.order_hist(order)
    

if __name__ == "__main__":
    #prepare_mock_data()
    main()