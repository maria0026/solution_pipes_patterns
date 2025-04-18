import plots
import prepare_data
import voronoi_analysis
import numpy as np
import matplotlib.pyplot as plt
import os



folder_path = "new_processed_data/"
file_list = [f for f in os.listdir(folder_path) if not f.startswith("random")]
print(file_list)

for file in file_list:
    data_path = os.path.join(folder_path, file)
    file = file[:-4] #bez .dat

    print(file)
    if file == "Australia_Portland_01_Area2":
        data_path=folder_path + file + ".dat"
        df = prepare_data.read_data(data_path, preprocessed=True)

        voronoi_analyser = voronoi_analysis.VoronoiAnalyser(df)
        voronoi_plotter = plots.Voronoi_Plotter(df)

        x_min, x_max = df["Center x coordinate"].min(), df["Center x coordinate"].max()
        y_min, y_max = df["Center y coordinate"].min(), df["Center y coordinate"].max()


        updated_df = voronoi_analyser.calculate_orientational_order(absolute = False)
        order=updated_df.loc[df['Point of Voronoi']==1, 'Hexatic order']
        '''
        plt.hist(order, bins = 20,  edgecolor='black', label=f'Mean {np.mean(order):.2f}, std {np.std(order):.2f}')
        plt.title("Orientational order " +file)
        plt.legend()
        plt.xlabel("Order parameter")
        plt.ylabel("Number of points")
        plt.savefig("Figures/Histogram " + file)'
        '''

        #n_splits = int(input(file + " - na ile części podzielić?"))
        #if  n_splits == 0:
        #    continue
        voronoi_plotter.hexatic_order_split(order, grid_n=1, name=file + " 2")
    #voronoi_plotter.hexatic_order(order, x_lim_min=x_min, x_lim_max=x_max, y_lim_min=y_min, y_lim_max=y_max, save = True, name = file)