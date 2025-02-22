# solution_pipes_patterns
The **BaseVoronoi** class is defined in the _voronoi.py_ file. It is designed to generate a Voronoi diagram based on a set of points provided in a pandas DataFrame. It stores key attributes of the Voronoi diagram, including self.vertices, self.regions, and self.point_to_region. 

To obtain analysis results, follow these steps: 
1. Run _preprocessing.py_ to read data in .dat format and generate random data. This script uses functions from the _prepare_data.py_ file.
2. Run the Jupyter notebook _results.ipynb_. It utilizes functions from _prepare_data.py_, _voronoi_analysis.py_, and _plots.py_. 
