import argparse
import prepare_data
import pandas as pd 
from voronoi_diagrams import voronoi_diagrams



def main(args):
    df = prepare_data.read_data(args.data_path)
    voronoi_diagrams(df)
    
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser("parser for solution pipes pattern analysis")
    parser.add_argument("--data_path", nargs="?", default="pipes3.dat", help="path of data", type=str)
    args = parser.parse_args()
    main(args)
