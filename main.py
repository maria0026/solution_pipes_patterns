import argparse
import prepare_data

def main(args):
    prepare_data.read_data(args.data_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("parser for solution pipes pattern analysis")
    parser.add_argument("--data_path", nargs="?", default="pipes3.dat", help="path of data", type=str)
    args = parser.parse_args()
    main(args)
