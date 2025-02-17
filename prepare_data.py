import pandas as pd
import numpy as np

def read_original_data(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            value = line.strip().split()
            data.append(value)

    df = pd.DataFrame(data, columns = ["Pipe radius", "Center x coordinate", "Center y coordinate"] )  # Update column names
    df["Center x coordinate"] = pd.to_numeric(df["Center x coordinate"], errors="coerce")
    df["Center y coordinate"] = pd.to_numeric(df["Center y coordinate"], errors="coerce")
    df["Pipe radius"] = pd.to_numeric(df["Pipe radius"], errors="coerce")
    return df


def read_preprocessed_data(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            value = line.strip().split()
            data.append(value)

    df = pd.DataFrame(data, columns = ["Pipe radius", "Center x coordinate", "Center y coordinate", "Point of Voronoi"] )  # Update column names
    df["Center x coordinate"] = pd.to_numeric(df["Center x coordinate"], errors="coerce")
    df["Center y coordinate"] = pd.to_numeric(df["Center y coordinate"], errors="coerce")
    df["Pipe radius"] = pd.to_numeric(df["Pipe radius"], errors="coerce")
    df["Point of Voronoi"] = pd.to_numeric(df["Point of Voronoi"], errors="coerce")
    return df


def min_distance(points):
    min_dist = np.inf
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if dist < min_dist:
                min_dist = dist
    return min_dist

