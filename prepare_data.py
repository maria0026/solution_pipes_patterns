import pandas as pd

def read_data(path):
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