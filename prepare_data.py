def read_data(path):
    with open(path, 'r') as file:
        for line in file:
            print(line.strip())
