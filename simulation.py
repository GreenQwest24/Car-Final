# graph.py

class Graph:
    def __init__(self):
        self.adjacency_list = {}

    def add_edge(self, start_node, end_node, weight):
        if start_node not in self.adjacency_list:
            self.adjacency_list[start_node] = []
        self.adjacency_list[start_node].append((end_node, weight))

    def load_from_file(self, filename):
        import csv
        try:
            with open(filename, mode='r') as file:
                reader = csv.reader(file)
                for row in reader:
                    start, end, weight = row[0], row[1], float(row[2])
                    self.add_edge(start, end, weight)
        except FileNotFoundError:
            print(f"Map file '{filename}' not found.")





       
