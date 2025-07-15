import csv

class Graph:
    def __init__(self):
        self.adjacency_list = {}

    def add_edge(self, start_node, end_node, weight):
        if start_node not in self.adjacency_list:
            self.adjacency_list[start_node] = []
        self.adjacency_list[start_node].append((end_node, weight))

    def load_from_file(self, filename):
        try:
            with open(filename, mode='r', newline='') as csvfile:
                reader = csv.reader(csvfile)  # FIXED!
                next(reader)
                for row in reader:
                    if len(row) != 3:
                        print(f"Skipping invalid row: {row}")
                        continue
                    start, end, weight = row
                    try:
                        weight = int(weight)
                        self.add_edge(start.strip(), end.strip(), weight)
                    except ValueError:
                        print(f"Invalid weight '{weight}' in row: {row}")
        except FileNotFoundError:
            print(f"File not found: {filename}")

    def __str__(self):
        result = "Graph adjacency list:\n"
        for node, neighbors in self.adjacency_list.items():
            connections = ', '.join([f"{neighbor}({weight})" for neighbor, weight in neighbors])
            result += f"  {node} -> {connections}\n"
        return result

if __name__ == "__main__":
    g = Graph()
    g.load_from_file("map.csv")  # Double-check this path!
    print(g)
