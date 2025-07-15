# simulation.py
from graph import Graph

class Simulation:
    def __init__(self, map_filename):
        self.cars = {}
        self.riders = {}

        # Create and load the graph
        self.map = Graph()
        self.map.load_from_file(map_filename)

if __name__ == "__main__":
    sim = Simulation("map.csv")  # Make sure this file exists
    print(sim.map.adjacency_list)
