
from graph import Graph
from pathfinding import find_shortest_path

def main():
    """
    Test script for verifying Dijkstra's algorithm.
    Loads a graph from map.csv, then finds and prints the shortest path.
    """
    # Load the graph
    graph = Graph()
    graph.load_from_file('map.csv')

    # Choose start and end nodes
    start = 'A'
    end = 'D'

    # Run Dijkstra's algorithm
    path, distance = find_shortest_path(graph.adjacency_list, start, end)

    # Display results
    if path:
        print(f"Shortest path from {start} to {end}: {path}")
        print(f"Total distance: {distance}")
    else:
        print(f"No path found from {start} to {end}")


if __name__ == "__main__":
        main()