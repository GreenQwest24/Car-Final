import csv

class Graph:
    """
    A class to represent a graph using an adjacency list.
    
    Attributes:
        adjacency_list (dict): A dictionary where keys are nodes and values are
                               lists of tuples, each containing a neighbor and the weight
                               of the edge to that neighbor.
                               Example: {'A': [('B', 10), ('C', 5)], 'B': [('A', 10)]}
    """
    def __init__(self):
        self.adjacency_list = {}

    def add_edge(self, node1, node2, weight):
        """
        Adds a bidirectional edge between two nodes with a given weight.

        Parameters:
            node1 (str): The first node.
            node2 (str): The second node.
            weight (int): The weight of the edge between the two nodes.
        """
        # Ensure both nodes exist in the adjacency list
        if node1 not in self.adjacency_list:
            self.adjacency_list[node1] = []
        if node2 not in self.adjacency_list:
            self.adjacency_list[node2] = []
        
        # Add the edge in both directions for an undirected graph
        self.adjacency_list[node1].append((node2, weight))
        self.adjacency_list[node2].append((node1, weight))

    def load_from_file(self, filename):
        """
        Reads a CSV file to populate the graph with nodes and edges.
        
        The CSV file should have a headerless format of: `node1,node2,weight`
        Example `map.csv` content:
        A,B,10
        B,C,20
        C,D,30
        
        Parameters:
            filename (str): The path to the CSV file.
        """
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                node1, node2, weight = row
                self.add_edge(node1, node2, int(weight))

# --- Execution Block ---
# This block of code will only run when the script is executed directly.
if __name__ == "__main__":
    print("--- Starting Graph Execution ---")
    
    # 1. Create a Graph instance
    my_graph = Graph()
    print("Graph object created.")

    # 2. Add some edges manually to test the `add_edge` method
    print("\nAdding edges manually...")
    my_graph.add_edge('A', 'B', 10)
    my_graph.add_edge('B', 'C', 20)
    print("Manual edges added.")

    # 3. Print the adjacency list to see the result
    print("\nGraph after manual edges:")
    print(my_graph.adjacency_list)

    # 4. Test the `load_from_file` method
    # It's assumed a 'map.csv' file exists in the same directory.
    print("\nAttempting to load graph from 'map.csv'...")
    try:
        my_graph.load_from_file('map.csv')
        print("Successfully loaded graph from 'map.csv'.")
        print("\nGraph after loading from file:")
        print(my_graph.adjacency_list)
    except FileNotFoundError:
        print("\n'map.csv' not found. Please create it to test this method.")

    print("\n--- Graph Execution Complete ---")