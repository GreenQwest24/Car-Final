import heapq

def find_shortest_path(graph, start_node, end_node):
    """
    Implements Dijkstra's Algorithm to find the shortest path in a weighted graph.

    Parameters
    ----------
    graph : dict
        Adjacency list representation of the graph: {node: [(neighbor, weight), ...]}
    start_node : str
        The starting node.
    end_node : str
        The destination node.

    Returns
    -------
    tuple
        (path as list of nodes, total distance).
        If no path exists, returns (None, float('inf')).
    """
    heap = [(0, start_node)]
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    predecessors = {node: None for node in graph}

    while heap:
        current_distance, current_node = heapq.heappop(heap)

        if current_node == end_node:
            # Reconstruct path
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = predecessors[current_node]
            return (path[::-1], current_distance)

        for neighbor, weight in graph.get(current_node, []):
            new_distance = current_distance + weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                predecessors[neighbor] = current_node
                heapq.heappush(heap, (new_distance, neighbor))

    return (None, float('inf'))

# --- Add the following code to call the function and test it ---

if __name__ == "__main__":
    # Example graph (you'd typically load this from a file or define it more robustly)
    sample_graph = {
        'A': [('B', 1), ('C', 4)],
        'B': [('A', 1), ('C', 2)],
        'C': [('A', 4), ('B', 2), ('D', 1)],
        'D': [('B', 1), ('D', 5)],
    }

    start_node = 'C'
    end_node = 'D'

    path, distance = find_shortest_path(sample_graph, start_node, end_node)

    if path:
        print(f"Shortest path from {start_node} to {end_node}: {path}")
        print(f"Total distance: {distance}")
    else:
        print(f"No path found from {start_node} to {end_node}")