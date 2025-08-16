import argparse
import heapq
import random
import math
import collections
import os
from matplotlib import pyplot as plt

# =========================================================
# Graph Class
# =========================================================
class Graph:
    """
    Represents a road network with nodes (intersections) and weighted edges (roads).
    Provides functionality for loading from CSV, nearest-node lookup,
    and shortest-path distance calculation using Dijkstra's algorithm.
    """
    def __init__(self):
        self.adjacency_list = collections.defaultdict(list)  # {node: [(neighbor, weight), ...]}
        self.node_coordinates = {}  # {node_id: (x, y)}

    def load_map_data(self, filename):
        """
        Load graph data from a CSV file.
        If the file is missing, a fallback hardcoded triangle map is used.
        CSV format: start_node,start_x,start_y,end_node,end_x,end_y,weight
        """
        if not os.path.exists(filename):
            print(f"[WARN] Map file '{filename}' not found. Using fallback map.")
            self._load_fallback()
            return

        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue  # Skip comments and blanks
                try:
                    parts = line.strip().split(',')
                    start_id, start_x, start_y, end_id, end_x, end_y, weight = parts
                    # Store node coordinates
                    self.node_coordinates[start_id] = (float(start_x), float(start_y))
                    self.node_coordinates[end_id] = (float(end_x), float(end_y))
                    # Undirected edges
                    self.adjacency_list[start_id].append((end_id, float(weight)))
                    self.adjacency_list[end_id].append((start_id, float(weight)))
                except (ValueError, IndexError) as e:
                    print(f"[ERROR] Skipping malformed line: {line.strip()} | {e}")

    def _load_fallback(self):
        """Load a simple 3-node triangular graph (for debugging)."""
        edges = [
            ("A", 0, 0, "B", 1, 0, 1),
            ("B", 1, 0, "C", 0.5, 1, 1),
            ("C", 0.5, 1, "A", 0, 0, 1)
        ]
        for start_id, sx, sy, end_id, ex, ey, w in edges:
            self.node_coordinates[start_id] = (sx, sy)
            self.node_coordinates[end_id] = (ex, ey)
            self.adjacency_list[start_id].append((end_id, w))
            self.adjacency_list[end_id].append((start_id, w))

    def find_nearest_vertex(self, point):
        """Find the nearest graph node to a given (x, y) location."""
        x, y = point
        return min(
            self.node_coordinates,
            key=lambda n: (self.node_coordinates[n][0] - x) ** 2 +
                          (self.node_coordinates[n][1] - y) ** 2
        )

    def dijkstra(self, start, end):
        """Compute shortest path distance between two nodes using Dijkstra's algorithm."""
        dist = {node: math.inf for node in self.node_coordinates}
        dist[start] = 0
        pq = [(0, start)]  # (distance, node)
        while pq:
            d, u = heapq.heappop(pq)
            if u == end:
                return d  # Found shortest path
            if d > dist[u]:
                continue
            for v, w in self.adjacency_list[u]:
                alt = d + w
                if alt < dist[v]:
                    dist[v] = alt
                    heapq.heappush(pq, (alt, v))
        return math.inf

# =========================================================
# Rider & Car Classes
# =========================================================
class Rider:
    """Represents a rider request with start and destination coordinates."""
    def __init__(self, rider_id, start, dest):
        self.id = rider_id
        self.start = start
        self.dest = dest
        self.request_time = None
        self.pickup_time = None
        self.dropoff_time = None

class Car:
    """Represents a car in the fleet."""
    def __init__(self, car_id, location):
        self.id = car_id
        self.location = location
        self.status = "available"

# =========================================================
# Quadtree Stub
# =========================================================
class Quadtree:
    """
    A very simple spatial lookup (not a real quadtree).
    Stores objects as {id: (x, y)} and finds nearest neighbors by brute force.
    """
    def __init__(self):
        self.objects = {}

    def insert(self, obj_id, point):
        self.objects[obj_id] = point

    def remove(self, obj_id):
        if obj_id in self.objects:
            del self.objects[obj_id]

    def find_k_nearest(self, point, k=5):
        """Return the k closest objects to a given (x, y) point."""
        x, y = point
        return sorted(
            self.objects.items(),
            key=lambda kv: (kv[1][0] - x) ** 2 + (kv[1][1] - y) ** 2
        )[:k]

# =========================================================
# Simulation Class
# =========================================================
class Simulation:
    """
    Core event-driven rideshare simulation.
    Uses a priority queue to manage events like rider requests, pickups, and dropoffs.
    """
    def __init__(self, graph, max_time, mean_arrival, num_cars):
        self.time = 0
        self.events = []  # Min-heap for events
        self.graph = graph
        self.max_time = max_time
        self.mean_arrival = mean_arrival
        self.quadtree = Quadtree()
        self.cars = []
        self.riders = {}
        self.trip_log = []

        node_ids = list(graph.node_coordinates.keys())
        if not node_ids:
            print("[ERROR] No nodes available. Exiting.")
            return

        # Initialize cars at random nodes
        for i in range(num_cars):
            start_node = random.choice(node_ids)
            car = Car(f"CAR-{i}", graph.node_coordinates[start_node])
            self.cars.append(car)
            self.quadtree.insert(car.id, car.location)

        # First rider request event
        self.add_event(0, "RIDER_REQUEST", None)

    def add_event(self, delay, event_type, data):
        """Schedule a new event after `delay` time units."""
        heapq.heappush(self.events, (self.time + delay, event_type, data))

    def generate_rider_request(self):
        """Create a new rider request with random start and destination points."""
        rider_id = f"R-{len(self.riders) + 1}"
        start = (random.uniform(0, 1), random.uniform(0, 1))
        dest = (random.uniform(0, 1), random.uniform(0, 1))
        rider = Rider(rider_id, start, dest)
        rider.request_time = self.time
        self.riders[rider_id] = rider
        return rider

    def run(self):
        """Run the simulation until time exceeds max_time or no events remain."""
        while self.events and self.time < self.max_time:
            self.time, etype, data = heapq.heappop(self.events)
            if etype == "RIDER_REQUEST":
                self.handle_rider_request()
            elif etype == "PICKUP_ARRIVAL":
                self.handle_pickup(data)
            elif etype == "DROPOFF_ARRIVAL":
                self.handle_dropoff(data)
        print(f"Simulation finished at t={self.time:.2f}")

    def handle_rider_request(self):
        """Handle a new rider request event."""
        rider = self.generate_rider_request()
        print(f"[t={self.time:.2f}] New rider {rider.id}")

        # Schedule next request
        delta = random.expovariate(1.0 / self.mean_arrival)
        self.add_event(delta, "RIDER_REQUEST", None)

        # Find nearest cars
        candidates = self.quadtree.find_k_nearest(rider.start, k=5)
        best_car, best_time = None, math.inf
        for cid, loc in candidates:
            start_node = self.graph.find_nearest_vertex(loc)
            rider_node = self.graph.find_nearest_vertex(rider.start)
            travel_time = self.graph.dijkstra(start_node, rider_node)
            if travel_time < best_time:
                best_car, best_time = cid, travel_time

        if best_car:
            car = next(c for c in self.cars if c.id == best_car)
            car.status = "enroute_pickup"
            self.quadtree.remove(car.id)
            self.add_event(best_time, "PICKUP_ARRIVAL", (car, rider))
            print(f"  Dispatched {best_car} -> {rider.id} (ETA {best_time:.2f})")
        else:
            print("  No cars available!")

    def handle_pickup(self, data):
        """Handle a car arriving at the rider's pickup location."""
        car, rider = data
        rider.pickup_time = self.time
        car.status = "enroute_dropoff"
        car.location = rider.start
        print(f"[t={self.time:.2f}] {car.id} picked up {rider.id}")

        # Travel to destination
        start_node = self.graph.find_nearest_vertex(rider.start)
        dest_node = self.graph.find_nearest_vertex(rider.dest)
        travel_time = self.graph.dijkstra(start_node, dest_node)
        self.add_event(travel_time, "DROPOFF_ARRIVAL", (car, rider))

    def handle_dropoff(self, data):
        """Handle a car dropping off the rider at their destination."""
        car, rider = data
        rider.dropoff_time = self.time
        car.status = "available"
        car.location = rider.dest
        self.quadtree.insert(car.id, car.location)
        self.trip_log.append({
            "rider": rider.id,
            "wait_time": rider.pickup_time - rider.request_time,
            "trip_duration": rider.dropoff_time - rider.pickup_time
        })
        print(f"[t={self.time:.2f}] {car.id} dropped off {rider.id}")

    def analyze_and_visualize(self, output_file="simulation_summary.png"):
        """Analyze simulation results and create visualization."""
        if not self.trip_log:
            print("No trips to analyze.")
            return

        waits = [t["wait_time"] for t in self.trip_log]
        durations = [t["trip_duration"] for t in self.trip_log]
        avg_wait = sum(waits) / len(waits)
        avg_dur = sum(durations) / len(durations)

        fig, (ax_map, ax_info) = plt.subplots(
            1, 2, figsize=(12, 6),
            gridspec_kw={"width_ratios": [2, 1]}
        )

        # Car positions
        if self.cars:
            xs, ys = zip(*[c.location for c in self.cars])
            ax_map.scatter(xs, ys, c="blue", label="Cars", s=50)
        gx, gy = zip(*self.graph.node_coordinates.values())
        ax_map.scatter(gx, gy, c="gray", marker="x", label="Graph Nodes")
        ax_map.set_title("Final Car Locations")
        ax_map.legend()
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.grid(True)

        # Metrics
        ax_info.axis("off")
        ax_info.set_title("Simulation Results", fontsize=14)
        ax_info.text(0, 0.9, f"Total Trips: {len(self.trip_log)}", fontsize=12)
        ax_info.text(0, 0.8, f"Avg Wait: {avg_wait:.2f}", fontsize=12)
        ax_info.text(0, 0.7, f"Avg Duration: {avg_dur:.2f}", fontsize=12)

        # Histogram of wait times
        if waits:
            ax_hist = fig.add_axes([0.55, 0.15, 0.35, 0.3])
            ax_hist.hist(
                waits, bins=max(5, len(waits)//2),
                alpha=0.8, color="skyblue", edgecolor="darkblue"
            )
            ax_hist.set_title("Wait Time Distribution")

        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Saved results to {output_file}")

# =========================================================
# CLI Entry Point
# =========================================================
def main():
    """Parse CLI arguments and run the simulation."""
    parser = argparse.ArgumentParser(description="Run a rideshare simulation.")
    parser.add_argument("--map", default="city_map.csv", help="Path to map CSV file.")
    parser.add_argument("--max-time", type=int, default=50, help="Simulation time limit.")
    parser.add_argument("--mean-arrival", type=float, default=10, help="Mean rider interarrival time.")
    parser.add_argument("--num-cars", type=int, default=3, help="Number of cars.")
    args = parser.parse_args()

    g = Graph()
    g.load_map_data(args.map)

    sim = Simulation(g, args.max_time, args.mean_arrival, args.num_cars)
    sim.run()
    sim.analyze_and_visualize()

if __name__ == "__main__":
    main()
