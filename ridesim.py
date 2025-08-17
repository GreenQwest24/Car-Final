import argparse
import heapq
import random
import math
import collections
import matplotlib.pyplot as plt
from collections import deque

# -------------------------
# Graph + Dijkstra
# -------------------------
class Graph:
    def __init__(self):
        self.adjacency_list = collections.defaultdict(list)
        self.node_coordinates = {}

    def add_edge(self, u, v, w):
        self.adjacency_list[u].append((v, w))
        self.adjacency_list[v].append((u, w))

    def add_node(self, node, x, y):
        self.node_coordinates[node] = (x, y)

    def dijkstra(self, start, goal):
        """Return shortest path distance using Dijkstra."""
        pq = [(0, start)]
        dist = {start: 0}
        visited = set()

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            if u == goal:
                return d
            for v, w in self.adjacency_list[u]:
                if v not in dist or d + w < dist[v]:
                    dist[v] = d + w
                    heapq.heappush(pq, (dist[v], v))
        return math.inf

# -------------------------
# Quadtree (simplified for cars)
# -------------------------
class Quadtree:
    def __init__(self, points):
        self.points = points  # [(id, (x, y))]

    def nearest(self, query_point):
        """Return the closest car (id, distance)."""
        qx, qy = query_point
        best = None
        best_dist = math.inf
        for cid, (x, y) in self.points:
            dist = math.sqrt((qx - x)**2 + (qy - y)**2)
            if dist < best_dist:
                best = (cid, dist)
                best_dist = dist
        return best

# -------------------------
# Rider + Car Entities
# -------------------------
class RiderRequest:
    def __init__(self, rider_id, pickup, dropoff, request_time):
        self.rider_id = rider_id
        self.pickup = pickup
        self.dropoff = dropoff
        self.request_time = request_time

class Car:
    def __init__(self, car_id, location):
        self.car_id = car_id
        self.location = location
        self.available = True

# -------------------------
# Simulation Engine
# -------------------------
class Simulator:
    def __init__(self, graph, args):
        self.graph = graph
        self.max_time = args.max_time
        self.num_cars = args.num_cars
        self.rider_queue = deque()
        self.cars = [Car(i, random.choice(list(graph.node_coordinates.keys())))
                     for i in range(self.num_cars)]
        self.completed_trips = 0
        self.total_wait_time = 0
        self.trip_durations = []

    def generate_riders(self, t):
        if random.random() < 0.05:  # 5% chance per tick
            nodes = list(self.graph.node_coordinates.keys())
            pickup = random.choice(nodes)
            dropoff = random.choice(nodes)
            rider = RiderRequest(len(self.rider_queue), pickup, dropoff, t)
            self.rider_queue.append(rider)
            print(f"[t={t}] New rider {rider.rider_id} requests trip {pickup} → {dropoff}")

    def step(self, t):
        # New riders arrive
        self.generate_riders(t)

        # Assign riders if available
        if self.rider_queue:
            rider = self.rider_queue.popleft()
            # Use Quadtree to find nearest available car
            car_positions = [(c.car_id, self.graph.node_coordinates[c.location])
                             for c in self.cars if c.available]
            if not car_positions:
                return
            qt = Quadtree(car_positions)
            nearest_car_id, _ = qt.nearest(self.graph.node_coordinates[rider.pickup])
            car = self.cars[nearest_car_id]

            # Compute wait + trip time using Dijkstra
            wait_time = self.graph.dijkstra(car.location, rider.pickup)
            trip_time = self.graph.dijkstra(rider.pickup, rider.dropoff)

            self.total_wait_time += wait_time
            self.trip_durations.append(trip_time)
            self.completed_trips += 1

            print(f"[t={t}] Car {car.car_id} → Rider {rider.rider_id} "
                  f"(wait={wait_time}, trip={trip_time})")

            # Move car to new location
            car.location = rider.dropoff

    def run(self):
        for t in range(self.max_time):
            self.step(t)
        self.visualize()

    def visualize(self):
        plt.figure(figsize=(7, 6))
        # Plot nodes
        for u, coords in self.graph.node_coordinates.items():
            plt.scatter(coords[0], coords[1], color="blue")
            plt.text(coords[0] + 0.1, coords[1] + 0.1, str(u), fontsize=8)

        # Plot edges
        for u, neighbors in self.graph.adjacency_list.items():
            x1, y1 = self.graph.node_coordinates[u]
            for v, _ in neighbors:
                x2, y2 = self.graph.node_coordinates[v]
                plt.plot([x1, x2], [y1, y2], color="gray", alpha=0.5)

        plt.title("Simulation Summary")
        plt.xlabel("X")
        plt.ylabel("Y")

        metrics_text = (
            f"Trips: {self.completed_trips}\n"
            f"Avg Wait: {self.total_wait_time / max(1, self.completed_trips):.2f}\n"
            f"Avg Trip: {sum(self.trip_durations) / max(1, len(self.trip_durations)):.2f}"
        )
        plt.figtext(0.99, 0.5, metrics_text, ha="right", va="center", fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.7))
        plt.savefig("simulation_summary.png")
        print("\n✅ Simulation complete. Results saved to simulation_summary.png")

# -------------------------
# Build a Demo Graph
# -------------------------
def build_demo_graph():
    g = Graph()
    # Simple square grid
    g.add_node(0, 0, 0)
    g.add_node(1, 2, 0)
    g.add_node(2, 0, 2)
    g.add_node(3, 2, 2)
    g.add_edge(0, 1, 2)
    g.add_edge(0, 2, 2)
    g.add_edge(1, 3, 2)
    g.add_edge(2, 3, 2)
    return g

# -------------------------
# Main Entry Point
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ride-sharing Simulator")
    parser.add_argument('--max-time', type=int, default=50,
                        help='Maximum simulation time in discrete steps')
    parser.add_argument('--num-cars', type=int, default=5,
                        help='Number of cars in the fleet')
    args = parser.parse_args()

    graph = build_demo_graph()
    sim = Simulator(graph, args)
    sim.run()

