"""
Event-Driven Rideshare Simulation with Car Status Visualization
---------------------------------------------------------------
This program simulates a rideshare service in a 2D city grid using an event-driven engine.
It models rider requests and car assignments based on a sporadic, random selection method.
The simulation uses a priority queue to manage events, a simple graph for the road network,
and a quadtree for spatial indexing to find potential cars efficiently.

Car Status:
- Blue: Available
- Orange: Enroute Pickup
- Green: Enroute Dropoff

Author: Dr. Laurence
Date: 2025
"""

import heapq
import random
import math
import collections
from typing import List, Optional, Tuple, Dict
import matplotlib
# Use Agg backend to save figures without a display
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from itertools import count

# =========================================================
# Graph Class with Embedded Map
# =========================================================
class Graph:
    """
    Represents a road network with an embedded default map.

    This class handles the city's road structure, including nodes (intersections) and
    edges (roads) with associated weights (travel time). It provides methods for
    finding the nearest road node to a given point and calculating the shortest
    path between nodes using Dijkstra's algorithm.
    """
    def __init__(self):
        """
        Initializes the Graph with an adjacency list and node coordinates.
        The default map is loaded upon initialization.
        """
        self.adjacency_list: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
        self.node_coordinates: Dict[str, Tuple[float, float]] = {}
        self._load_default_map()

    def _load_default_map(self):
        """
        Loads a hard-coded default road network into the graph.

        The map is defined by a list of edges, where each edge specifies the
        start node, start coordinates, end node, end coordinates, and travel weight.
        This method populates the adjacency list and node coordinate dictionary.
        """
        edges = [
            ("A",0,0,"B",1,0,1),
            ("A",0,0,"C",0,1,1),
            ("B",1,0,"D",1,1,1),
            ("C",0,1,"D",1,1,1),
            ("B",1,0,"E",2,0,1),
            ("D",1,1,"F",2,1,1),
            ("E",2,0,"F",2,1,1),
            ("C",0,1,"G",0,2,1),
            ("D",1,1,"G",0,2,1),
            ("F",2,1,"H",3,1,1)
        ]
        for edge in edges:
            start_id, start_x, start_y, end_id, end_x, end_y, weight = edge
            self.node_coordinates[start_id] = (start_x, start_y)
            self.node_coordinates[end_id] = (end_x, end_y)
            self.adjacency_list[start_id].append((end_id, weight))
            self.adjacency_list[end_id].append((start_id, weight))
        print("[INFO] Embedded default map loaded successfully.")

    def find_nearest_vertex(self, point: Tuple[float, float]) -> str:
        """
        Finds the nearest graph node (intersection) to a given (x, y) point.

        Args:
            point (Tuple[float, float]): The (x, y) coordinates of a location.

        Returns:
            str: The ID of the nearest graph node.
        """
        x, y = point
        return min(
            self.node_coordinates,
            key=lambda n: (self.node_coordinates[n][0]-x)**2 + (self.node_coordinates[n][1]-y)**2
        )

    def dijkstra(self, start: str, end: str) -> float:
        """
        Computes the shortest path distance between two nodes using Dijkstra's algorithm.

        Args:
            start (str): The ID of the starting node.
            end (str): The ID of the destination node.

        Returns:
            float: The shortest travel time (distance) between the two nodes.
                   Returns infinity if no path exists.
        """
        dist = {node: math.inf for node in self.node_coordinates}
        dist[start] = 0
        pq: list[Tuple[float, str]] = [(0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == end:
                return d
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
    """
    Represents a rider's trip request.

    Each rider has a unique ID, a start location, a destination, and
    timestamps for key events like request, pickup, and dropoff times.
    """
    def __init__(self, rider_id: str, start: Tuple[float, float], dest: Tuple[float, float]):
        """
        Initializes a Rider object.

        Args:
            rider_id (str): A unique identifier for the rider.
            start (Tuple[float, float]): The rider's starting (x, y) coordinates.
            dest (Tuple[float, float]): The rider's destination (x, y) coordinates.
        """
        self.id = rider_id
        self.start = start
        self.dest = dest
        self.request_time: Optional[float] = None
        self.pickup_time: Optional[float] = None
        self.dropoff_time: Optional[float] = None

class Car:
    """
    Represents a car in the rideshare fleet.

    Each car has a unique ID, its current location, and its status, which can
    be "available," "enroute_pickup," or "enroute_dropoff."
    """
    def __init__(self, car_id: str, location: Tuple[float, float]):
        """
        Initializes a Car object.

        Args:
            car_id (str): A unique identifier for the car.
            location (Tuple[float, float]): The car's starting (x, y) coordinates.
        """
        self.id = car_id
        self.location = location
        self.status = "available"

# =========================================================
# Quadtree Class
# =========================================================
class Quadtree:
    """
    Spatial subdivision quadtree for fast nearest-neighbor search.

    The quadtree divides a 2D space into smaller quadrants to efficiently
    find objects (in this case, cars) located near a specific point.
    """
    MAX_POINTS = 4
    MAX_DEPTH = 10

    def __init__(self, bbox: Tuple[float,float,float,float]=(0,0,3,3), depth: int=0):
        """
        Initializes a Quadtree node with a bounding box and a depth.

        Args:
            bbox (Tuple[float,float,float,float]): The bounding box of the quadrant (xmin, ymin, xmax, ymax).
            depth (int): The current depth of the node in the tree.
        """
        self.bbox = bbox
        self.points: list[Tuple[str, Tuple[float,float]]] = []
        self.children: list['Quadtree'] = []
        self.depth = depth

    def insert(self, obj_id: str, point: Tuple[float,float]) -> bool:
        """
        Inserts an object's ID and coordinates into the quadtree.

        Args:
            obj_id (str): The unique ID of the object (e.g., a car ID).
            point (Tuple[float,float]): The (x, y) coordinates of the object.

        Returns:
            bool: True if insertion was successful, False otherwise.
        """
        if not self._in_bbox(point):
            return False
        if len(self.points) < Quadtree.MAX_POINTS or self.depth >= Quadtree.MAX_DEPTH:
            self.points.append((obj_id, point))
            return True
        if not self.children:
            self._subdivide()
        for child in self.children:
            if child.insert(obj_id, point):
                return True
        return False

    def remove(self, obj_id: str) -> bool:
        """
        Removes an object from the quadtree by its ID.

        Args:
            obj_id (str): The ID of the object to remove.

        Returns:
            bool: True if the object was found and removed, False otherwise.
        """
        for i, (pid, _) in enumerate(self.points):
            if pid == obj_id:
                self.points.pop(i)
                return True
        for child in self.children:
            if child.remove(obj_id):
                return True
        return False

    def find_k_nearest(self, point: Tuple[float,float], k: int=5) -> list[Tuple[str, Tuple[float,float]]]:
        """
        Finds the k nearest objects to a given point.

        Note: This is a simplified implementation. It retrieves all points from the
              tree and then sorts them by distance, which is not the most efficient
              way to find k-nearest neighbors in a quadtree.

        Args:
            point (Tuple[float,float]): The reference point.
            k (int): The number of nearest objects to find.

        Returns:
            list[Tuple[str, Tuple[float,float]]]: A list of tuples containing the
                                                  ID and coordinates of the k nearest objects.
        """
        result: list[Tuple[str, Tuple[float,float]]] = []
        self._search_knn(point, result)
        result.sort(key=lambda x: (x[1][0]-point[0])**2 + (x[1][1]-point[1])**2)
        return result[:k]

    def _in_bbox(self, point: Tuple[float,float]) -> bool:
        """Checks if a point is within the quadrant's bounding box."""
        x, y = point
        xmin, ymin, xmax, ymax = self.bbox
        return xmin <= x <= xmax and ymin <= y <= ymax

    def _subdivide(self):
        """Splits the current quadrant into four smaller children quadrants."""
        xmin, ymin, xmax, ymax = self.bbox
        mx, my = (xmin+xmax)/2, (ymin+ymax)/2
        self.children = [
            Quadtree((xmin, ymin, mx, my), self.depth+1),
            Quadtree((mx, ymin, xmax, my), self.depth+1),
            Quadtree((xmin, my, mx, ymax), self.depth+1),
            Quadtree((mx, my, xmax, ymax), self.depth+1)
        ]
        old_points = self.points
        self.points = []
        for pid, pt in old_points:
            for child in self.children:
                if child.insert(pid, pt):
                    break

    def _search_knn(self, point: Tuple[float,float], result: list):
        """Recursively searches for points in the quadtree."""
        result.extend(self.points)
        for child in self.children:
            child._search_knn(point, result)

# =========================================================
# Event & Simulation Classes
# =========================================================
class Event:
    """
    Represents a scheduled event in the simulation.

    Events are stored in a priority queue and processed in chronological order.
    """
    def __init__(self, time: float, event_type: str, data=None):
        """
        Initializes an Event object.

        Args:
            time (float): The timestamp at which the event should occur.
            event_type (str): A string identifying the type of event.
            data: Any data payload associated with the event (e.g., a car or rider object).
        """
        self.time = time
        self.type = event_type
        self.data = data

    def __lt__(self, other: 'Event') -> bool:
        """
        Defines the less-than comparison for Event objects based on time.
        This allows the priority queue to order events correctly.
        """
        return self.time < other.time

class Simulation:
    """
    The main event-driven rideshare simulation engine.

    This class manages the simulation's state, including the event queue, the fleet
    of cars, and the trip log. It processes events sequentially and handles car
    assignments and trip completion.
    """
    def __init__(self, graph: Graph, max_time: int=50, mean_arrival: float=10, num_cars: int=5):
        """
        Initializes the Simulation object.

        Args:
            graph (Graph): The road network graph for the simulation.
            max_time (int): The maximum simulation time.
            mean_arrival (float): The mean inter-arrival time for new rider requests (in seconds).
            num_cars (int): The number of cars in the fleet.
        """
        self.time: float = 0
        self.events: list = []
        self.counter = count()  # Used as a tie-breaker for the heapq
        self.graph = graph
        self.max_time = max_time
        self.mean_arrival = mean_arrival
        self.quadtree = Quadtree()
        self.cars: list[Car] = []
        self.cars_by_id: Dict[str, Car] = {}
        self.riders: Dict[str, Rider] = {}
        self.trip_log: list[dict] = []

        node_ids = list(graph.node_coordinates.keys())
        for i in range(num_cars):
            start_node = random.choice(node_ids)
            car = Car(f"CAR-{i}", graph.node_coordinates[start_node])
            self.cars.append(car)
            self.cars_by_id[car.id] = car
            self.quadtree.insert(car.id, car.location)

        self.add_event(0, "RIDER_REQUEST", None)

    def add_event(self, delay: float, event_type: str, data):
        """
        Adds a new event to the priority queue with a specified delay.

        Args:
            delay (float): The time delay from the current simulation time.
            event_type (str): The type of the event.
            data: The data payload for the event.
        """
        heapq.heappush(self.events, (self.time + delay, next(self.counter), Event(self.time + delay, event_type, data)))

    def generate_rider_request(self) -> Rider:
        """
        Creates and returns a new rider request with random start and destination points.
        """
        rider_id = f"R-{len(self.riders)+1}"
        start = (random.uniform(0,3), random.uniform(0,3))
        dest = (random.uniform(0,3), random.uniform(0,3))
        rider = Rider(rider_id, start, dest)
        rider.request_time = self.time
        self.riders[rider_id] = rider
        return rider

    def run(self):
        """
        Runs the simulation by processing events from the priority queue until
        the maximum simulation time is reached.
        """
        while self.events and self.time < self.max_time:
            event_time, _, event = heapq.heappop(self.events)
            self.time = event_time
            if event.type=="RIDER_REQUEST": self.handle_rider_request()
            elif event.type=="PICKUP_ARRIVAL": self.handle_pickup(event.data)
            elif event.type=="DROPOFF_ARRIVAL": self.handle_dropoff(event.data)
            elif event.type=="RIDER_REQUEST_RETRY": self.handle_rider_request_retry(event.data)

    # -------------------------
    # Event Handlers
    # -------------------------
    def handle_rider_request(self):
        """
        Handles a new rider request event by generating a request and attempting to
        assign a car.
        """
        rider = self.generate_rider_request()
        # Exponentially distribute the next rider arrival time
        delta = random.expovariate(120/self.mean_arrival)
        self.add_event(delta, "RIDER_REQUEST", None)
        self._assign_car_to_rider(rider)

    def handle_rider_request_retry(self, rider):
        """
        Handles a re-queued rider request by attempting to assign a car again.
        """
        self._assign_car_to_rider(rider)

    def _assign_car_to_rider(self, rider):
        """
        Assigns a car to a rider by sporadically selecting from all available cars.
        This modifies the original behavior to implement random assignment.

        Args:
            rider (Rider): The rider requesting a trip.
        """
        # Find all available cars
        available_cars = [car for car in self.cars if car.status == "available"]

        if available_cars:
            # Sporadically select one car from the available list
            best_car = random.choice(available_cars)
            
            # Now, proceed with the logic to assign the car
            rider_node = self.graph.find_nearest_vertex(rider.start)
            start_node = self.graph.find_nearest_vertex(best_car.location)
            travel_time = self.graph.dijkstra(start_node, rider_node)
            
            best_car.status = "enroute_pickup"
            self.quadtree.remove(best_car.id)
            self.add_event(travel_time, "PICKUP_ARRIVAL", (best_car, rider))
        else:
            print(f"[INFO] No car available for rider {rider.id}. Re-queuing request.")
            self.add_event(1, "RIDER_REQUEST_RETRY", rider)

    def handle_pickup(self, data):
        """
        Handles the pickup arrival event. The car's status is updated, and a
        new DROPOFF_ARRIVAL event is scheduled.
        """
        car,rider = data
        rider.pickup_time=self.time
        car.status="enroute_dropoff"
        car.location=rider.start
        start_node=self.graph.find_nearest_vertex(rider.start)
        dest_node=self.graph.find_nearest_vertex(rider.dest)
        travel_time=self.graph.dijkstra(start_node,dest_node)
        self.add_event(travel_time,"DROPOFF_ARRIVAL",(car,rider))

    def handle_dropoff(self,data):
        """
        Handles the dropoff arrival event. The car's status is reset to "available,"
        and the trip details are logged.
        """
        car,rider=data
        rider.dropoff_time=self.time
        car.status="available"
        car.location=rider.dest
        self.quadtree.insert(car.id,car.location)
        self.trip_log.append({
            "rider":rider.id,
            "wait_time":rider.pickup_time-rider.request_time,
            "trip_duration":rider.dropoff_time-rider.pickup_time
        })

    # -------------------------
    # Visualization
    # -------------------------
    def analyze_and_visualize(self, output_file="simulation_summary.png"):
        """
        Generates a summary plot showing final car locations and key simulation metrics.
        The plot is saved to a file.
        """
        if not self.trip_log:
            print("[INFO] No trips recorded")
            return
        waits=[t["wait_time"] for t in self.trip_log]
        durations=[t["trip_duration"] for t in self.trip_log]
        avg_wait=sum(waits)/len(waits)
        avg_dur=sum(durations)/len(durations)

        fig,(ax_map,ax_info)=plt.subplots(1,2,figsize=(12,6),gridspec_kw={"width_ratios":[2,1]})
        status_colors = {"available": "blue", "enroute_pickup": "orange", "enroute_dropoff": "green"}
        for car in self.cars:
            ax_map.scatter(car.location[0], car.location[1], color=status_colors[car.status], s=80)
        gx, gy = zip(*self.graph.node_coordinates.values())
        ax_map.scatter(gx, gy, c="gray", marker="x", label="Graph Nodes")
        ax_map.set_title("Final Car Locations")
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.grid(True)
        legend_elements = [
            plt.Line2D([0],[0], marker='o', color='w', label='Available', markerfacecolor=status_colors['available'], markersize=10),
            plt.Line2D([0],[0], marker='o', color='w', label='Enroute Pickup', markerfacecolor=status_colors['enroute_pickup'], markersize=10),
            plt.Line2D([0],[0], marker='o', color='w', label='Enroute Dropoff', markerfacecolor=status_colors['enroute_dropoff'], markersize=10),
            plt.Line2D([0],[0], marker='x', color='w', label='Graph Nodes', markeredgecolor='gray', markersize=10)
        ]
        ax_map.legend(handles=legend_elements, loc='upper right')

        ax_info.axis("off")
        ax_info.set_title("Simulation Results", fontsize=14)
        ax_info.text(0,0.9,f"Total Trips: {len(self.trip_log)}", fontsize=12)
        ax_info.text(0,0.8,f"Avg Wait: {avg_wait:.2f}", fontsize=12)
        ax_info.text(0,0.7,f"Avg Duration: {avg_dur:.2f}", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"[INFO] Saved simulation summary to {output_file}")

# =========================================================
# Run Simulation
# =========================================================
if __name__ == "__main__":
    print("[DEBUG] Script started")
    g = Graph()
    sim = Simulation(g, max_time=50, mean_arrival=10, num_cars=5)
    sim.run()
    sim.analyze_and_visualize()