import heapq
from pprint import pprint

TRAVEL_SPEED_FACTOR = 1  # time units per grid unit


class Car:
    """Represents a car in the simulation."""
    def __init__(self, car_id, location):
        self.id = car_id
        self.location = location
        self.status = "available"
        self.assigned_rider = None
 
    def __repr__(self):
        return (f"Car(id={self.id}, loc={self.location}, status={self.status}, "
                f"assigned_rider={self.assigned_rider.id if self.assigned_rider else None})")


class Rider:
    """Represents a rider in the simulation."""
    def __init__(self, rider_id, start_location, destination):
        self.id = rider_id
        self.start_location = start_location
        self.destination = destination
        self.status = "waiting"

    def __repr__(self):
        return (f"Rider(id={self.id}, start={self.start_location}, "
                f"dest={self.destination}, status={self.status})")


class Simulation:
    """Discrete-Event Simulation for ride-hailing with verbose logging."""
    def __init__(self, verbose=False):
        self.current_time = 0
        self.event_queue = []
        self.cars = {}
        self.riders = {}
        self.verbose = verbose

    def schedule_event(self, timestamp, event_type, data):
        """Push event to priority queue."""
        heapq.heappush(self.event_queue, (timestamp, event_type, data))

    def calculate_travel_time(self, start, end):
        """Manhattan distance × TRAVEL_SPEED_FACTOR."""
        distance = abs(start[0] - end[0]) + abs(start[1] - end[1])
        return distance * TRAVEL_SPEED_FACTOR

    def find_closest_car_brute_force(self, rider_location):
        """Return closest available car."""
        available_cars = [c for c in self.cars.values() if c.status == "available"]
        if not available_cars:
            return None
        return min(available_cars, key=lambda c: abs(c.location[0] - rider_location[0]) +
                                                abs(c.location[1] - rider_location[1]))

    def handle_rider_request(self, rider):
        """Assign nearest available car and schedule pickup arrival."""
        car = self.find_closest_car_brute_force(rider.start_location)
        if car:
            car.assigned_rider = rider
            car.status = "en_route_to_pickup"
            pickup_duration = self.calculate_travel_time(car.location, rider.start_location)
            self.schedule_event(self.current_time + pickup_duration, "ARRIVAL", car)
            print(f"TIME {self.current_time}: CAR {car.id} dispatched to RIDER {rider.id}")
        else:
            print(f"TIME {self.current_time}: No available car for RIDER {rider.id}")

    def handle_arrival(self, car):
        """Handle pickup or drop-off arrival."""
        if car.status == "en_route_to_pickup":
            rider = car.assigned_rider
            car.location = rider.start_location
            car.status = "en_route_to_destination"
            rider.status = "in_car"
            dropoff_duration = self.calculate_travel_time(rider.start_location, rider.destination)
            self.schedule_event(self.current_time + dropoff_duration, "ARRIVAL", car)
            print(f"TIME {self.current_time}: CAR {car.id} picked up RIDER {rider.id}")

        elif car.status == "en_route_to_destination":
            rider = car.assigned_rider
            car.location = rider.destination
            car.status = "available"
            rider.status = "completed"
            car.assigned_rider = None
            print(f"TIME {self.current_time}: CAR {car.id} dropped off RIDER {rider.id}")

    def log_state(self):
        """Verbose logging of current state and queue."""
        if self.verbose:
            print("\n--- STATE SNAPSHOT ---")
            print(f"Current Time: {self.current_time}")
            print("\nCars:")
            pprint(self.cars)
            print("\nRiders:")
            pprint(self.riders)
            print("\nEvent Queue:")
            pprint(self.event_queue)
            print("--- END SNAPSHOT ---\n")

    def run(self):
        """Main simulation loop."""
        while self.event_queue:
            timestamp, event_type, data = heapq.heappop(self.event_queue)
            self.current_time = timestamp

            if event_type == "REQUEST":
                self.handle_rider_request(data)
            elif event_type == "ARRIVAL":
                self.handle_arrival(data)

            self.log_state()


# -------------------------------
# Example usage with verbose mode
# -------------------------------
if __name__ == "__main__":
    sim = Simulation(verbose=True)

    # Add cars
    sim.cars[1] = Car(1, (0, 0))
    sim.cars[2] = Car(2, (5, 5))

    # Add riders
    sim.riders[1] = Rider(1, (2, 2), (8, 8))
    sim.riders[2] = Rider(2, (6, 6), (0, 0))

    # Schedule rider requests
    sim.schedule_event(0, "REQUEST", sim.riders[1])
    sim.schedule_event(3, "REQUEST", sim.riders[2])

    # Run simulation
    sim.run()
