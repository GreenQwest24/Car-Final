# rider.py

class Rider:
    def __init__(self, rider_id, start_location, destination):
        self.id = rider_id                          # Unique rider identifier (e.g., "RIDER_A")
        self.start_location = start_location        # Pickup coordinates (x, y)
        self.destination = destination              # Dropoff coordinates (x, y)
        self.status = "waiting"                     # Initial status

    def __str__(self):
        return (f"Rider {self.id} at {self.start_location} "
                f"{self.status} for ride to {self.destination}")


if __name__ == "__main__":
    r = Rider("RIDER_A", (1, 2), (20, 15))
    print(r)

from car import Car
from rider import Rider
