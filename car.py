
# car.py

class Car:
    def __init__(self, car_id, location):
        self.id = car_id                        # Unique car identifier (e.g., "CAR001")
        self.location = location                # Tuple (x, y) representing current coordinates
        self.status = "available"               # Initial status
        self.destination = None                 # Destination, if any

    def __str__(self):
        return f"Car {self.id} at {self.location} - Status: {self.status}"

if __name__ == "__main__":
    c = Car("CAR001", (10, 5))
    print(c)
