
from car import Car
from rider import Rider


class Simulation:
    def __init__(self):
        # Dictionary of car_id: Car object
        self.cars = {}

        # Dictionary of rider_id: Rider object
        self.riders = {}

    def add_car(self, car):
        """Add a Car object to the simulation."""
        self.cars[car.id] = car

    def add_rider(self, rider):
        """Add a Rider object to the simulation."""
        self.riders[rider.id] = rider

    def __str__(self):
        return (f"Simulation State:\n"
                f"Cars:\n" + "\n".join(str(c) for c in self.cars.values()) +
                "\nRiders:\n" + "\n".join(str(r) for r in self.riders.values()))

# Basic test run
if __name__ == "__main__":
    # Create a car and rider
    car1 = Car("CAR001", (5, 5))
    rider1 = Rider("RIDER_A", (1, 2), (20, 15))

    # Create simulation and add them
    sim = Simulation()
    sim.add_car(car1)
    sim.add_rider(rider1)

    # Display the simulation state
    print(sim)
