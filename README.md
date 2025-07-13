This project defines a simple `Car` class in Python, representing an individual vehicle in a ride-sharing or fleet management simulation. Each car has a unique ID, current location, status, and an optional destination.

## 📦 Project Structure

```
car_project/
├── car.py
└── README.md
```

- `car.py`: Contains the `Car` class definition and a simple test block.
- `README.md`: This documentation file you are reading.

## 🧠 Features

- **Initialization** with car ID and location.
- Maintains a **status** (`available`, `occupied`, etc.).
- Stores an optional **destination**.
- Custom `__str__` method for clean, human-readable output.

## 🛠️ Getting Started

### Requirements

- Python 3.6 or higher

### Running the Code

1. Clone this repository:

```bash
git clone https://github.com/your-username/car_project.git
cd car_project
```

2. Run the script:

```bash
python car.py
```

3. Output should look like this:

```
Car CAR001 at (10, 5) - Status: available
```

## 🧪 Example Usage

```python
from car import Car

my_car = Car("CAR007", (4, 2))
print(my_car)
```

## 🚧 Future Enhancements (To-Do)

- Add methods for:
  - `assign_destination()`
  - `update_location()`
  - `set_status()`
- Add a `Rider` class and connect cars to active rides
- Simulate a basic ride-sharing environment

## 🤖 Author

**Dr. Laurence (a.k.a the Chief Droidsmith of AI Engineering)**

---

May your classes be tight, and your objects well-behaved.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
