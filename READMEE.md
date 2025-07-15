# 🚗 Car Ride Simulation System

This project is a modular Python simulation of a ride-sharing service like Uber/Ola. It uses a weighted graph loaded from a CSV file to represent a city map. Cars and riders interact in the simulation, where riders request pickups and cars are assigned based on availability and graph-based routing.

---

## 📁 Project Structure

```
├── graph.py           # Graph class (weighted, directed)
├── simulation.py      # Main Simulation class
├── simulation2.py     # Extended Simulation with advanced logic (e.g. Dijkstra)
├── car.py             # Car class (mobility & status)
├── rider.py           # Rider class (identity & journey info)
├── map.csv            # Weighted map of city in CSV format
└── README.md          # This file
```

---

## 🔧 How It Works

### 1. `graph.py`
Implements a directed, weighted graph using an adjacency list.

**Key Methods:**
- `add_edge(start_node, end_node, weight)`: Adds an edge with weight.
- `load_from_file(file_path)`: Reads edges from `map.csv`.

---

### 2. `map.csv`
A CSV file representing the city map. Each row defines a connection:

**Format:**
```
start_node,end_node,weight
A,B,5
B,C,3
C,D,7
```

Ensure this file is in the same directory as your Python scripts.

---

### 3. `simulation.py`
Base simulation class.

**Responsibilities:**
- Stores registered cars and riders.
- Assigns cars to riders.
- Tracks simulation state.

---

### 4. `simulation2.py`
An enhanced simulation layer. Includes:

- **Pathfinding**: Dijkstra’s algorithm or similar shortest path logic.
- **ETA calculations** based on graph weights.
- May override/extend base Simulation methods.

---

### 5. `car.py`
Each Car object has:
- `id`: Unique identifier (e.g., `CAR_1`)
- `location`: Current node on the graph
- `available`: Boolean flag for ride availability

---

### 6. `rider.py`
Each Rider object includes:
- `id`: Unique string ID (e.g., `RIDER_X`)
- `start_location`: Graph node (tuple or string)
- `end_location`: Destination node

---

## ▶️ Running the Simulation

1. Ensure Python 3 is installed.
2. Place all `.py` files and `map.csv` in the same directory.
3. Example usage:
```python
from graph import Graph
from simulation2 import Simulation
from car import Car
from rider import Rider

g = Graph()
g.load_from_file("map.csv")

sim = Simulation(graph=g)
sim.add_car(Car("CAR_1", "A"))
sim.add_rider(Rider("RIDER_1", "A", "D"))

sim.assign_rides()
```

---

## ✅ Features (Current)
- Graph-based routing and distance calculation.
- Ride assignment logic based on proximity.
- Modular car and rider classes.
- CSV-based map loading for flexibility.

---

## 🚀 Future Features
- Real-time updates to car positions.
- Support for traffic (dynamic edge weights).
- GUI or web dashboard.
- AI-powered ride scheduling using reinforcement learning.

---

## 🧠 Requirements

- Python 3.x
- No external libraries (unless `heapq` or `csv` for standard Dijkstra logic)

---

## 🛠️ Troubleshooting

- **FileNotFoundError**: Make sure `map.csv` is in the correct path.
- **ImportError**: Ensure all module filenames match their imports and no circular imports exist.

---

## 📜 License

MIT License. Use it, modify it, launch your own Uber-for-aliens with it. Just give credit.

---

## ✨ Author

**Dr. Laurence**  
Quantum-tech enthusiast. Star Wars lorekeeper. AI whisperer.  
"May your simulations always converge, and your code never segfault."
