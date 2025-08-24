# 🚖 Event-Driven Rideshare Simulation with Car Status Visualization

This project simulates a simplified **rideshare system** using an **event-driven simulation engine**.  
It models rider requests, car assignments, and trip completions in a small **2D city grid** with an embedded default map.  

The simulation includes:
- A **graph-based road network** for shortest-path calculation (Dijkstra’s algorithm).
- A **quadtree spatial index** to manage and locate cars efficiently.
- A **priority queue event system** for scheduling and processing events.
- **Visualization of car statuses** after the simulation.

---

## ✨ Features
- Event-driven simulation loop.
- Riders request rides at **random intervals** (exponentially distributed).
- Cars are assigned to riders if available.
- Cars move through statuses:
  - **Blue** → Available  
  - **Orange** → Enroute Pickup  
  - **Green** → Enroute Dropoff  
- Simulation summary includes:
  - Total trips
  - Average wait time
  - Average trip duration
- Generates a **summary plot** (`simulation_summary.png`) showing car locations and statistics.

---

## 📂 Project Structure
