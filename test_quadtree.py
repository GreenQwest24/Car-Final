# test_quadtree.py

import random
import math
from quadtree import Quadtree, Rectangle


def generate_random_points(n, range_min=-500, range_max=500):
    return [(random.uniform(range_min, range_max), random.uniform(range_min, range_max)) for _ in range(n)]


def brute_force_nearest(points, query_point):
    best_point = None
    min_dist = float('inf')
    for point in points:
        dist = math.dist(point, query_point)
        if dist < min_dist:
            best_point = point
            min_dist = dist
    return best_point, min_dist


def main():
    BOUNDARY = Rectangle(0, 0, 1000, 1000)
    CAPACITY = 4
    NUM_POINTS = 5000

    # Initialize Quadtree
    qt = Quadtree(BOUNDARY, CAPACITY)

    # Generate and insert random points
    points = generate_random_points(NUM_POINTS)
    for pt in points:
        inserted = qt.insert(pt)
        if not inserted:
            print(f"❌ Failed to insert point {pt} — outside boundary!")

    # Pick a random query point
    query_point = (random.uniform(-500, 500), random.uniform(-500, 500))
    print(f"\n🔍 Query point: {query_point}")

    # Use Quadtree to find nearest neighbor
    qt_point, qt_dist = qt.find_nearest(query_point)

    # Use brute force to find nearest neighbor
    bf_point, bf_dist = brute_force_nearest(points, query_point)

    # Show results
    print(f"\n📦 Quadtree result: {qt_point} at distance {qt_dist:.6f}")
    print(f"🦾 Brute-force result: {bf_point} at distance {bf_dist:.6f}")

    # Validate
    assert math.isclose(qt_dist, bf_dist, abs_tol=1e-6), "ERROR: Quadtree result does not match brute-force result!"
    print("\n✅ Test passed: Quadtree matches brute-force result.")


if __name__ == "__main__":
    main()
