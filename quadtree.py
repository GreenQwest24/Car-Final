# quadtree.py

import math

class Rectangle:
    def __init__(self, x, y, width, height):
        self.x = x  # Center x
        self.y = y  # Center y
        self.width = width
        self.height = height

    def contains(self, point):
        px, py = point
        return (self.x - self.width / 2 <= px <= self.x + self.width / 2 and
                self.y - self.height / 2 <= py <= self.y + self.height / 2)

    def intersects(self, range_rect):
        return not (range_rect.x - range_rect.width / 2 > self.x + self.width / 2 or
                    range_rect.x + range_rect.width / 2 < self.x - self.width / 2 or
                    range_rect.y - range_rect.height / 2 > self.y + self.height / 2 or
                    range_rect.y + range_rect.height / 2 < self.y - self.height / 2)

    def distance_to_point(self, point):
        """Returns the shortest distance from the point to this rectangle."""
        px, py = point
        dx = max(abs(px - self.x) - self.width / 2, 0)
        dy = max(abs(py - self.y) - self.height / 2, 0)
        return math.hypot(dx, dy)


class QuadtreeNode:
    def __init__(self, boundary, capacity=4):
        self.boundary = boundary
        self.capacity = capacity
        self.points = []
        self.divided = False

        self.northwest = None
        self.northeast = None
        self.southwest = None
        self.southeast = None

    def subdivide(self):
        x, y = self.boundary.x, self.boundary.y
        w, h = self.boundary.width / 2, self.boundary.height / 2

        nw = Rectangle(x - w / 2, y - h / 2, w, h)
        ne = Rectangle(x + w / 2, y - h / 2, w, h)
        sw = Rectangle(x - w / 2, y + h / 2, w, h)
        se = Rectangle(x + w / 2, y + h / 2, w, h)

        self.northwest = QuadtreeNode(nw, self.capacity)
        self.northeast = QuadtreeNode(ne, self.capacity)
        self.southwest = QuadtreeNode(sw, self.capacity)
        self.southeast = QuadtreeNode(se, self.capacity)

        self.divided = True

    def insert(self, point):
        if not self.boundary.contains(point):
            return False

        if len(self.points) < self.capacity:
            self.points.append(point)
            return True

        if not self.divided:
            self.subdivide()
            for existing_point in self.points:
                self._insert_into_children(existing_point)
            self.points = []

        return self._insert_into_children(point)

    def _insert_into_children(self, point):
        return (
            self.northwest.insert(point) or
            self.northeast.insert(point) or
            self.southwest.insert(point) or
            self.southeast.insert(point)
        )

    def find_nearest(self, query_point, best_point=None, min_dist=float('inf')):
        if self.boundary.distance_to_point(query_point) > min_dist:
            return best_point, min_dist

        for point in self.points:
            dist = math.dist(point, query_point)
            if dist < min_dist:
                best_point, min_dist = point, dist

        if self.divided:
            children = [
                self.northwest,
                self.northeast,
                self.southwest,
                self.southeast,
            ]

            # Prioritize quadrant nearest to query point
            children.sort(key=lambda child: math.dist((child.boundary.x, child.boundary.y), query_point))

            for child in children:
                best_point, min_dist = child.find_nearest(query_point, best_point, min_dist)

        return best_point, min_dist


class Quadtree:
    def __init__(self, boundary, capacity=4):
        self.boundary = boundary
        self.root = QuadtreeNode(boundary, capacity)

    def insert(self, point):
        return self.root.insert(point)

    def find_nearest(self, query_point):
        return self.root.find_nearest(query_point)


# 🧪 Test the whole thing
if __name__ == "__main__":
    boundary = Rectangle(0, 0, 200, 200)
    qt = Quadtree(boundary, capacity=4)

    # Insert test points
    points = [
        (10, 20), (-30, 50), (70, 80), (-90, -90),
        (100, 0), (95, 10), (60, 10), (-60, -70)
    ]

    for pt in points:
        qt.insert(pt)
        print(f"Inserted: {pt}")

    # Test find_nearest
    query = (85, 10)
    nearest_point, distance = qt.find_nearest(query)
    print(f"\nNearest to {query} is {nearest_point} at distance {distance:.2f}")
