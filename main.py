import networkx as nx
from queue import PriorityQueue
from dataclasses import dataclass, field
import argparse
from typing import Any
import matplotlib.pyplot as plt
import datetime
import csv
import random
import math
import numpy as np



class Node:
    def __init__(self, name, pop, income, lat, long):
        # Name, latitude, longitude, population, weekly household income, default colour (1-5), empty list of neighbours
        self.name = name
        self.lat = lat
        self.long = long
        self.pop = pop
        self.income = income
        self.colour = 1
        self.neighbours = []

    def add_neighbour(self, neighbour):
        # Adds a neighbour (node object) after checking to see if it was there already
        if neighbour not in self.neighbours:
            self.neighbours.append(neighbour)


class Edge:
    def __init__(self, place1, place2, dist, time):
        # Two places (order unimportant), distance in km, time in mins, default colour (1-5)
        self.place1 = place1
        self.place2 = place2
        self.dist = dist
        self.time = time
        self.colour = 2


class Graph:
    def __init__(self):
        # List of edge objects and node objects
        self.edges = []
        self.nodes = []
        self.colour_dict = {
            0: "black",
            1: "blue",
            2: "red",
            3: "green",
            4: "yellow",
            5: "lightblue",
        }

    def load_data(self):
        # Reads the CSV files you are provided with and creates node/edges accordingly.
        # You should not need to change this function.

        # Read the nodes, create node objects and add them to the node list.
        with open("nodes.csv", "r", encoding="utf-8-sig") as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                name = row[0]
                pop = row[1]
                income = row[2]
                lat = float(row[3])
                long = float(row[4])
                node = Node(name, pop, income, lat, long)
                self.nodes.append(node)

        # Read the edges, create edge objects and add them to the edge list.
        with open("edges.csv", "r", encoding="utf-8-sig") as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                place1 = row[0]
                place2 = row[1]
                dist = int(row[2])
                time = int(row[3])
                edge = Edge(place1, place2, dist, time)
                self.edges.append(edge)

                for node in self.nodes:
                    if node.name == place1:
                        node.add_neighbour(place2)
                    if node.name == place2:
                        node.add_neighbour(place1)

            # Generate a dictionary of nodes with the node name as the key
            self.node_dict = {node.name: node for node in self.nodes}


    def get_dist(self, place1, place2):
        # Returns the distance between two place names (strings) if an edge exists,
        # otherwise returns -1.

        for edge in self.edges:
            if edge.place1 == place1 and edge.place2 == place2:
                return edge.dist
            if edge.place1 == place2 and edge.place2 == place1:
                return edge.dist
        return float("inf")

    def get_time(self, place1, place2):
        # Returns the time between two place names (strings) if an edge exists,
        # otherwise returns infinity.

        for edge in self.edges:
            if edge.place1 == place1 and edge.place2 == place2:
                return edge.time
            if edge.place1 == place2 and edge.place2 == place1:
                return edge.time
        return float("inf")


    def display(self, filename="map.png"):
        # Displays the object on screen and also saves it to a PNG named in the argument.

        edge_labels = {}
        edge_colours = []
        G = nx.Graph()
        node_colour_list = []
        for node in self.nodes:
            G.add_node(node.name, pos=(node.long, node.lat))
            node_colour_list.append(self.colour_dict[node.colour])
        for edge in self.edges:
            G.add_edge(edge.place1, edge.place2)
            edge_labels[(edge.place1, edge.place2)] = edge.dist
            edge_colours.append(self.colour_dict[edge.colour])
        node_positions = nx.get_node_attributes(G, "pos")

        plt.figure(figsize=(10, 8))
        nx.draw(
            G,
            node_positions,
            with_labels=True,
            node_size=50,
            node_color=node_colour_list,
            font_size=8,
            font_color="black",
            font_weight="bold",
            edge_color=edge_colours,
        )
        nx.draw_networkx_edge_labels(G, node_positions, edge_labels=edge_labels)
        plt.title("")
        plt.savefig(filename)
        plt.show()

    def haversine(self, lat1, lon1, lat2, lon2):
        # Returns the distance in km between two places with given latitudes and longitudes.

        # Radius of the Earth in kilometers
        R = 6371.0

        # Convert latitude and longitude from degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Differences in coordinates
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        # Haversine formula
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        # Calculate the distance
        distance = R * c

        return distance

    # This is where you will write your algorithms. You don't have to use
    # these names/parameters but they will probably steer you in the right
    # direction.
    """
        Create a list of nodes within a given radius of a starting node using a breadth-first search algorithm to traverse the graph.
        Distance between nodes is calculated using the haversine formula to account for the curvature of the Earth and Summed from the starting point out.

        Signature: graph x node x int -> list
    """
    def bfs(self, start, radius: int) -> list:
        # Create a dictionary to store the distance from the starting node to each node
        visited = {start: 0.0}
        # Create a queue to store nodes to visit with the starting node
        queue = [start]

        # While there are nodes in the queue
        while queue:
            # Get the next node to visit
            node = queue.pop(0)

            # For each neighbour of the current node
            for neighbour in node.neighbours:
                neighbour = self.node_dict[neighbour]
                # Calculate the distance from the starting node to the neighbour
                distance = self.haversine(  # Add the previous distance to the distance to the neighbour to generate the total distance
                    start.lat, start.long, neighbour.lat, neighbour.long
                )

                # If the distance is outside the radius, break the loop
                if distance > radius:
                    continue

                # If the neighbour has not been visited or the new distance is less than the previous distance
                if neighbour not in visited or distance < visited[neighbour]:
                    # Update the distance to the neighbour
                    visited[neighbour] = distance
                    # Add the neighbour to the queue
                    queue.append(neighbour)

        # Convert the dictionary of visited nodes to a list of node names
        nodes = list(visited.keys())

        return nodes  # Return the list of nodes within the radius

@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any = field(compare=False)

class AStar:
    class PathNode:
        def __init__(self, node: Node) -> None:
            self.node = node
            self.g = float("inf")
            self.h = float("inf")
            self.f = float("inf")

    def __init__(self, graph: Graph, start: Node, target: Node) -> None:
        self.graph = graph
        self.start = self.PathNode(start)
        self.target = target

        self.start.g = 0
        self.start.h = self.heuristic(self.start.node)
        self.start.f = self.start.g + self.start.h

        self.open = PriorityQueue()
        self.closed = set()

        self.parent = {}

        self.path = []
        self.path_found = False

    def heuristic(self, node: Node) -> float:
        return self.graph.haversine(node.lat, node.long, self.target.lat, self.target.long)

    def stats(self, path: list) -> tuple:
        total_dist = 0
        total_time = 0
        for i,j in zip(path, path[1:]): # For each pair of nodes in the path
            total_dist += self.graph.get_dist(i.name, j.name)
            total_time += self.graph.get_time(i.name, j.name)
        return total_time, total_dist

    def find_path(self) -> list:
        self.open.put(PrioritizedItem(self.start.f, self.start)) # Add the starting node to the open list

        while not self.open.empty(): # While there are nodes in the open list
            current = self.open.get().item # Dequeue the node with the lowest f value

            if current.node == self.target:
                self.path_found = True
                break

            self.closed.add(current.node) # Add the current node to the closed list

            for neighbour in current.node.neighbours:
                # Convert the neighbour name to a node object, Could probably be fixed by refactoring the node.neighbours method to return nodes
                neighbour = self.graph.node_dict[neighbour] 

                if neighbour in self.closed:
                    continue

                neighbour_node = self.PathNode(neighbour)

                g = current.g + self.graph.get_dist(current.node.name, neighbour.name)
                h = self.heuristic(neighbour)
                f = g + h

                if f < neighbour_node.f:
                    neighbour_node.g = g
                    neighbour_node.h = h
                    neighbour_node.f = f

                    self.parent[neighbour] = current.node

                    self.open.put(PrioritizedItem(neighbour_node.f, neighbour_node)) # Add the neighbour to the open list

        if self.path_found:
            current = self.target
            while current != self.start.node:
                self.path.insert(0, current) # Insert the current node at the start of the path
                current = self.parent[current]

            self.path.insert(0, self.start.node) # Insert the starting node at the start of the path

            return self.path

        return [None]
        
class SimulatedAnnealing:
    def __init__(self, graph, start, radius, initial_temp=100.0, stop_temp=1e-8, max_iterations=100000, cooling_rate=0.995):
        self.graph = graph
        self.start = start
        self.nodes = self.graph.bfs(self.start, radius) # Get the nodes within the radius of the starting node
        
        self.T = initial_temp
        self.stop_T = stop_temp
        self.cooling_rate = cooling_rate

        self.max = max_iterations

    def acceptance_probability(self, best: float, candidate: float) ->float:
        return np.exp((best - candidate) / self.T)

    def stats(self, path: list) -> tuple:
        total_dist = 0
        total_time = 0
        for i,j in zip(path, path[1:]): # For each pair of nodes in the path
            total_dist += self.graph.get_dist(i.name, j.name)
            total_time += self.graph.get_time(i.name, j.name)
        return total_time, total_dist

    """
    Convert a list of potentially disconnected nodes to a path with virtual edges between nodes that are not connected.
    """
    def verify_path(self, path: list) -> list:
        verified_path = [path[0]]
        for i,j in zip(path, path[1:]): # For each pair of nodes in the path
            a_star = AStar(self.graph, i, j)
            verified_path.extend(a_star.find_path()[1:]) # Create Vertual edges between nodes in the path that are not connected
        return verified_path

    def two_opt_swap(self, path: list, i: int, k: int) -> list:
        new_path = path[:i] # New path with the first i nodes
        new_path.extend(reversed(path[i:k+1])) # Add the nodes from i to k in reverse order
        new_path.extend(path[k+1:]) # Add the remaining nodes
        return new_path

    def anneal(self) -> tuple:
        best_path_nodes = self.nodes
        best_path = self.verify_path(best_path_nodes)
        best_stats = self.stats(best_path)

        current_path_nodes = best_path_nodes
        current_path = best_path
        current_stats = best_stats

        iter = 0
        while self.T >= self.stop_T and iter <= self.max: # Iterate until the temperature is below the stop temperature or the maximum iterations is reached
            i = random.randint(1, len(current_path) - 2) # Random ints for i and j
            k = random.randint(i+1, len(current_path) - 1)

            new_path_nodes = self.two_opt_swap(current_path_nodes, i, k)
            new_path = self.verify_path(new_path_nodes)
            new_stats = self.stats(new_path)

            # Could be moved into its own method with some OOP vooodoo
            if new_stats[0] < current_stats[0]: # If new path has a lower time than the current path
                current_path_nodes = new_path_nodes
                current_path = new_path
                current_stats = new_stats

                if new_stats[0] < best_stats[0]: # If new path has a lower time than the best path
                    best_path_nodes = new_path_nodes
                    best_path = new_path
                    best_stats = new_stats
            else: # Otherwise use the acceptance probability, See https://en.wikipedia.org/wiki/Simulated_annealing for more information
                if random.random() < self.acceptance_probability(current_stats[0], new_stats[0]):
                    current_path_nodes = new_path_nodes
                    current_path = new_path
                    current_stats = new_stats

            self.T *= self.cooling_rate
            iter += 1

        return best_path, best_stats


# These commands run the code.

# Create a new graph object called 'original'
original = Graph()

# Load data into that object.
original.load_data()

# Display the object, also saving to map.png
# original.display("map.png")

# You will add your own functions under the Graph object and call them in this way:
# original.findpath("Alexandra")
def print_path(graph, start, end):
    print("A* Pathfinding (SSSP)")
    print("======================")

    print(f"Start: {start}")
    print(f"Target: {end} \n")

    a_star = AStar(graph, graph.node_dict[start], graph.node_dict[end])

    path = a_star.find_path()

    if path is None:
        print("No path found")
        return None

    # Path Distance and Time
    stats = a_star.stats(path)
    hours, mins = divmod(stats[0], 60)

    # Convert the path to a list of node names
    print(f"Path: {" -> ".join([node.name for node in path])}")

    print(f"Total Time: {hours}hr {mins} min")
    print(f"Total Distance: {stats[1]}km")
    print(f"Towns Visited: {len(path)}")

    return path

def print_all(graph, start, radius, temp, stop_temp, timeout, cooling_rate):
    print("Simulated Annealing (TSP)")
    print("=========================")
    print(f"Start: {start}")
    print(f"Radius: {radius}")
    print(f"Initial Temperature: {temp}")
    print(f"Stop Temperature: {stop_temp}")
    print(f"Timeout: {timeout}")
    print(f"Cooling Rate: {cooling_rate}")

    print()

    sa = SimulatedAnnealing(graph, graph.node_dict[start], radius, temp, stop_temp, timeout, cooling_rate)

    best_path, best_stats = sa.anneal()

    if best_path is None:
        print("No path found")
        return None

    # Convert the best path to a list of node names
    hours, mins = divmod(best_stats[0], 60)

    print(f"Best Path: {" -> ".join([node.name for node in best_path])}")
    print(f"Total Time: {hours}hr {mins} min")
    print(f"Total Distance: {best_stats[1]}km")
    print(f"Towns Visited: {len(best_path)}")

    return best_path, best_stats

# CLI Arguments, Start Node, Radius, T (opt), Stop Temp (opt), Timeout (opt), Cooling Rate (opt)
boolopts = {
        "tsp": {"short": "-tsp", "long": "--tsp", "help": "Travelling Salesman Problem"},
        "sssp": {"short": "-sssp", "long": "--sssp", "help": "Single Source Shortest Path"},
        "display": {"short": "-d", "long": "--display", "help": "Display map"},
        }
opts = {
        "node": {"short": "-n", "long": "--node", "help": "Start node"},
        "target": {"short": "-t", "long": "--target", "help": "Target node (Pathfinding)"},
        "radius": {"short": "-r", "long": "--radius", "help": "Radius"},
        "temp": {"short": "-tp", "long": "--temp", "help": "Initial temperature (default: 100.0)"},
        "stop": {"short": "-s", "long": "--stop", "help": "Stop temperature (default: 1e-8)"},
        "timeout": {"short": "-o", "long": "--timeout", "help": "Timeout iterations (default: 100000)"},
        "cooling": {"short": "-c", "long": "--cooling", "help": "Cooling rate (default: 0.995)"},
        }

parser = argparse.ArgumentParser(description="Simulated Annealing for Travelling Salesman Problem")
for opt, val in opts.items():
    parser.add_argument(val["short"], val["long"], help=val["help"])
for opt, val in boolopts.items():
    parser.add_argument(val["short"], val["long"], help=val["help"], action=argparse.BooleanOptionalAction)
args = parser.parse_args()

start = args.node

print(f"Algotithmics SAT 2: Advanced Algorithms")
print("=======================================\n")


if args.tsp:
    radius = int(args.radius)
    T = 100.0 if args.temp is None else float(args.temp)
    stop_temp = 1e-8 if args.stop is None else float(args.stop)
    timeout = 100000 if args.timeout is None else int(args.timeout)
    cooling_rate = 0.995 if args.cooling is None else float(args.cooling)

    path = print_all(original, start, radius, T, stop_temp, timeout, cooling_rate)

if args.sssp:
    print()
    target = args.target

    path = print_path(original, start, target)

if args.display:
    original.display("map.png")



