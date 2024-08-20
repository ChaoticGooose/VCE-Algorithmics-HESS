import networkx as nx
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import math
from itertools import permutations

class SimulatedAnnealing:
    def __init__(self, graph, start, radius, T=100.0, stop_temp=-1, timeout=100000, cooling_rate=0.995):
        self.graph = graph
        self.start = start
        self.radius = radius

        # Constatnts for the simulated annealing algorithm
        self.T = T
        self.STOP_T = 1e-8 if stop_temp == -1 else stop_temp
        self.cooling_rate = cooling_rate

        self.nodes = set(self.graph.bfs(self.start, self.radius))
        self.all_nodes = set(self.graph.nodes)
        self.timeout_iter = timeout

        self.iter = 0

        self.best_stats = (float("inf"), float("inf"))
        self.best_path = None

        self.path_list = []

    """
    Get distance and time of path
    list -> tuple
    """
    def path_stats(self, path):
        total_dist = 0
        total_time = 0

        for i in range(len(path) - 1):
            total_dist += self.graph.get_dist(path[i].name, path[i + 1].name)
            total_time += self.graph.get_time(path[i].name, path[i + 1].name)

        return total_dist, total_time

    def initial_solution(self):
        """
        Greedy algorithm to find the initial solution to the Travelling Salesman Problem.
        """

        solution = [self.start]

        unvisited = set(self.nodes)
        unvisited.remove(self.start)

        while unvisited:
            current = solution[-1]
            current_neighbours = current.neighbours

            # convert node names to node objects
            current_neighbours = [self.graph.node_dict[node] for node in current_neighbours]

            if unvisited.intersection(current_neighbours):
                next_node = min(
                        unvisited.intersection(current_neighbours),
                        key=lambda x: self.graph.get_time(current.name, x.name),
                        )
                unvisited.remove(next_node)
            else:
                next_node = min(
                        self.all_nodes.difference(unvisited),
                        key=lambda x: self.graph.get_time(current.name, x.name),
                        )
                self.all_nodes.remove(next_node)

            solution.append(next_node)

        path_stats = self.path_stats(solution)
        if path_stats[1] < self.best_stats[1]:
            self.best_stats = path_stats
            self.best_path = solution

        self.path_list.append(path_stats)
        return solution, path_stats

    def prob_accept(self, candidate):
        """
        Probability of accepting a candidate solution based on the current temperature and the difference in cost.
        See: https://en.wikipedia.org/wiki/Simulated_annealing#Acceptance_probabilities
        """
        return math.exp(-abs(candidate - self.best_stats[1]) / self.T)

    def accept(self, candidate, stats):
        """
        Accept the candidate solution if it is better than the current solution.
        If the candidate solution is worse, accept it with a probability based on the current temperature.
        """

        if stats[1] < self.current_stats[1]:
            self.current_solution = candidate
            self.current_stats = stats
            if stats[1] < self.best_stats[1]:
                self.best_stats = stats
                self.best_path = candidate
        else:
            if random.random() < self.prob_accept(stats[1]):
                self.current_solution = candidate
                self.current_stats = stats


    def anneal(self):
        self.current_path, self.current_stats = self.initial_solution()

        while self.T >= self.STOP_T and self.iter < self.timeout_iter:
            path = list(self.current_path)
            l = random.randint(2, len(path) - 1)
            i = random.randint(0, len(path) - l)

            path[i : (i + l)] = reversed(path[i : (i + l)])

            self.accept(path, self.path_stats(path))

            self.T *= self.cooling_rate
            self.iter += 1

        print(f"self.iter: {self.iter}")
        print(f"self.T: {self.T}")

        return self.best_path, self.best_stats

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




# These commands run the code.

# Create a new graph object called 'original'
original = Graph()

# Load data into that object.
original.load_data()

# Display the object, also saving to map.png
# original.display("map.png")

# You will add your own functions under the Graph object and call them in this way:
# original.findpath("Alexandra")
def print_all(graph, start, radius, temp, stop_temp, timeout, cooling_rate):
    print(f"Start: {start}")
    print(f"Radius: {radius}")
    print(f"Initial Temperature: {temp}")
    print(f"Stop Temperature: {stop_temp}")
    print(f"Timeout: {timeout}")
    print(f"Cooling Rate: {cooling_rate}")

    sa = SimulatedAnnealing(graph, graph.node_dict[start], radius, temp, stop_temp, timeout, cooling_rate)

    best_path, best_stats = sa.anneal()

    print(f"Best Path: {best_path}")
    print(f"Best Stats: {best_stats}")

    return best_path, best_stats

def help():
    print("Usage: main.py -n <start> -r <radius> [-t <temp>] [-s <stop>] [-o <timeout>] [-c <cooling>] [-d]")
    print("Options:")
    print("  -n, --node     Start node")
    print("  -r, --radius   Radius")
    print("  -t, --temp     Initial temperature (default: 100.0)")
    print("  -s, --stop     Stop temperature (default: 1e-8)")
    print("  -o, --timeout  Timeout iterations (default: 100000)")
    print("  -c, --cooling  Cooling rate (default: 0.995)")
    print("  -d, --display  Display map")
    exit(1)

# CLI Arguments, Start Node, Radius, T (opt), Stop Temp (opt), Timeout (opt), Cooling Rate (opt)
opts = {
        "node": {"short": "-n", "long": "--node", "help": "Start node"},
        "radius": {"short": "-r", "long": "--radius", "help": "Radius"},
        "temp": {"short": "-t", "long": "--temp", "help": "Initial temperature (default: 100.0)"},
        "stop": {"short": "-s", "long": "--stop", "help": "Stop temperature (default: 1e-8)"},
        "timeout": {"short": "-o", "long": "--timeout", "help": "Timeout iterations (default: 100000)"},
        "cooling": {"short": "-c", "long": "--cooling", "help": "Cooling rate (default: 0.995)"},
        "display": {"short": "-d", "long": "--display", "help": "Display map"},
        }

parser = argparse.ArgumentParser(description="Simulated Annealing for Travelling Salesman Problem")
for opt, val in opts.items():
    parser.add_argument(val["short"], val["long"], help=val["help"])
args = parser.parse_args()

start = args.node
radius = args.radius
T = 100.0 if args.temp is None else float(args.temp)
stop_temp = 1e-8 if args.stop is None else float(args.stop)
timeout = 100000 if args.timeout is None else int(args.timeout)
cooling_rate = 0.995 if args.cooling is None else float(args.cooling)
display = True if args.display is not None else False

print_all(original, start, radius, T, stop_temp, timeout, cooling_rate)





"""
args = sys.argv[1:]

opts = "n:r:t:s:o:c:d?"
longopts = ["node=", "radius=", "temp=", "stop=", "timeout=", "cooling=", "display", "help"]

# Default values
start = None
radius = None
T = 100.0
stop_temp = 1e-8
timeout = 100000
cooling_rate = 0.995
display = False

try:
    arguments, values = getopt.getopt(args, opts, longopts)

    for arg, val in arguments:
        match arg:
            case "-n", "--node":
                start = val
            case "-r", "--radius":
                radius = int(val)
            case "-t", "--temp":
                T = float(val)
            case "-s", "--stop":
                stop_temp = float(val)
            case "-o", "--timeout":
                timeout = int(val)
            case "-c", "--cooling":
                cooling_rate = float(val)
            case "-d", "--display":
                display = True
            case "-?", "--help":
                help()
except getopt.error as err:
    print(str(err))

if start is None or radius is None:
    print("Error: Start node and radius are required")
    help()

print_all(original, start, radius, T, stop_temp, timeout, cooling_rate)
"""

if display:
    original.display("map.png")



