import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import math
from itertools import permutations


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
        return -1

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

    # Find shortest hamiltonian path covering all nodes within a given radius
    # Nearest neighbour algorithm
    # https://medium.com/@suryabhagavanchakkapalli/solving-the-traveling-salesman-problem-in-python-using-the-nearest-neighbor-algorithm-48fcf8db289a
    def tsp(self, start, radius):
        # Create a list of all nodes within the given radius of the starting nodes
        nodes = self.bfs(start, radius)
        nodes = [self.get_node(node) for node in nodes]

        # Find the number of towns
        n = len(nodes)

        # Create an n*n matrix to store distances between nodes
        distances = np.zeros((n, n))

        # Fill the matrix with the distances between nodes, None for the same node
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j:
                    distances[i][j] = None
                else:
                    distances[i][j] = self.haversine(
                        nodes[i].lat, nodes[i].long, nodes[j].lat, nodes[j].long
                    )

        # Create a list to store the path
        path = [start]
        # Create a list to store nodes visited
        visited = [start] # Probably should be a set

        while len(visited) < n:
            prev = path[-1] # Get the previous node

            # Get index of the previous node from nodes list
            prev_index = nodes.index(prev)

            # Find the nearest neighbour using the distance matrix
            # Then resolve the index to the node
            nearest_town = min([(i, distances[prev_index][i]) for i in range(n) if nodes[i] not in visited], key=lambda x: x[1])
            nearest_town = nodes[nearest_town[0]]

            # Add the nearest neighbour to the path and visited sets
            path.append(nearest_town)
            visited.append(nearest_town)
        path.append(start) # Return to the starting node
        return path

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
                neighbour = self.get_node(neighbour)
                # Calculate the distance from the starting node to the neighbour
                distance = visited[node] + self.haversine( # Add the previous distance to the distance to the neighbour to generate the total distance
                    node.lat, node.long, neighbour.lat, neighbour.long
                )

                # If the distance is outside the radius, break the loop
                if distance > radius:
                    break

                # If the neighbour has not been visited or the new distance is less than the previous distance
                if neighbour not in visited or distance < visited[neighbour]:
                    # Update the distance to the neighbour
                    visited[neighbour] = distance
                    # Add the neighbour to the queue
                    queue.append(neighbour)

        # Convert the dictionary of visited nodes to a list of node names
        nodes = list(visited.keys())
        nodes = [node.name for node in nodes]
        return nodes # Return the list of nodes within the radius

    def get_node(self, name: str):
        # Return the node with the given name
        for node in self.nodes: # Iterate through the nodes in the graph
            if node.name == name: # If the name of the node matches the given name
                return node
        return None

    def searchteam(self, target, radius):
        pass

    def vaccinate(self, target, radius):
        pass

    def findpath(self, target):
        pass


# These commands run the code.

# Create a new graph object called 'original'
original = Graph()

# Load data into that object.
original.load_data()

path = original.tsp(original.get_node("Mildura"), 300)
print([node.name for node in path])

# Display the object, also saving to map.png
# original.display("map.png")

# You will add your own functions under the Graph object and call them in this way:
# original.findpath("Alexandra")
