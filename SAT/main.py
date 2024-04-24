import networkx as nx
import numpy as np 
import matplotlib.pyplot as plt
import csv
import random
import math
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
        # otherwise returns -1

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

        plt.figure(figsize=(15, 10))
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
    def tsp(self, start: Node, radius: int) -> list:
        # Get list of nodes within the radius of the starting node
        nodes = self.bfs(start, radius)

        # Create path list with the starting node
        path = [start]
    
        visited = set([start])
        # Generate shortest hamiltonian path using the nearest neighbour algorithmk
        while len(visited) < len(nodes):
            # Get the last node in the path
            current = path[-1]
            # Get the neighbours of the current node
            neighbours = current.neighbours
            # Create a list of unvisited neighbours
            unvisited = [self.node_dict[neighbour] for neighbour in neighbours if self.node_dict[neighbour] not in visited and self.node_dict[neighbour] in nodes]
            # If there are no unvisited neighbours, break the loop
            if not unvisited:
                break
            # Sort the unvisited neighbours by distance to the current node
            unvisited.sort(key=lambda node: self.haversine(current.lat, current.long, node.lat, node.long))
            # Add the closest neighbour to the path
            path.append(unvisited[0])
            visited.add(unvisited[0])

        print([node.name for node in path])


        return

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
                distance = self.haversine( # Add the previous distance to the distance to the neighbour to generate the total distance
                    start.lat, start.long, neighbour.lat, neighbour.long
                )

                # If the distance is outside the radius, break the loop
                if distance > radius:
                    print(neighbour.name, distance, radius)
                    continue

                # If the neighbour has not been visited or the new distance is less than the previous distance
                if neighbour not in visited or distance < visited[neighbour]:
                    # Update the distance to the neighbour
                    visited[neighbour] = distance
                    # Add the neighbour to the queue
                    queue.append(neighbour)

        # Convert the dictionary of visited nodes to a list of node names
        nodes = list(visited.keys())
        return nodes # Return the list of nodes within the radius
    
# These commands run the code.

# Create a new graph object called 'original'
original = Graph()

# Load data into that object.
original.load_data()

towns = original.tsp(original.node_dict["Warracknabeal"], 150)

# Display the object, also saving to map.png
original.display("map.png")

# You will add your own functions under the Graph object and call them in this way:
# original.findpath("Alexandra")
