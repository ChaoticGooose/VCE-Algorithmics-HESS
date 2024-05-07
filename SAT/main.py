import networkx as nx
import numpy as np 
import matplotlib.pyplot as plt
import csv
import queue
from dataclasses import dataclass, field
from typing import Any
from itertools import permutations
import math

@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any=field(compare=False)

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
        # otherwise returns infinity.

        for edge in self.edges:
            if edge.place1 == place1 and edge.place2 == place2:
                return edge.dist
            if edge.place1 == place2 and edge.place2 == place1:
                return edge.dist
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

    # Bruteforce TSP
    def tsp(self, start, radius):

        nodes = self.bfs(start, radius)
        n = len(nodes)
        
        # Brute force algorithm to find the shortest path connecting all nodes using the matrix of distances
        min_path = None
        min_dist = float("inf")
        for path in permutations(range(n)):
            path = list(path)
            path_dist = 0
            for i in range(n-1):
                path_dist += self.get_dist(nodes[path[i]].name, nodes[path[i+1]].name)
            if path_dist < min_dist:
                min_dist = path_dist
                min_path = path

        return [nodes[i] for i in min_path], min_dist
        

    def dijkstra(self, start, target):
        # Create a priority queue
        pq = queue.PriorityQueue()
        pq.put(PrioritizedItem(0, start)) # Add the starting node to the priority queue with a priority of 0

        # Create a dictionary to store the distance from the starting node to each node
        visited = {start: 0.0}
        
        # Create a dictionary to store the previous node in the shortest path
        previous = {start: None}

        # Set all nodes to infinity distance with no prev and add them to the priority queue
        for node in self.nodes:
            if node != start:
                visited[node] = float("inf")
                previous[node] = None
                pq.put(PrioritizedItem(float("inf"), node)) # Add the node to the priority queue with a distance of infinity
        
        # Find the shortest path to each node
        while len(pq.queue) > 0:
            current = pq.get().item # Get the node with the shortest distance
            for neighbour in current.neighbours:
                dist = self.get_dist(current.name, neighbour)
                neighbour = self.node_dict[neighbour] # Convert the neighbour name to a node object
                
                alt = dist + visited[current] # Calculate the new distance to the neighbour
                if alt < visited[neighbour]: # If the new distance is less than the previous distance
                    visited[neighbour] = alt
                    previous[neighbour] = current
                    pq.put(PrioritizedItem(alt, neighbour)) # Add the neighbour to the priority queue with the new distance

        # Create a list of nodes in the shortest path
        path = []
        current = target
        while current != None:
            path.insert(0, current)
            current = previous[current]

        return path, visited[target]

# Create a sub graph object only containing given nodes
    def create_subgraph(self, nodes):
        subgraph = Graph()
        subgraph.nodes = nodes

        # Get existing edges between nodes
        for edge in self.edges:
            if self.node_dict[edge.place1] in nodes and self.node_dict[edge.place2] in nodes:
                subgraph.edges.append(edge)

        subgraph.node_dict = {node.name: node for node in subgraph.nodes}


        return subgraph



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

path = original.tsp(original.node_dict["Alexandra"], 100)
print([node.name for node in path[0]])
print(path[1])

# Display the object, also saving to map.png
original.display("map.png")

# You will add your own functions under the Graph object and call them in this way:
# original.findpath("Alexandra")
