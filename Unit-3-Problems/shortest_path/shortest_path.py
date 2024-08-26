import networkx as nx
import matplotlib.pyplot as plt
import random

class Node:
    def __init__(self, id):
        self.id = id
        self.visited = False
        self.neighbours = []

    def __str__(self):
        return str(self.id)
        
    def add_neighbour(self, neighbour):
      if neighbour not in self.neighbours:
        self.neighbours.append(neighbour)
        self.sort_neighbours()
        
        
    def sort_neighbours(self):
      sorted = []
      while len(self.neighbours) > 0:
        smallest = self.neighbours[0]
        for node in self.neighbours:
          if node.id < smallest.id:
            smallest = node
        sorted.append(smallest)
        self.neighbours.remove(smallest)
      self.neighbours = sorted

class Edge:
    def __init__(self, source, target, weight):
        self.source = source
        self.target = target
        self.weight = weight
        self.id1 = source.id
        self.id2 = target.id
        self.color = 'black'

class RandomGraph:

  def __init__(self,V_N,E_N):
    # Creates a random graph objects with V_N vertices each with E_N edges
    self.edges = []
    self.nodes = [Node(i) for i in range(1,V_N+1)]
    
    for node in self.nodes:
    
      neighbours = random.sample(self.nodes,3)
      for neighbour in neighbours:
        if node != neighbour and self.get_weight(node.id, neighbour.id) == -1:
          rw = random.randint(1,7)
          edge = Edge(node, neighbour, rw)
          self.edges.append(edge)
          node.add_neighbour(neighbour)
          neighbour.add_neighbour(node)

  def get_weight(self, node1, node2):
    # Returns the edge weight between two node IDs
    for edge in self.edges:
      if edge.id1 == node1 and edge.id2 == node2:
        return edge.weight
      if edge.id2 == node1 and edge.id1 == node2:
        return edge.weight
    return -1

  def set_colour(self, id1, id2, colour):
    # Sets the edge between id1 and id2 to the stated colour
    for edge in self.edges:
      if edge.id1 == id1 and edge.id2 == id2:
        edge.color = colour
      if edge.id1 == id2 and edge.id2 == id1:
        edge.color = colour  

  def display_graph(self):
    
    self.G = nx.Graph()
    for node in self.nodes:
      self.G.add_node(node)
    for edge in self.edges:
      self.G.add_edge(edge.source, edge.target, weight = edge.weight)
    pos = nx.spring_layout(self.G)
    nx.draw(self.G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=12, font_color='black', font_weight='bold')
      
    #edge_labels = nx.get_edge_attributes(self.G, 'weight')
    edge_labels = {}
    edge_colors = []
    for edge in self.edges:
      edge_colors.append(edge.color)  
      edge_labels[(edge.source, edge.target)]= edge.weight
    
    
    
    nx.draw(self.G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=12, font_color='black', font_weight='bold', edge_color=edge_colors, width=2.0)
    nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)

    plt.title("Prims")
    plt.axis('off')
    plt.show()

  def list_nodes_and_edges(self):
    for node in self.nodes:
      print(f"Node: {node.id}")
    for edge in self.edges:
      print(f"Edge from {edge.id1} to {edge.id2} with weight {edge.weight} and colour {edge.color}")


  def reset_visits(self):
    for node in self.nodes:
      node.visited = False

  def Dijkstra(self,source,target):
    # Dijkstra's algorithm to find the shortest path between two nodes
    dist = {}
    path = []
        
    for node in self.nodes:
        dist[node.id] = 1000
    dist[source.id] = 0
    self.reset_visits()
    current = source
    while current != target:
        for neighbour in current.neighbours:
            if dist[current.id] + self.get_weight(current.id, neighbour.id) < dist[neighbour.id]:
                dist[neighbour.id] = dist[current.id] + self.get_weight(current.id, neighbour.id)
                path.append((current.id, neighbour.id))
        current.visited = True
        min_dist = 1000
        for node in self.nodes:
            if node.visited == False and dist[node.id] < min_dist:
                min_dist = dist[node.id]
                current = node
    print(f"Shortest path from {source.id} to {target.id} is {dist[target.id]}")
    print(f"Path is {path}")

    for i in range(len(path)):
        self.set_colour(path[i][0], path[i][1], 'red')

    return

RG = RandomGraph(10,3)
RG.Dijkstra(RG.nodes[0], RG.nodes[-1])
RG.list_nodes_and_edges()
RG.display_graph()
