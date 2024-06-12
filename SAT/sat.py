import networkx as nx
import matplotlib.pyplot as plt
import csv

G = nx.Graph()

# Read Nodes in from nodes.csv
with open('nodes.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        G.add_node(row[0], pos=(float(row[4]), float(row[5])))
        if row[0] == "Mildura":
            pos = nx.get_node_attributes(G, 'pos')
            print(pos)
            print(repr(row[0]))
        
        # Set node attributes name, population, income, latitude, longitude
        G.nodes[row[0]]['name'] = row[1]
        G.nodes[row[0]]['population'] = row[2]
        G.nodes[row[0]]['income'] = row[3]
        G.nodes[row[0]]['latitude'] = row[4]
        G.nodes[row[0]]['longitude'] = row[5]

# Read Edges in from edges.csv
with open('edges.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        G.add_edge(row[0], row[1], weight=row[3])
        # Add edge attribute distance
        G.edges[row[0], row[1]]['distance'] = row[2]

# Draw the graph
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue')
plt.show()
