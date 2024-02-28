import networkx as nx
import matplotlib.pyplot as plt
from names import get_first_name as name
from random import random, randrange


NUMBER_OF_MONKEYS = 5 # Number of nodes in the graph

class Monkey: # Examle class
    def __init__(self, name, bananas):
        self.bananas = bananas
        self.status = int(bananas**2/100)
        self.name = name


def main():
    tribe = nx.Graph()

    # Randonly generate a graph of monkies with randomised values
    for i in range(NUMBER_OF_MONKEYS):
        tribe.add_node(i, monkey=Monkey(name(), randrange(100)))
    for i in range(NUMBER_OF_MONKEYS): # Add edges between nodes pseudorandomly
        for j in range(i+1, NUMBER_OF_MONKEYS):
            if random() < 0.5:  # Adjust the probability for edge creation as needed
                weight = abs(tribe.nodes[i]['monkey'].status - tribe.nodes[j]['monkey'].status)
                tribe.add_edge(i, j, weight=weight)


    # Display graph using matplotlib and networkx
    labels = {i: f"{tribe.nodes[i]['monkey'].name}\n(Bananas: {tribe.nodes[i]['monkey'].bananas})" for i in tribe.nodes()}  # Create labels from monkey attributes
    edge_labels = nx.get_edge_attributes(tribe,'weight')

    pos=nx.spring_layout(tribe, k=3, scale=0.5) # Generate network layout

    plt.figure(figsize=(10, 8))  # Adjust figure size as needed
    nx.draw_networkx_nodes(tribe, pos, node_size=1000, node_color='skyblue')  # Adjust node size and color as needed
    nx.draw_networkx_edges(tribe, pos, width=2, alpha=0.6, edge_color='gray')  # Adjust edge width and color as needed
    nx.draw_networkx_labels(tribe, pos, labels=labels, font_weight='bold', font_size=12)  # Adjust label font size as needed
    nx.draw_networkx_edge_labels(tribe,pos,edge_labels=edge_labels, font_size=10)  # Adjust edge label font size as needed

    plt.title("Monkey Tribe Graph", fontsize=14)  # Add a title
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
