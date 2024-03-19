# Finds the shortest path between covering all nodes in the graph.
# INPUT: Graph object
# OUTPUT: List of sets of edges, each set is a path, the list is the order of paths
def shortestPath(graph) -> list:
    # modified dijkstra's algorithm
    visited = [graph.nodes[0]]
    paths = []

    graph_length = len(graph.nodes)

    while len(visited) < graph_length:
        # find the closest node to the visited nodes
        min_dist = float('inf')
        for node in visited:
            for neighbour in node.neighbours:
                if neighbour not in visited:
                    dist = graph.get_dist(node, neighbour)
                    if dist < min_dist:
                        min_dist = dist
                        closest_node = neighbour
                        closest_edge = (node, neighbour)
        # add the closest node to the visited nodes
        visited.append(closest_node)
        paths.append(closest_edge)
    return paths
