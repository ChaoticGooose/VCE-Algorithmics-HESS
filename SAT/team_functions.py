# Finds the shortest path from root to a radius of nodes
# INPUT: Graph object, desired path radius, starting node object
# OUTPUT: List of node pairs
def shortestPath(graph, radius: int, root) -> list:
    # modified dijkstra's algorithm



    return paths



# Search graph out from a given node to find all nodes within a given radius.
# INPUT: Graph object, radius, starting node object
# OUTPUT: List of node objects
def radiusSearch(graph, radius: int, root) -> list:
    visited = set()
    result = []

    def DFS(node, depth):
        if depth > radius:
            return

        visited.add(node)
        result.append(node)

        for neighbour in node.neighbours:
            if neighbour not in visited:
                DFS(neighbour, depth + 1)
    DFS(root, 0)

    return [node.name for node in result]
