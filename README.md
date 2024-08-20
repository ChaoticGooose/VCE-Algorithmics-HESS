# Algorithmics SAT Part 3: Advanced Algorithmic Design
## 1. Improved Algorithm design
### TSP Algorithm (Simulated Annealing)
```
Algorithm SimulatedAnnealing(graph, start, radius, T=100.0, stop_temp=-1, timeout=100000, cooling_rate=0.995)
// Input graph, start node, and radius for the simulated annealing algorithm.
// Initializes the parameters and performs the simulated annealing process to find an optimized path.

initialize T to T
initialize STOP_T to 1e-8 if stop_temp is -1, otherwise stop_temp
initialize cooling_rate to cooling_rate

nodes := set of nodes within the radius from the start node, found using BFS
all_nodes := set of all nodes in the graph
timeout_iter := timeout
iter := 0

best_stats := (∞, ∞)
best_path := None

path_list := empty list

Function path_stats(path)
// Calculates the total distance and time for the given path.

total_dist := 0
total_time := 0

for i from 0 to length(path) - 2 do
    total_dist := total_dist + distance between path[i] and path[i+1]
    total_time := total_time + time between path[i] and path[i+1]
end for

return (total_dist, total_time)
end Function

Function initial_solution()
// Uses a greedy algorithm to find the initial solution to the Travelling Salesman Problem.

solution := [start]
unvisited := set of nodes that are within the radius, excluding the start node

while unvisited is not empty do
    current := last node in solution
    current_neighbours := neighbours of current converted to node objects

    if there are unvisited nodes in current_neighbours then
        next_node := node in unvisited and current_neighbours with minimum travel time from current
        remove next_node from unvisited
    else
        next_node := node in all_nodes excluding unvisited with minimum travel time from current
        remove next_node from all_nodes
    end if

    append next_node to solution
end while

path_stats := path_stats(solution)

if path_stats[1] < best_stats[1] then
    best_stats := path_stats
    best_path := solution
end if

append path_stats to path_list

return (solution, path_stats)
end Function

Function prob_accept(candidate)
// Calculates the probability of accepting a candidate solution based on the current temperature and the difference in cost.

return exp(-abs(candidate - best_stats[1]) / T)
end Function

Function accept(candidate, stats)
// Accepts the candidate solution if it is better than the current solution.
// If the candidate solution is worse, accepts it with a probability based on the current temperature.

if stats[1] < current_stats[1] then
    current_solution := candidate
    current_stats := stats
    if stats[1] < best_stats[1] then
        best_stats := stats
        best_path := candidate
    end if
else
    if random value < prob_accept(stats[1]) then
        current_solution := candidate
        current_stats := stats
    end if
end if
end Function

Function anneal()
// Performs the simulated annealing process to optimize the path.

current_path, current_stats := initial_solution()

while T >= STOP_T and iter < timeout_iter do
    path := copy of current_path
    l := random integer between 2 and length(path) - 1
    i := random integer between 0 and length(path) - l

    reverse the sublist of path from i to (i + l)

    new_stats := path_stats(path)
    accept(path, new_stats)

    T := T * cooling_rate
    iter := iter + 1
end while

print iter
print T

return (best_path, best_stats)
end Function

end Algorithm
```

This TSP algorithm utilises the Simulated Annealing approch for finding an optimal solution to the Traveling Salesman Problem. The algorithm starts by initializing the parameters and finding the nodes within the specified radius from the start node using Breadth-First Search (BFS), alike the origional Brute force algorithm. It then uses a greedy, Nearest Neighbour, Algorithm to find the initial solution to the TSP, Simply picking the next closest node from the start. The simulated annealing process is performed to optimize the path by accepting candidate solutions based on the current temperature and the difference in cost. The algorithm iterates until the temperature reaches a given stopping temperature or a given timeout limit (default 100,000)  is reached. The final optimized path and its statistics are then returned as the output.


### A* Algorithm (SSSP)
```
Algorithm AStar(graph, start, target)
// Input graph, start node, and target node for the A* algorithm.
// Initializes the parameters and performs the A* search to find the shortest path.

Class PathNode(node)
// A class to represent a node in the path with its cost values.

initialize node to node
initialize g (cost from start to node) to ∞
initialize h (heuristic cost from node to target) to ∞
initialize f (total cost) to ∞
end Class

Initialize graph to graph
start_node := PathNode(start)
target_node := target

start_node.g := 0
start_node.h := heuristic(start_node.node)
start_node.f := start_node.g + start_node.h

open := PriorityQueue to hold nodes to be evaluated
closed := set to hold nodes already evaluated

parent := dictionary to store the parent of each node

path := empty list to store the final path
path_found := False

Function heuristic(node)
// Calculates the heuristic cost (h) from the current node to the target node.

return haversine distance between node and target_node
end Function

Function find_path()
// Performs the A* search algorithm to find the shortest path from start to target.

insert start_node into open with priority start_node.f

while open is not empty do
    current := node in open with the lowest f value

    if current.node equals target_node then
        path_found := True
        break
    end if

    add current.node to closed

    for each neighbour in current.node.neighbours do
        neighbour_node := PathNode(neighbour node from graph)

        if neighbour_node in closed then
            continue to next neighbour
        end if

        g := current.g + distance between current.node and neighbour_node
        h := heuristic(neighbour_node.node)
        f := g + h

        if f < neighbour_node.f then
            neighbour_node.g := g
            neighbour_node.h := h
            neighbour_node.f := f

            set parent of neighbour_node to current.node

            insert neighbour_node into open with priority neighbour_node.f
        end if
    end for
end while

if path_found then
    current := target_node
    while current is not start_node.node do
        insert current at the beginning of path
        current := parent[current]
    end while

    insert start_node.node at the beginning of path

    return path
end if

return None
end Function

end Algorithm
```

This A* algorithm is used for finding the shortest path from a start node to a target node in a graph. The algorithm initializes the parameters and uses a priority queue to hold nodes to be evaluated based on their total cost (f). It calculates the heuristic cost (h) from the current node to the target node using the Haversine distance. The A* search algorithm iterates until the target node is reached or all nodes are evaluated. The final path is reconstructed by tracing back the parent nodes from the target node to the start node. The algorithm returns the shortest path if found, otherwise None.

## 2. Advanced Algorithms VS Naive Algorithms
### Comparison of TSP Algorithms
This new algorithm improves on the original brute force algorithm through the use of Simulated Annealing, which allows for a solution to be found in fewer iterations at the cost of some accuracy. Simulated Annealing is a metaheuristic algorithm, inspired by metalwork, that uses a probabilistic approach to find the global optimum. The algorithm finds a inital solution that may not be optimal, then iteratively explores neighboring solutions by making small changes to the current solution. If a new solution is better, it is accepted; otherwise, it might still be accepted based on a probability that decreases with temperature, simulating the process of annealing in metalwork. This probability is given by the equation \(e^{-\frac{|candidate - best|}{T}}\), where T is the current temperature, candidate is the cost of the new solution, and best is the cost of the current best solution. This probability allows the algorithm to escape local minima in search of a globally optimal solution. 

Due to the nature of the algorithm though, it is not guaranteed to find the most optimal solution contained in the graph, even if let run for an infinite amount of time. It does not guarantee the most optimal solution because the algorithm's success depends on the cooling schedule and random transitions. As the temperature decreases, the algorithm becomes less likely to accept worse solutions, potentially trapping it in a local minimum rather than the global minimum.  Therefore, while it increases the likelihood of finding an optimal or near-optimal solution, it cannot ensure that the absolute best solution will always be found.

In contrast, the brute-force approach, implemented previously, uses an exaustive search that considers all possible permutations of the nodes, excluding the start node, and evaluates each path's total distance and time. This means that the brute force algorithm is always gaurenteed to find the most optimal solution possible, at the cost of processing time and memory requirements. This also means that the Brute Force algorithm is deterministic, and will always return the same result given the same input, unlike the Simulated Annealing algorithm which may return different results on different runs.

(Where V is the number of nodes and E is the number of edges in the graph)
It is also useful in this comparison to note the Time and Space complexities of these algorithms. As established in SAT Part 2, The implimentation of Brute Force TSP previously was `O(V+E+V!)` and a space complexity of `O(V)`, as at any given point the algorithm only stores the best and current permutation of the path. As for the Simulated Annealing TSP algorithm, the time complexity can be difficult to express precisely due to its heuristic nature but generally performs much better in practice than brute-force approaches. The complexity depends on factors like the number of iterations, the cooling rate, and the size of the problem (number of nodes), but is typically considered to be `O(V)` or `O(V^2)`. The space complexity is much more straight forward, being `O(V)`, as it only requires the storage of the current path and the best path found so far.

Through this analysis, it is shown that the Simulated Annealing algorithm is a more efficient and practical approach for solving the TSP problem in real-world scenarios, where the goal is to find a near-optimal solution in a reasonable amount of time. The algorithm is much more scalable and can handle much larger road networks, being reasonably scalable to the entireity of Australia, if the Itchy Nose Syndrome had spread further. Although the Simulated Annealing Algorithm may not always find the most optimal solution, on average it will find a solution that is close to the optimal solution in a much shorter time than the brute force algorithm and in very large graphs, the amount of time to find the most optimal solution can be more significant than the difference between the optimal and near-optimal solutions.
