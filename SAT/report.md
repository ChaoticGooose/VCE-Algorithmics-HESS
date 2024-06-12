# VSV VCE Algorithmics SAT U3 #
Task 1: **Shortest Path**
<sub>The response force needs to be deployed from Bendigo to reach the target site as quickly as possible.</sub>

Task 2: **Radius Vaccination**
<sub>The medical team must visit every town within a given radius and vaccinate everybody in the shortest possible time.</sub>

### Table of Contents ###
1. [Model and Specification](#part-1-model-and-specification)
2. [Design your Algorithm](#part-2-design-your-algorithm)
3. [Implementation](#part-3-implementation)
4. [Evaluation](#part-4-evaluation)

## 0. Aim and Introduction ##
The following report outlines the design and implementation of two algorithms to address real-world problems faced by the Sanitation and Medical Teams in a fictional urban environment. The goal is to develop suitable algorithms to combat the spread of "Pangobats", Making use of efficient algorithmic design to optimise the task.

## Part 1: **Model and Specification** ##
### 1.1 Model ###
The algorithmic problem at hand is to design an efficient system for managing the government’s response to sightings of Flying Pangolins (Pangobats) in Victoria, which are carriers of Itchy Nose Syndrome (INS). This disease is highly contagious and detrimental, necessitating a swift and coordinated response. The government has established two teams to address this issue: a Response Team and a Medical Team. The Response Team is tasked with quickly reaching the target site, while the Medical Team must visit every town within a given radius and vaccinate all residents. In the real world, the spread of infectious diseases carried by animals like Pangobats can have severe consequences for public health and the economy. Timely response, efficient resource allocation, and strategic planning are crucial to containing such outbreaks. The algorithmic solution needs to optimize the deployment of resources, minimize response time, prioritize vulnerable populations, and ensure thorough coverage of affected areas to mitigate the impact of INS outbreaks.

The teams are provided with a map of the relevant areas. The map is represented using a Graph ADT, with each town as a node and each road as an edge. This graph will be a weighted cyclic undirected graph, with each edge containing the length of the road as the weight and the time to travel the road as additional attributes. Each node also contains attributes such as population, median household weekly income, average age, and 2D GPS coordinates. This information is vital to determining the order of towns to visit and determining the quickest path to take. For simplicity, the third dimension of towns (altitude) is disregarded, thereby excluding altitude-based distance calculations. In the real world this would be relevant as the Pandobats would have to fly over mountains thereby increasing the time taken to reach the target site, this is not taken into account in this model.

The Medical team operates as a single unit with only one vehicle, It is assumed that the vehicle can carry an infinite amount of vaccines, has an infinite amount of fuel and can vaccinate any sized town instantaneously once reached. In addition to this, road traffic will not be taken into account in the model. For the purposes of this model these assumptions are made, although further iterations of the algorithms would have to take into account refuelling and restocking for longer routes containing a greater number of towns and potential traffic. With these assumptions in mind, the vehicle will travel at a constant speed and can visit all towns within the radius in a single trip.

The Algorithm that must be designed to generate a path for the medical team to travel is generically known as a Travelling Salesman Problem (TSP), where the goal is to visit every node in a graph once and return to the starting point in the shortest possible time. In this instance the algorithm will need to find all the relevant nodes inside of a given radius, then generate a path connecting all of the relevant nodes. As the generated series of nodes is not always Hamiltonian, the generated path may revisit nodes where reasonable. This is a slight difference between a TSP, where a TSP would generate a Hamiltonian Circuit, This algorithm will create a more general circuit.

When calculating the relevant towns within the radius the distance between towns should be calculated using the Haversine formula, which takes into account the curvature of the Earth, providing a more accurate distance between two points on the Earth's surface. As the Pangobats can fly, they needn't follow the roads and can travel in a straight line between towns. This is important to note as the distance between towns is not the same as the distance between roads, this is taken into account when calculating the distance between towns within the radius. The Edge weights (Time taken to Travel) and Edge Length should be used when calculating the shortest path between two towns for the vehicle to travel.

In this task real world considerations must be taken into account to ensure the algorithm can prioritise higher risk towns. The algorithm must prioritise towns based on population density, income, and age. This is important as outbreaks are more common in towns with higher population density, lower income, and older populations. Addressing these towns first should lower the impact that an outbreak can have on the surrounding community and stop further spread, and therefore minimising the overall resources required to completely exterminate the population of Pandobats and Treat the overall community. This prioritisation will be represented through the use of a priority queue, which will order the towns based on the given attributes. Towns with a more dense population will be treated with the greatest priority, Then towns with a higher median age and then towns with a lower median income. This will ensure that the towns most at risk are visited first, then a shorter path can be calculated between these towns to minimise travel time.

For the response team's task fewer considerations need to be taken into account for the algorithm to work effectively. The response team's primary goal is to reach the given town in the shortest possible time, starting at bendigo. This is designed to minimise the resources that the response team will need once the target site is reached, as the Pangobats will be quickly exterminated and the outbreak will be contained to only a single town. This means that no other towns will need to be stopped at and the response team can travel directly to the target site. The effect of this is that no prioritisation needs to happen for at risk towns.

The algorithm designed will assume that the response team will travel at a constant speed and will not need to stop for refuelling or restocking. This is a simplification of the real world for the sake of simplicity. Traffic and other environmental road features will also not be taken into account in the model, resulting in the response team traveling at a constant speed and never having to accellerate or decellerate.

#### Relevant ADT Signatures ####
- Graph
	- **get_edge_value**: Graph, Node, Node -> int
	- **get_node_value**: Graph, Node -> dict <span style="color:blue">*(returns node attributes as a dictionary)*</span>
	- **get_nodes**: Graph -> list
	- **get_edges**: Graph -> list
- List
	- **append**: element, List -> List
	- **contains**: element, List -> bool
	- **length**: List -> int
- Stack
	- **push**: element, Stack -> Stack
	- **pop**: Stack -> element
	- **isEmpty**: Stack -> bool
- Priority Queue
	- **enqueue**: element, int -> Priority Queue
	- **dequeue**: Priority Queue -> element
	- **isEmpty**: Priority Queue -> bool
- Queue
	- **enqueue**: element, Queue -> Queue
	- **dequeue**: Queue -> element
	- **Peek**: Queue -> element

- Map
	- **addTown**: Map, Town -> Map
	- **addRoad**: Map, Road -> Map
	- **getTowns**: Map -> List
	- **getRoads**: Map -> List
	- **dist**: Map, Town, Town -> float
    - **time**: Map, Town, Town -> int
    - **get_edge_attribute**: Map, Town, Town, str -> int

### Relevent Function Signatures ###
- **bfs**: Graph, Node, int -> List
- **tsp**: Graph, List -> List, int
- **dijkstra**: Graph, Node, Node -> List, int
- **haversineDistance**: float, float, float, float -> float
- **floydWarshall**: List, List -> List, List
- **permutations**: List -> Iterator

![Graph Model](./map.png)


## Part 2: **Design your Algorithm** ##
As previously mentioned, the medical team's task can be modelled as a Travelling Salesman Like Problem (TSP). There are a number of suitable algorithms that all have advantages and disadvantages. The most notable of these being running time and accuracy. As TSP style problems are NP-Hard -- meaning that the time taken to solve the problem cannot be determined in polynomial time -- these are directly related to each other. The more accurate the solution, the longer it will take to solve -- With a few exceptions. These solution types are generally known as Approximate and Exact solutions, with the former being faster but less accurate and the latter being more accurate but slower.

The first operation the algorithm must complete is determining the relevant towns within the radius. The simplest way to find this is to use either a Breadth First Search (BFS) or a Depth First Search (DFS). Depth first search finds the relevant towns by searching, as the name suggests, the depth of the graph before moving on. This is not optimal for the task at hand, as the algorithm doesn't guarantee that the towns are searched in an effective order -- It can get stuck exploring a branch of nodes deeply before exploring other branches, which might contain towns within the given radius. A Breadth First Search is a better fit for the task as it searches the graph layer by layer, in a breadth like manner. This means that towns closer to the source can be found first, and the search can be stopped once all towns in the layer are outside the radius.

The simplest solution for a TSP style problem is to generate all possible paths and calculate the time taken to travel each path. This is known as the Brute Force method. This method is the most accurate but is also the slowest. It computes all permutations of the towns and calculates the travel time of each path, before choosing the path with the shortest travel time. This method is quite slow on larger sets as the number of possible paths increases factorially with the number of towns. This method is not suitable for large sets of towns, as the time taken to compute the solution is not feasible, This means that the team would not be able to scale their operation up if the outbreak were to spread outside of Victoria. Although, as this type of algorithm finds all possible solutions, it falls under the exact category of solutions, where it is guaranteed to find the optimal solution. It is also more accurate than other solutions, as it does not make any assumptions about the data and will never reach a local minimum. This means that when the number of towns is small, this method is the best to use.

A more optimal TSP solution is the Nearest Neighbour Algorithm. This algorithm is a heuristic solution, meaning that it is not guaranteed to find the optimal solution. This algorithm works by starting at a random town and then visiting the nearest town, then the nearest town to that town, and so on until all towns have been visited. This algorithm is much faster than the Brute Force method, but is less accurate. This algorithm is not guaranteed to find the optimal solution, as it may reach a local minimum. This means that the solution may not be the shortest path, but it will be close to the shortest path. This algorithm is much faster than the Brute Force method, as it only needs to calculate the distance between each town once, and then it can find the shortest path. The main issue with this algorithm for the task listed here is the graph generated as a function of the radius is not always hamiltonian, meaning that the nearest neighbour algorithm may not always find a valid solution or the shortest path. The main cause of this is the core idea of the algorithm, where it chooses only the most locally optimal solution at each step, this means that if node A is the closest to node B, then the algorithm will choose node A, even if the only way to get to node C is through node B. This is the main issue with the nearest neighbour algorithm, as it may completely miss required nodes, or choose a suboptimal path. This is not suitable for the task at hand, as the algorithm must visit all towns within the radius, and the nearest neighbour algorithm may not always do this.

Without the use of Dynamic Programming, Simulated annealing, Backtracking, Genetic Algorithms or Ant Colony Optimization, these are the two most practical algorithms. The Nearest Neighbour solution would potentially require changes in the graph to make it hamiltonian, and it therefore would not generate the most optimal path. A Brute force algorithm would take longer to run and be less scalable than a Nearest Neighbour like design, although it is an Exact algorithm rather than a Heuristic one. The most suitable algorithm for this task is a Brute Force algorithm, as the number of towns is most likely to be less than ten, and the accuracy of the solution is more important than the time taken to compute the solution. If the number of towns were to increase, then a more scalable solution would be required, at the cost of accuracy. This is the most suitable algorithm for the task at hand, as the number of towns is small and the accuracy of the solution is more important than the time taken to compute the solution.

This implementation will only include priority for towns with a higher average age for the sake of simplicity. Other prioritizations can be added in further iterations of the algorithm, accounting for extensibility and modularity.

A good optimization to note for the Brute Force solution is to use the Floyd Warshall Algorithm to calculate the shortest path between each pair of towns. This will mean that the algorithm does not need to calculate the distance between each pair of towns each time it generates a path, as the shortest path between each pair of towns will already be known. This will make the algorithm much faster, as it only needs to calculate the distance between each pair of towns once, and then it can find the shortest path.

The Research team's task can be modelled as a Shortest Path Problem, where the goal is to find the shortest path between Bendigo and the target site. This is a well-known P problem in graph theory, and there are several algorithms that can be used to solve it. For this task we will focus on exact solutions over heuristic solutions as the given restrictions limit the types of solutions that can be used.

Once again the simplest solution for a Shortest Path Problem is to generate all possible paths and calculate the time taken to travel each path. This type of algorithm has almost no advantages over other solutions, as it is not the only exact solution, and it is the slowest. Alike the TSP Brute Force method, The number of towns increases factorially with the number of towns, making it useless for sets of towns with a number of towns greater than four. It still requires a large amount of memory and cpu cycles, therefore should never be used.

One of the two most common algorithms for this task is Dijkstra's Algorithm. Dijkstra's Algorithm is a greedy algorithm that finds the shortest path between two nodes in a graph. It works by starting at the source node and then visiting the nearest node, then the nearest node to that node, and so on until the target node is reached. This algorithm is guaranteed to find the shortest path between two nodes, as it calculates the shortest path to each node from the source node. This algorithm is much faster than the Brute Force method, as it only needs to calculate the distance between each node once, and then it can find the shortest path. This algorithm is quite suitable for the task at hand, as the goal is to find the shortest path between two nodes, and the accuracy of the solution is more important than the time taken to compute the solution. This algorithm would not work though if the graph contained negative edge weights, as the algorithm would not be able to find the shortest path. This is not an issue in this task, as the edge weights are all positive.

The other of these two algorithms is the Bellman-Ford Algorithm. The Bellman-Ford Algorithm is another exact solution to the Shortest Path Problem. This algorithm on the principle of edge relaxation, where the algorithm relaxes the edges of the graph until the shortest path is found. This algorithm is slower than Dijkstra's Algorithm, as it relaxes all edges in the graph for each node, rather than just the nearest node -- meaning it is not a greedy algorithm. This algorithm is more suitable for graphs with negative edge weights, as it can find the shortest path even if the graph contains negative edge weights. Another case where this algorithm is more suitable is when the graph is dense, as the algorithm does not need to maintain priority queues for each node.

The most suitable algorithm for this task is Dijkstra's Algorithm, as the graph is not dense and does not contain negative edge weights. This algorithm is faster than the Bellman-Ford Algorithm, and it is more suitable for the task at hand. In the implementation a priority queue will be used to optimise the algorithm slightly. This will mean that paths with shorter distances will be searched first. This algorithm is guaranteed to always find the shortest path between two nodes, this means there are no advantages to brute forcing the solution. In this task there would be almost no reason to choose Bellman Ford or Brute Force over Dijkstra's Algorithm.

## Part 3: Pseudocode ##
### Radius Search ###
	Function bfs(G, start, radius):
    	visited = new Dictionary ADT
    	queue = new Queue ADT

    	Add start to queue
    	Add start to visited with distance 0

    	While queue is not empty:
        	current = queue.dequeue()
        	for each neighbour of current:
                // Calculate distance from the start account for curvature of the Earth
            	distance = haversineDistance(
                	start.latitude,
                	start.longitude,
                	neighbour.latitude,
                	neighbour.longitude
            	)
            	if distance > radius:
                	continue

            	if neighbour not in visited or distance < visited[neighbour]:
                	Add neighbour to visited with distance
                	Add neighbour to queue

    	nodes = Keys of visited

    	return nodes

### Floyd Warshall Algorithm ###
    Function floydWarshall(G, nodes):
    	n = length(nodes)

    	dist_matrix = new 2D array of size n x n with default value infinity
    	prev_matrix = new 2D array of size n x n with default value null

    	// Fill the distance matrix with adjacency matrix values
    	For i in range(n):
        	For j in range(n):
            	dist_matrix[i][j] = G.get_edge_attribute(nodes[i], nodes[j], 'time')
            	if dist_matrix[i][j] != infinity:
                	prev_matrix[i][j] = j

    	// Floyd Warshall Algorithm
    	For k in range(n):
        	For i in range(n):
            	For j in range(n):
                	if dist_matrix[i][j] > dist_matrix[i][k] + dist_matrix[k][j]:
                    	dist_matrix[i][j] = dist_matrix[i][k] + dist_matrix[k][j]
                    	prev_matrix[i][j] = prev_matrix[i][k]

    	return dist_matrix, prev_matrix

### Medical Team Path (TSP) ###
	Function tsp(G, nodes):
    	nodes = bfs(G, start, radius)
    	n = length(nodes)

        dist_matrix, prev_matrix = floydWarshall(G, nodes)

    	// Brute Force Algorithm
    	min_path = null
        scaled_min_time = infinity
    	min_time = infinity

        // Calculate average age for each node
        avg_ages = new Dict ADT of average ages for each node

    	For each permutation of numbers 0-n:
            if path[0] == start:
                path = permutation + start
        	    path_time = 0
                scaled_time = 0 // Used to scale time based on priority

        	    for i=0 to n:
                    if i < n-1:
                       current_dist = dist_matrix[path[i]][path[i+1]]
                    else:
                        current_dist = dist_matrix[path[i]][start]

                    // Priority for towns with higher average age
                    avg_age = avg_ages[nodes[i]]
                    if avg_age > 50:
                        scaled_time += current_dist * 0.8
                    else:
                        scaled_time += current_dist

            // Update min path if new path is shorter
        	if scaled_time < scaled_min_time:
            	min_time = path_time
            	min_path = path
                scaled_min_time = scaled_time

    	return min_path, min_time

### Response Team Path (Shortest Path) ###
	Function dijkstra(G, start, target):
    	pq = New Priority Queue ADT // Priority Queue based on distance Min Heap
    	visited = new Dictionary ADT
    	prev = new Dictionary ADT

    	Add start to pq with priority 0
    	Add start to visited with distance 0
    	Add start to dist with distance 0

    	// Initialize all other nodes
    	for node in G.nodes:
        	if node != start:
            	Add node to pq with priority infinity
            	Add node to visited with distance infinity
            	Add node to dist with distance infinity

    	// Find shortest path to each node
    	while pq is not empty:
        	current = pq.dequeue()
        	for each neighbour of current:
            	time = G.get_edge_attribute(current, neighbour, 'time')
            	alt = (dist + visited[current])

                // Update distance if shorter path found
            	if alt < visited[neighbour]:
                	visited[neighbour] = alt
                	prev[neighbour] = current
                	pq.enqueue(neighbour, alt)

    	path = []
    	current = target
        // Reconstruct path from target to start
    	while current != start:
        	Insert current at the beginning of path
        	current = prev[current]

    	return path, visited[target]

## Part 4: **Evaluation** ##
The model and algorithm design presented here demonstrate a comprehensive approach to managing the government’s response to Flying Pangolin sightings in Victoria, specifically addressing the Itchy Nose Syndrome outbreak. By incorporating key factors such as population density, income levels, average age, and geographic locations of towns, the model accurately reflects real-world complexities in pest control and infectious disease outbreak managment.

A Breadth-First Search algorithm was chosen as the most suitable method for finding towns within a given radius, as it ensures that towns are visited in an efficient order and stops the search once all towns within the radius have been found. This algorithm provides a clear list of towns for the medical team to visit, radiating outwards from the source town. For the medical team's path, a Brute Force algorithm was selected as it is the most accurate and simple solution for finding the shortest path between towns. This algorithm guarantees the optimal path, although it is slower than other heuristic solutions. The accuracy of the solution is more important than the time taken to compute the solution, as the number of towns is small and the goal is to minimise travel time. This means though, that the algorithm is not scalable, and would not be suitable for radiuses greater than ~100km.

The Medical team's algorithm also factors in the priority of towns based on population density, income, and age, ensuring that higher-risk towns are visited first. This models what a real-world medical team and government response would look like, as it prioritises vulnerable populations and minimises the impact of an outbreak therefore saving time, money and resources. Another key feature to modelling the task accurately is the use of the Haversine formula to calculate the straight-line distance between towns, accounting for the curvature of the Earth. This would more accurately represent the distance and path the Pangobats would take as they fly between towns.

For the Response team's task, Dijkstra's Algorithm was chosen as the most suitable method for finding the shortest path between Bendigo and the target site. This algorithm is faster than the Bellman-Ford Algorithm and is more suitable for the task at hand, as the graph is not dense and does not contain negative edge weights. This means that the Response team can decide on the path to take in the shortest amount of time, and the algorithm is guaranteed to find the absolute shortest path between two nodes. This will minimise the impact a specific outbreak may have on the community and surrounding areas, also saving time, money and resources.

The model and algorithms presented here provide a solid foundation for addressing the challenges posed by the Pengobat sightings and Itchy Nose Syndrome outbreak in Victoria. By incorporating real-world data and algorithmic design principles, the government can effectively manage the response to these threats and protect public health and safety. The algorithms are designed to be as accurate as possible first, before providing a more efficient solution.This ensures that they can be applied to a wide range of scenarios and respond to changing conditions. The model can be further refined and expanded to address additional factors and optimise resource allocation, making it a valuable tool for pest control and infectious disease management.
