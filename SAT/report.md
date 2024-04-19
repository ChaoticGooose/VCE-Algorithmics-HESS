# VSV VCE Algorithmics SAT U3 #
Task 1: **Radius Repellent**  
<sub>(The sanitation team must traverse every road within a given radius and spray it with Pangobat repellent in the shortest possible time.)</sub>  
  
Task 2: **Radius Vaccination**  
<sub>The medical team must visit every town within a given radius and vaccinate everybody in the shortest possible time.</sub>  

### Table of Contents ###
1. [Model and Specification](#part-1-model-and-specification)
2. [Design your Algorithm](#part-2-design-your-algorithm)
3. [Implementation](#part-3-implementation)
4. [Evaluation](#part-4-evaluation)

## Part 1: **Model and Specification** ##
### 1 Radius Repellent ###
In urban environments, the Sanitation Team faces the task of ffectively traversing every road to spray Pangobat repellent, ensuring the area's pest control. This real-world problem demands an algorithmic solution that optimizes time efficiency while covering all roads within a given radius, as to minimise community impact because of extended road closures, resource consumption and further spread. To model this effectively, a Graph ADT will be used to represent the road network, with each road as an edge and each Town as a Node. Each node includes attributes such as location coordinates (latitude and longitude), population density, income level (reflecting road importance, as poorer areas may be more vaunrable). The Edges, representing roads, contain data like distance and time required for traversal.

To navigate this graph efficiently, we utilize a Priority Queue ADT, prioritizing roads based on the shortest time required for spraying Pangobat as well as average income in each town. This will be combined with a set containing all visited roads, ensuring no road is sprayed more than once.


### 2. Radius Vaccination ### 
This second task requires the Medical Team to visit every town within a given radius and vaccinate all residents. The problem is similar to the first, but with different constraints and objectives. This problem is of paramount importance in scenarios like mass vaccination campaigns, where timely and comprehensive coverage can significantly impact public health outcomes, similar to what was seen in the COVID pandemic. To model this real-world problem effectively, several key features need consideration.

As with the first task, a Graph ADT will be used to represent the town network, with each town as a Node and each road as an Edge. Each node will include attributes like location coordinates, population density, average income and average age. The edges will contain data like distance and time required for traversal. This information is vital for route planning and resource allocation and provides the basis for optimising routes abd minimising travel time. Furthermore, the algorithm must account for the vaccination process itself. A priority queue ADT is introduced to manage the order of towns to be visited based on factors like population density, urgency of vaccination (derived from median age and income), and travel distances. This ensures that high-priority locations are addressed promptly, optimizing the overall vaccination schedule

### 3. ADT Signitures ###
- Priority Queue
    - **enqueue**: element, int -> Priority Queue
    - **dequeue**: Priority Queue -> element
    - **isEmpty**: Priority Queue -> bool
- Set
    - **add**: element, Set -> Set
    - **remove**: element, Set -> Set
    - **contains**: element, Set -> bool
    - **length**: Set -> int

## Part 2: **Design your Algorithm** ##

## Part 3: **Implementation** ##

## Part 4: **Evaluation** ##
