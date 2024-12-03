# Clustering and Graph Analysis Toolkit

This project provides tools for clustering graph-based data, generating datasets, and visualizing results. Below is a 
detailed breakdown of the modules, supported algorithms, data representations, and visualization capabilities.


## Modules

### main.py

**Purpose:** Main module for handling interaction with the user.

**Features:**
* Reads input files and writes output files.
* Visualizes graphs and clusters.
* Handles user interaction.

**Functions:**
* `read_file()`: Reads a specified graph from file.
* `validator()`: Validates the content of the file so the graph is correct.
* `visualize()`: Visualizes the clustered graph using `pyvis` library.
* `main()`: Handles user interaction using `argparse`.

### utilities.py

**Purpose:** contains helper functions for the main algorithms.

**Features:**
* Calculates the shortest path from one node to every other one using Dijkstra's algorithm.
* Computes a distance matrix.

**Functions:**
* `dijkstra()`: An implementation of Dijkstra's algorithm.
* `compute_distance_matrix()`: Computes the pairwise distance matrix using Dijkstra's algorithm.

### k_medoids.py

**Purpose:** contains one of the two main clustering algorithms - k-medoids.

**Features:**
* Divides a graph to clusters using **k-medoids** algorithm.
* Defines an optimal number of clusters for a given graph.

**Functions:**
* `kmedoids_clustering()`: An implementation of k-medoids algorithm adapted for weighed graphs.
* `find_optimal_cluster_count()`: Finds the optimal number of clusters to use with k-medoids.

### louvain.py

**Purpose:** contains the other one of the two main clustering algorithms - Louvain method.

**Features:**
* Divides a graph to clusters using **Louvain method**.
* Calculates the modularity of a given graph.

**Functions:**
* `louvain_algorithm()`: An implementation of the Louvain method algorithm.
* `calculate_modularity()`: Returns the modularity of the given graph and its partition into communities.

### dataset_generator

**Purpose:** Generates datasets within specified boundaries.

**Features:**
* Creates random or structured data suitable for clustering tasks.
* Provides flexibility in defining dataset parameters.

**Functions:**
* `make_demo_graph()`: Generates connected, undirected and highly customizable weighted graphs.
* `convert_to_dot()`: Converts the generated graph to a .dot file and writes it to a specified directory.

## Algorithms

### When the number of clusters is provided:

**K-Medoids:** A robust clustering algorithm that minimizes the distance between points and their cluster centers.
Dijkstra algorithm is used to compute distance matrix.

### When the number of clusters is not specified:

**DBSCAN:** A density-based clustering algorithm that identifies clusters based on the density of data points.

## Data representation

### Graph representation

Graphs are represented as dictionaries with the following structure:

```
graph = {
    "node1": {"node2": distance1_2, "node3": distance1_3},
    "node2": {"node1": distance1_2},
    "node3": {"node1": distance1_3},
}
```

**Nodes:** Represented as strings.\
**Distances:** Represented as floats, greater than zero.

This structure allows for efficient storage and manipulation of graph data.
### Cluster Representation

Clusters are also represented as a list, structured as follows:

```
clusters = [
    {'nodes':{node1, node2, node 33}, center:node2}
    {'nodes':{node3, node55, node 32}, center:node3}
]
```

Each element is a dictionary, represents cluster,
which has a compulsory key "nodes" with value as a set with nodes that are
associated with this cluster, there could be additional keys depending on
the algorithm used, such as center, color etc.

This format simplifies cluster analysis and visualization.
## Visualization

The function `visualize()` works both for visualizing simple weighted graphs and also 
for visualizing weighted graphs divided by clusters. It takes an optional parameter `clusters`
and paints every node accordingly to its cluster. The center nodes of the cluster appear visually bigger.
If nothing is provided, paints every node to blue.

![Graph visualization](https://i.imgur.com/LQGiGsn.jpeg)

## Work distribution

### Luka Konovalov
Developed the following functionality:

* K-medoids algorithm. Function `kmedoids_clustering()`
* Louvain method. Function `louvain_algorithm()`
* Dijkstra's algorithm. Function `djikstra()`
* Dataset generator. Module `dataset_generator.py`

As well as:

* Developed a core concept, thought out the data representation and researched clustering algorithms.
* Wrote `README.md`.
* Created the presentation.

### Kyrylo Omelianchuk
Developed the following functionality:

* Graph visualization. Function `visualize()`
* Finding the optimal number of clusters to use in the k-medoids algorithm. Function `find_optimal_cluster_count()`
* All the user interaction in `main.py` and `dataset_generator.py`. Function `main()`

As well as:

* Developed a core concept, thought out the data representation and researched clustering algorithms.
* Maintained the repository, merged branches and divided code into different modules.
* Wrote `README.md`.

### Roman Pempus
Developed the following functionality:

* A function for user interaction `command_line_interface()`. Now deprecated.

As well as:

* Wrote down insights from every meeting.

### Mykhailo Brytan
Developed the following functionality:

* Validating the input file. Function `validator()`.

As well as:

* Helped with the presentation.

### Sofiia Pereima
Developed the following functionality:

* Reading the input file. Function `read_file()`.