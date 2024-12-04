# Clustering and Graph Analysis Toolkit

This project provides tools for clustering graph-based data, generating datasets, and visualizing results. Below is a 
detailed breakdown of the modules, supported algorithms, data representations, and visualization capabilities.

# How to use

### Introduction

The program contains several modules, but you only need to launch main.py using the command line. Note that the others 
are still necessary for the program to work correctly.

### Step 0: Prepare the data

This program works with weighted graphs recorded as .dot files. The file must have the following structure:

```
graph GRAPH_NAME {
    A -- B [distance=DISTANCE];
    B -- C [distance=DISTANCE];
    A -- C [distance=DISTANCE];
}
```

Where A, B and C are the names of your nodes, GRAPH_NAME is the name of your graph and DISTANCE is the respective 
distance between the two specified nodes (a weighted edge). Not every node must have a connection with every other, 
but it is required that an alternative path exists for reaching each one. That effectively means that your graph must be 
connected.

If you do not have a file to work with, there is an option to use the `dataset_generator.py` module. Please refer to **Step 4**.

### Step 1 : Opening the Terminal

To get started, open Terminal in the folder containing the program. For Windows, you can achieve this by using the cd command or 
by opening the folder in File Explorer and choosing the option "Open in Terminal" in the right-click menu.

### Step 2: Launching the program

Type the following command into the terminal: 
```commandline
python main.py "FILE.dot"
```
Where FILE.dot is the path to your 
weighted graph in the form of a .dot file. Executing this will apply the default clustering algorithm to your graph and 
print the result into console.

### Step 3: Advanced

For advanced clustering, the module `main.py` has several arguments you can specify when running the program. For a 
detailed list of these arguments, type `python main.py --help` or refer to the following instructions:

To run the program with an argument, type `python main.py "FILE.dot"` as usual and then the arguments you want to use. 
Most of the arguments require a value to be passed to them, so be sure to specify it after the argument's name.

Running the program to divide a graph into 4 clusters might look like this:

```commandline
python main.py "graph.dot" -n 4
```

**List of optional arguments:**

* **-a** : Specifies algorithm. Accepts 1 or 2: 1 for k-medoids algorithm and 2 for Louvain method.
* **-n** : Specifies the number of clusters to divide the graph into. Accepts int.
* **-s** : Specifies the random seed to use for clustering. Use when you want to get a specific non-changing result. Accepts int. 
* **-w** : Specifies the path to a file to write the clustering results into. Does not create a file when left blank. Accepts str.
* **-v** : Use to visualize the result in your web browser. Does not require a value.

### Step 4: Using `dataset_generator.py`

If you wish to generate a file to work with for testing or other purposes, you may use the `dataset_generator.py` module. 
Its purpose is to generates highly customizable weighted graphs you can use in the main module right away.

The usage is similar to the `main.py`. (Please read **Step 3** if not familiar with arguments). Launch the module like this:

```commandline
python dataset_generator.py "FILE.dot"
```

Where FILE.dot is the path to a file you want to record the graph into. The file may not exist - the program will create it.

**List of optional arguments:**

* **-a** : Specifies the number of nodes in the graph. Accepts int.
* **--min** : Specifies the min distance between two nodes. Accepts float.
* **--max** : Specifies the max distance between two nodes. Accepts float. 
* **-e** : Specifies the probability for an extra edge to occur between two nodes. Accepts a value between [0, 1].
* **-s** : Specifies the random seed to use for generating. Use when you want to get a specific non-changing result. Accepts int.
* **-l** : Use to assign random text labels to nodes. Does not require a value.

Running the generator to create a graph containing 100 nodes might look like this:

```commandline
python main.py "graph.dot" -a 100
```

# Documenation

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