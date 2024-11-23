## Clustering and Graph Analysis Toolkit

This project provides tools for clustering graph-based data, generating datasets, and visualizing results. Below is a detailed breakdown of the modules, supported algorithms, data representations, and visualization capabilities.

## Modules

### settlement_clustering

**Purpose:** Main module for clustering graphs.

**Features:**
* Reads input files and writes output files.
* Divides graphs into clusters using various algorithms.
* Visualizes graphs and clusters.
* Handles user interaction.

### data_set_generator

**Purpose:** Generates datasets within specified boundaries.

**Features:**
* Creates random or structured data suitable for clustering tasks.
* Provides flexibility in defining dataset parameters.

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
and paints every node accordingly to its cluster. If nothing is provided, paints every node to blue.

![Graph visualization](https://i.imgur.com/iVqu0pJ.jpeg)
