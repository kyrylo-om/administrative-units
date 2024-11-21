
----------------------------------------------------------
                    Modules
settlement_clustering - main module for clustering graphs,
reads file, writes file, divide graphs into clusters with 
diffenrent algorithms, visualize the graph and handles
the interaction with user.
data_set_generator -  module for generating data_sets 
within wanted boundary.
----------------------------------------------------------
                    Algorithms
If given the amount of clusters:
- k-medoids
If not given the wanted amount of clusters:
- DBscan
-----------------------------------------------------------
            Graph interpritaion in Python
We will use dictionary to represent it in such way:
graph = {
    node1 : {node2 : distance1_2, node3 : distance1_3 }
    node2 : {node1 : distance1_2}
    node3 : {node1 : distance1_3}
}
Where distance is float, node is string.
-----------------------------------------------------------
                Clusters representation
Clusters will be represented by dictionary in such way:
clusters = {
    cluster1 : {node1, node22, node29},
    cluster2 : {node3, node33, node50},
    ...
}
Where key are cluster_indices that are integer,
and values are tuples of nodes that are in key cluster.
-----------------------------------------------------------
                Visualization
...

