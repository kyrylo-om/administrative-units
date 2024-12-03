
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
-----------------------------------------------------------
                Interaction with user
All scenario of usage of our product contains in command_line function.
At the beginning of it is smooth_text function which was made so that all
text in the terminal could appear smoothly which enhance usage
After that using that function we tell user what is our program about 
and tell him all requirements to file. If something is wrong(format of file, there is
no graph inside etc.) we tell user about the problem and give opportunity 
to try again. Also user whenever user has to type something he could type "quit" to exit program
If everything is all right at the end we give user choice to see the result in terminal or in browser
