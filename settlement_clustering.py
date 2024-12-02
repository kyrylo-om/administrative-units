"""
A module for clustering settlements into administrative units.
"""


def read_file(file_name: str) -> dict[str, dict[str, float]]:
    """
    Reads the graph from file.

    The graph has the following structure:
    {
        settlement_name: {neighbour_settlement: distance}
    }

    Example:
    {
        Lviv: {Bruhovychi: 50, Novoyavorivsk: 100},
        Bruhovychi: {Lviv: 50},
        Novoyavorivsk: {Lviv: 100}
    }


    :param file_name: str, The file to read the graph from.
    :return: dict, The graph gotten from file.
    """
    pass



def calculate_modularity(graph: dict[str, dict[str, float]],
                         communities: dict[str, set[str]],
                         total_weight: float) -> float:
    """
    Calculate the modularity of the given graph and its partition into communities.

    Modularity is a measure of the quality of a graph partition, where higher values indicate
    better-defined communities.

    Args:
        graph (Dict[str, Dict[str, float]]): The input graph represented as an adjacency dictionary.
            Each node maps to a dictionary of its neighbors with edge weights.
        communities (Dict[str, Set[str]]): A dictionary where each key represents a community
            and the value is the set of nodes in that community.
        total_weight (float): The total weight of all edges in the graph.

    Returns:
        float: The modularity of the partition.
    """
    modularity = 0
    for community in communities.values():
        in_degree = sum(graph[u][v] for u in community for v in community if v in graph[u])
        degree = sum(sum(graph[u].values()) for u in community)
        modularity += in_degree / (2 * total_weight) - (degree / (2 * total_weight)) ** 2
    return modularity




def kmedoids_clustering(graph: dict[str, dict[str, float]], num_of_clusters: int) -> list[dict[str, dict[str, float]]]:
    """
    An algorithm for clustering with a predetermined number of clusters - k-medoids clustering.

    :param graph: dict, The graph of nodes.
    :param num_of_clusters: int, The number of clusters the nodes have to be divided to.
    :return: list, The nodes divided to clusters (each cluster is an element of the list).
    """
    pass


def command_line_interface():
    """
    The function for handling interaction with the user. For example: how many clusters
    should there be in the result, or blank.

    Launches all other functions.

    :return: None
    """
    pass


def visualize(clusters: list):
    """
    Visualizes a weighted graph using the pyvis library.

    :param clusters: list, The nodes divided to clusters.
    :return: None
    """
    pass
