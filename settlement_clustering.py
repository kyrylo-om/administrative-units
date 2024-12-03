"""
A module for clustering settlements into administrative units.
"""

from pyvis.network import Network
import random


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


def dbscan(graph: dict[str, dict[str, float]], eps: float, min_points: int) -> list[dict[str, dict[str, float]]]:
    """
    An algorithm for clustering without a predetermined number of clusters - DBSCAN.

    :param graph: dict, The graph of nodes.
    :param eps: float, The maximum distance between two points for them to be considered neighbours.
    (from the same cluster)
    :param min_points: int, The minimum number of points required to form a cluster.
    :return: list, The nodes divided to clusters (each cluster is an element of the list).
    """
    pass


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


def visualize(graph: dict[str, dict[str, float]], clusters: list[dict] = None) -> None:
    """
    Visualizes a weighted graph using the pyvis library. Optional parameter clusters: if not None,
    paints each node to a color corresponding to its cluster.

    :param graph: dict, The graph to visualize.
    :param clusters: list, Optional. A list of clusters of given graph.
    :return: None
    """
    node_clusters = {}
    central_nodes = set()

    if clusters:
        for cluster in clusters:
            for node in cluster["nodes"]:
                node_clusters[node] = clusters.index(cluster) + 1
                if "center" in cluster and cluster["center"] == node:
                    central_nodes.add(node)

    net = Network(notebook=True)

    added_nodes = []
    for node, edges in graph.items():
        title = f"Name: {node}" + (f"\nCluster: {node_clusters[node]}" if clusters else "") + f"\nConnections: {graph[node]}"
        net.add_node(node, size=40 if node in central_nodes else 20, group=node_clusters[node] if clusters else None,
                     title=title)
        added_nodes.append(node)
        for neighbour, distance in edges.items():
            if neighbour in added_nodes:
                if clusters:
                    net.add_edge(node, neighbour, width=distance / 2,
                                 color="gray" if node_clusters[neighbour] != node_clusters[node] else None)
                else:
                    net.add_edge(node, neighbour, width=distance / 2)

    net.force_atlas_2based()

    net.show("graph.html")
