"""
A module for clustering settlements into administrative units.
"""

from pyvis.network import Network


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


def assign_color_to_clusters(clusters: list[dict]) -> None:
    """
    Works in-place. Assigns a random color to every cluster of a parameter clusters.

    :param clusters: list, A list of clusters to assign colors to.
    :return: None
    """
    for cluster in clusters:
        cluster["color"] = "#{:06x}".format(random.randint(0, 0xFFFFFF))


def assign_color_to_nodes(clusters: list[dict]) -> dict[str, str]:
    """

    :param clusters: list, A list of clusters (dictionaries) with assigned color to each one.
    :return: dict, A dictionary of every node as key and its color as value.
    """
    assigned_nodes = {}

    for cluster in clusters:
        for node in cluster["nodes"]:
            assigned_nodes[node] = cluster["color"]

    return assigned_nodes


def visualize(graph: dict[str, dict[str, float]], colors: dict[str, str] = None) -> None:
    """
    Visualizes a weighted graph using the pyvis library. Optional parameter colors: if not None,
    paints each node to its color. If you wish to call the function with this parameter, please
    assign colors to your clusters list using the assign_color_to_clusters function and then
    call this function like this:

    visualize(graph, assign_color_to_nodes(your_clusters))

    :param graph: dict, The graph to visualize.
    :param colors: dict, A dictionary of every node as key and its color as value.
    :return: None
    """
    net = Network(notebook=True)

    added_nodes = []
    for node, edges in graph.items():
        net.add_node(node, color=colors[node] if colors else "blue")
        added_nodes.append(node)
        for neighbour, distance in edges.items():
            if neighbour in added_nodes:
                net.add_edge(node, neighbour, width=distance, length=distance)

    net.force_atlas_2based()

    net.show("graph.html")
