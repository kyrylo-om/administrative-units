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

def validator(graph: dict[str, dict[str, float]]) -> bool:
    '''
    Reads the graph, and check if it the length of the graph is non-negative,
    completely not isolated, and connected

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

    :param graph: dict, the graph gotten by read_file function.
    :return: bool, if allright -> True, else -> False.
    >>> validator({\
        "Lviv": {"Bruhovychi": 50, "Novoyavorivsk": 100},\
        "Bruhovychi": {"Lviv": 50},\
        "Novoyavorivsk": {"Lviv": 100}\
    })
    True
    >>> validator({\
        "Lviv": {"Bruhovychi": -50, "Novoyavorivsk": 100},\
        "Bruhovychi": {"Lviv": 50},\
        "Novoyavorivsk": {"Lviv": 100}\
    })
    False
    >>> validator({'Lviv': {'Bruhovychi': 50, 'Novoyavorivsk': 100},\
        'Novoyavorivsk': {'Lviv': 100}\
    })
    False
    '''
    for key, values in graph.items():
        for distance in values.values():
            if distance < 0:
                return False

    if graph and not all(graph.values()):
        return False

    visited = set()

    def in_down(node):
        if node in visited:
            return
        visited.add(node)
        for neighbour in graph.get(node, {}):
            in_down(neighbour)

    if graph:
        start_node = list(graph.keys())[0]
        in_down(start_node)
        if len(visited) != len(graph):
            return False

    return True

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


def visualize(clusters: list):
    """
    Visualizes a weighted graph using the pyvis library.

    :param clusters: list, The nodes divided to clusters.
    :return: None
    """
    pass

if __name__ == '__main__':
    import doctest
    print(doctest.testmod())
