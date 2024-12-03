"""
A module for clustering settlements into administrative units.
"""
import numpy as np

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

def dijkstra(graph:dict[str, dict[str, float]], start_node:str) ->dict[str,float]:
    """
    Compute shortest paths from the start_node to all other nodes using Dijkstra's algorithm.

    Args:
        graph_dict (dict): Adjacency dictionary representing the graph.
        start_node (str): The starting node for Dijkstra's algorithm.

    Returns:
        dict: Shortest distances from start_node to all other nodes.

    Example:
        >>> graph_dict = {
        ...     '0': {'4': 4.94, '3': 4.83},
        ...     '1': {'4': 9.68},
        ...     '2': {'4': 4.72, '3': 8.66},
        ...     '3': {'2': 8.66, '0': 4.83},
        ...     '4': {'0': 4.94, '2': 4.72, '1': 9.68},
        ... }
        >>> expected = {'0': 0, '1': 14.62, '2': 9.66, '3': 4.83, '4': 4.94}
        >>> dijkstra(graph_dict, '0') == expected
        True
    """
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    unvisited = set(graph.keys())

    while unvisited:
        current_node = min(unvisited, key=lambda node: distances[node])

        if distances[current_node] == float('inf'):
            break

        unvisited.remove(current_node)

        for neighbor, weight in graph[current_node].items():
            if neighbor in unvisited:
                new_distance = distances[current_node] + weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = round(new_distance,2)

    return distances

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
    'Some of values have negative lenth or == 0'
    >>> validator({\
        'Lviv': {'Bruhovychi': 50, 'Novoyavorivsk': 100},\
        'Novoyavorivsk': {'Lviv': 100}\
    })
    'Graph is isolated'
    >>> validator({\
        'Lviv': {'Bruhovychi': 50, 'Novoyavorivsk': 100},\
        'Novoyavorivsk': {'Lviv': 90},\
        'Bruhovychi': {'Lviv': 50}\
    })
    'The graph must be symmetric'
    >>> validator({\
        'Lviv': {},\
        'Novoyavorivsk': {'Lviv': 90},\
        'Bruhovychi': {'Lviv': 50}\
    })
    'Graph is isolated'
    >>> validator({\
        'Lviv': {'Bruhovychi': 50, 'Novoyavorivsk': 100},\
        "Bruhovychi": {"Lviv": 50, "Hutir": 100},\
        "Novoyavorivsk": {"Lviv": 100},\
        "Hutir": {"Bruhovychi": 100, "Donetsk": 1000},\
        "Donetsk": {}\
    })
    'Graph is isolated'
    >>> validator({})
    'Graph is empty'
    >>> validator({\
    "Lviv": {"Bruhovychi": 50, "Novoyavorivsk": 100},\
    "Bruhovychi": {"Lviv": 50, "Novoyavorivsk": 60},\
    "Novoyavorivsk": {"Lviv": 100, "Bruhovychi": 60},\
    "Hutir": {"Donetsk": 1000},\
    "Donetsk": {"Hutir": 1000}\
    })
    'The graph is not connected - some nodes cannot be reached'
    '''
    if not isinstance(graph, dict) or\
    any(not isinstance(neighbors, dict) for neighbors in graph.values()):
        return 'Graph is not dict type or values in graph is not dict type'

    if graph == {}:
        return 'Graph is empty'

    for key, values in graph.items():
        for distance in values.values():
            if distance <= 0:
                return 'Some of values have negative lenth or == 0'

    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors.keys())

    for node in all_nodes:
        if node not in graph or not graph.get(node, {}):
            return 'Graph is isolated'

    for node, connections in graph.items():
        for neighbor, distance in connections.items():
            if not graph or not distance or graph[neighbor].get(node) != distance:
                return 'The graph must be symmetric'

    distances = dijkstra(graph, node)
    if any(dist == float('inf') for dist in distances.values()):
        return 'The graph is not connected - some nodes cannot be reached'

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
