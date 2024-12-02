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
        >>> dijkstra_no_heapq(graph_dict, '0') == expected
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

def compute_distance_matrix(graph:dict[str, dict[str, float]]) -> list[list]:
    """
    Compute the pairwise distance matrix using Dijkstra's algorithm.

    Args:
        graph (dict): Adjacency dictionary representing the graph.

    Returns:
        tuple: A tuple containing:
               - List of nodes in sorted order.
               - 2D NumPy array representing the pairwise distance matrix.

    Example:
        >>> graph_dict = {
        ...     '0': {'4': 4.94, '3': 4.83},
        ...     '1': {'4': 9.68},
        ...     '2': {'4': 4.72, '3': 8.66},
        ...     '3': {'2': 8.66, '0': 4.83},
        ...     '4': {'0': 4.94, '2': 4.72, '1': 9.68},
        ... }
        >>> nodes, matrix = compute_distance_matrix(graph_dict)
        >>> nodes
        ['0', '1', '2', '3', '4']
        >>> matrix
        array([[ 0.  , 14.62,  9.66,  4.83,  4.94],
               [14.62,  0.  , 14.4 , 19.45,  9.68],
               [ 9.66, 14.4 ,  0.  ,  8.66,  4.72],
               [ 4.83, 19.45,  8.66,  0.  ,  9.77],
               [ 4.94,  9.68,  4.72,  9.77,  0.  ]])
        >>> graph_dict = {
        ...     'Lviv': {'Rivne': 4.94, 'Dnipro': 4.83},
        ...     'Kyiv': {'Rivne': 9.68},
        ...     'Kharkiv': {'Rivne': 4.72, 'Dnipro': 8.66},
        ...     'Dnipro': {'Kharkiv': 8.66, 'Lviv': 4.83},
        ...     'Rivne': {'Lviv': 4.94, 'Kharkiv': 4.72, 'Kyiv': 9.68},
        ... }
        >>> nodes, matrix = compute_distance_matrix(graph_dict)
        >>> nodes
        ['Dnipro', 'Kharkiv', 'Kyiv', 'Lviv', 'Rivne']
        >>> matrix
        array([[ 0.  ,  8.66, 19.45,  4.83,  9.77],
               [ 8.66,  0.  , 14.4 ,  9.66,  4.72],
               [19.45, 14.4 ,  0.  , 14.62,  9.68],
               [ 4.83,  9.66, 14.62,  0.  ,  4.94],
               [ 9.77,  4.72,  9.68,  4.94,  0.  ]])
    """
    nodes = sorted(graph.keys())
    num_nodes = len(nodes)
    node_indices = {node: i for i, node in enumerate(nodes)}

    distance_matrix = np.zeros((num_nodes, num_nodes))

    for node in nodes:
        distances = dijkstra(graph, node)
        for target_node, distance in distances.items():
            i, j = node_indices[node], node_indices[target_node]
            distance_matrix[i][j] = distance

    return nodes, np.round(distance_matrix, 2)

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
