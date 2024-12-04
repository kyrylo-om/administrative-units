"""
A module containing the utility functions for the main algorithms.
"""


import numpy as np


def dijkstra(graph: dict[str, dict[str, float]], start_node: str) -> dict[str, float]:
    """
    Compute the shortest paths from the start_node to all other nodes using Dijkstra's algorithm.

    Args:
        graph (dict): Adjacency dictionary representing the graph.
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
                    distances[neighbor] = round(new_distance, 2)

    return distances


def compute_distance_matrix(graph: dict[str, dict[str, float]]) -> tuple:
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


if __name__ == "__main__":
    print("Please use the main.py module.")
