"""
A module containing the algorithms such as k-medoids and Louvain method.
"""

import numpy as np
import random
from collections import defaultdict


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


def find_optimal_cluster_count(graph: dict[str, dict[str, float]]) -> int:
    """
    Finds the optimal number of clusters using the Elbow method.

    :param graph: dict, The graph to find optimal number of clusters for.
    :return: int, The optimal number of clusters.
    """
    wcss = []

    for i in range(1, len(graph) + 1):
        clusters = kmedoids_clustering(graph, i)
        distances_sum = 0
        for cluster in clusters:
            cluster_nodes = {key: value for key, value in graph.items() if key in cluster['nodes']}
            distances_sum += sum(d ** 2 for d in dijkstra(cluster_nodes, cluster['center']).values())

        wcss.append(distances_sum)

    median = sum(wcss) / len(wcss)

    optimal_count = 0
    min_diff = float('inf')
    for x in wcss:
        if abs(median - x) < min_diff:
            min_diff = abs(median - x)
            optimal_count += 1
        else:
            break

    return optimal_count


def kmedoids_clustering(graph: dict[str, dict[str, float]],
                        num_of_clusters: int, max_iter: int = 100) -> list[dict]:
    """
    An algorithm for clustering with a predetermined number of clusters - k-medoids clustering.

    :param graph: dict, The graph of nodes.
    :param num_of_clusters: int, The number of clusters the nodes have to be divided to.
    :param max_iter: int: Maximum number of iterations.
    :return: list, The nodes divided to clusters (each cluster is an element of the list).
    """
    nodes, distance_matrix = compute_distance_matrix(graph)
    n = len(nodes)

    # medoids = random.sample(range(n), num_of_clusters)

    medoids = [random.sample(range(n), 1)[0]]
    for _ in range(num_of_clusters - 1):
        distance_to_nearest_medoid = [
            min(distance_matrix[node][medoid] for medoid in medoids) ** 2
            for node in range(n)
        ]
        probabilities = [d / sum(distance_to_nearest_medoid)
                         for d in distance_to_nearest_medoid]
        next_medoid = np.argmax(probabilities)
        medoids.append(next_medoid)

    for _ in range(max_iter):

        clusters = {medoid: [] for medoid in medoids}
        labels = [0] * n

        for i in range(n):
            medoids_distance = [distance_matrix[i][medoid] for medoid in medoids]
            closest_medoid = medoids[np.argmin(medoids_distance)]
            clusters[closest_medoid].append(i)
            labels[i] = closest_medoid

        new_medoids = []
        for cluster in clusters.values():
            total_distance = [
                sum(distance_matrix[node][other] for other in cluster) for node in cluster
            ]
            new_medoids.append(cluster[np.argmin(total_distance)])

        if set(medoids) == set(new_medoids):
            break

        medoids = new_medoids

    output_clustering = []
    for medoid in clusters:
        cluster = {}
        cluster['center'] = nodes[medoid]
        cluster['nodes'] = {nodes[index] for index, j in enumerate(labels) if j == medoid}
        output_clustering.append(cluster)

    return output_clustering


def louvain_algorithm(graph: dict[str, dict[str, float]],
                      modularity_gain_threshold: float = 0.0001) -> list[dict[str, set[str]]]:
    """
    Perform community detection on a graph using the Louvain algorithm.

    The Louvain algorithm optimizes modularity in two phases:
    1. Phase 1: Nodes are iteratively moved between communities to maximize modularity.

    Args:
        graph (Dict[str, Dict[str, float]]): The input graph represented as an adjacency dictionary.
            Each node maps to a dictionary of its neighbors with edge weights.
        modularity_gain_threshold (float, optional): The minimum modularity gain required to move a
            node to another community. Default is 0.0001.

    Returns:
        List[Dict[str, Union[Set[str]]]]: A list of clusters, where each cluster is represented
            as a dictionary containing the set of nodes in that cluster.
    """

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

    communities = {node: {node} for node in graph}
    node_to_community = {node: node for node in graph}
    total_weight = sum(sum(edges.values()) for edges in graph.values()) / 2

    while True:
        improvement = False  # Reset improvement flag at the start of each iteration
        initial_modularity = calculate_modularity(graph, communities, total_weight)

        for node in graph:
            current_community = node_to_community[node]
            best_community = current_community
            best_modularity_gain = 0

            # Temporarily remove the node from its community
            communities[current_community].remove(node)
            if not communities[current_community]:
                del communities[current_community]

            # Check modularity gain for moving the node to neighboring communities
            neighbor_communities = defaultdict(float)
            for neighbor, weight in graph[node].items():
                neighbor_community = node_to_community[neighbor]
                neighbor_communities[neighbor_community] += weight

            for neighbor_community, edge_weight in neighbor_communities.items():
                if neighbor_community not in communities:
                    continue
                community_degree = sum(sum(graph[u].values()) for u in communities[neighbor_community])
                node_degree = sum(graph[node].values())
                modularity_gain = edge_weight / total_weight - (community_degree * node_degree) / (
                            2 * total_weight ** 2)

                if modularity_gain > best_modularity_gain:
                    best_modularity_gain = modularity_gain
                    best_community = neighbor_community

            # Move the node to the best community if the modularity gain is significant
            if best_modularity_gain > modularity_gain_threshold:
                improvement = True
                communities.setdefault(best_community, set()).add(node)
                node_to_community[node] = best_community
            else:
                # Return the node to its original community
                communities.setdefault(current_community, set()).add(node)
                node_to_community[node] = current_community

        # Calculate new modularity and check for convergence
        new_modularity = calculate_modularity(graph, communities, total_weight)
        if not improvement or abs(new_modularity - initial_modularity) < modularity_gain_threshold:
            break

    # # Phase 2: Aggregate the graph
    # new_graph = defaultdict(lambda: defaultdict(float))
    # for community, nodes in communities.items():
    #     for u in nodes:
    #         for v, weight in graph[u].items():
    #             if node_to_community[v] == community:
    #                 new_graph[community][community] += weight / 2
    #             else:
    #                 new_graph[community][node_to_community[v]] += weight

    # Prepare clusters for visualization
    clusters = []
    for community in communities.values():
        clusters.append({'nodes': community})

    return clusters


if __name__ == "__main__":
    print("Please use the settlement_clustering module.")
