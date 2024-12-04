"""
A module containing the main clustering algorithms such as k-medoids and Louvain method.
"""


import numpy as np
import random
import utilities


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
            distances_sum += sum(d ** 2 for d in utilities.dijkstra(cluster_nodes, cluster['center']).values())

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
    nodes, distance_matrix = utilities.compute_distance_matrix(graph)
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


if __name__ == "__main__":
    print("Please use the main.py module.")
