"""
A module containing an implementation of the Louvain method for weighted graphs.
"""


from collections import defaultdict


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

    # Prepare clusters for visualization
    clusters = []
    for community in communities.values():
        clusters.append({'nodes': community})

    return clusters


if __name__ == "__main__":
    print("Please use the main.py module.")
