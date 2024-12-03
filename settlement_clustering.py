"""
A module for clustering settlements into administrative units.
"""


from collections import defaultdict
from pyvis.network import Network
import argparse
import os
import random
import webbrowser


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
                modularity_gain = edge_weight / total_weight - (community_degree * node_degree) / (2 * total_weight ** 2)

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

    net.save_graph("graph.html")


def main():
    parser = argparse.ArgumentParser(
        prog="Settlement clustering",
        description='A module for clustering weighted graphs.',
        epilog="If you are unsure of what to do, run this script without arguments.",
        usage="settlement_clustering.py -f PATH [-a ALGORITHM] [-n NUM_OF_CLUSTERS] [-s SEED] [-v]")
    parser.add_argument('-f', help="path to the file to read your graph from (required)",
                        metavar="PATH", type=str)
    parser.add_argument('-a', metavar='ALGORITHM', type=int,
                        help="specifies the algorithm you want to use for clustering. "
                             "1 for k-medoids and 2 for Louvain method (-n must be void)",
                        default=None)
    parser.add_argument('-n', metavar='NUM_OF_CLUSTERS', type=int,
                        help="the number of clusters to divide your graph into, "
                             "if not specified will be determined automatically",
                        default=None)
    parser.add_argument('-s', help="the random seed to use for clustering, "
                                   "affects the choice of medoids in k-medoids algorithm",
                        default=None, type=int, metavar="SEED")
    parser.add_argument('-v', '--visualize', help="use to visualize the clustered graph",
                        default=None, action='store_true')

    args = parser.parse_args()

    if args.f is None and args.n is None and not args.visualize:
        command_line_interface()
    else:
        if not args.f:
            parser.error("the following argument is required: -f/--file")

        if args.n < 0:
            parser.error("argument -n: number of clusters cannot be less than zero.")
        elif args.n == 0:
            parser.error("argument -n: number of clusters cannot be zero. "
                         "Use without this argument to determine the number of clusters automatically.")

        if args.a and args.a not in (1, 2):
            parser.error("argument -a: value must be 1 or 2")

        if args.a == 2 and args.n is not None:
            parser.error("argument -n must be void if the chosen algorithm is Louvain")

        print("Reading file...\n")

        graph = read_file(args.f)
        print("Validating...\n")
        validating_result = validator(graph)
        if validating_result is not True:
            print(f"Validating failed: {validating_result}")

        if args.n > len(graph):
            parser.error("argument -n: number of clusters cannot be greater than node count.")

        if args.s:
            random.seed(args.s)

        if args.n is None:
            if args.a == 1:
                print("Finding optimal cluster count...")
                optimal_cluster_count = find_optimal_cluster_count(graph)
                print(f"Optimal cluster count: {optimal_cluster_count}\n")
                print("Running k-medoids algorithm...")
                clusters = kmedoids_clustering(graph, optimal_cluster_count)
            else:
                print("Running Louvain algorithm...\n")
                clusters = louvain_algorithm(graph)

        print("The result is:\n")
        # print_clusters(clusters)

        if args.visualize:
            print("Visualizing result...")
            visualize(graph, clusters)
            webbrowser.open(f"{os.getcwd()}\\graph.html")


if __name__ == "__main__":
    main()
