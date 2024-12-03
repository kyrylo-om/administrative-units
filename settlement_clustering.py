"""
A module for clustering settlements into administrative units.
"""


import time
from collections import defaultdict
from pyvis.network import Network
import argparse
import os
import random
import numpy as np
import webbrowser


def read_file(file_name: str) -> dict[str, dict[str, float]] | str:
    """
    Reads the graph from file.

    The graph has the following structure:

    graph <name> {
        Node1 -- Node2 [distance=Value];
        ...
    }

    Args:
        file_name (str): Path to the `.dot` file.

    Returns:
        dict[str, dict[str, float]]: The graph as an adjacency list.

    Example:
    >>> example_file = "datasets/example.dot"
    >>> with open(example_file, "w") as f:
    ...     _ = f.write("graph example {\\n")
    ...     _ = f.write("   A -- B [distance=5.0];\\n")
    ...     _ = f.write("   B -- C [distance=7.2];\\n")
    ...     _ = f.write("   A -- C [distance=8.0];\\n")
    ...     _ = f.write("}\\n")
    >>> read_file(example_file)
    {'A': {'B': 5.0, 'C': 8.0}, 'B': {'A': 5.0, 'C': 7.2}, 'C': {'B': 7.2, 'A': 8.0}}

    >>> example_file = "datasets/invalid_format.dot"
    >>> with open(example_file, "w") as f:
    ...     _ = f.write("{\\n")
    ...     _ = f.write("   A -- B [distance=5.0];\\n")
    ...     _ = f.write("}\\n")
    >>> try:
    ...     read_file(example_file)
    ... except ValueError as e:
    ...     print(e)
    File must start with 'graph <name> {'.

    >>> example_file = "datasets/invalid_distance.dot"
    >>> with open(example_file, "w") as f:
    ...     _ = f.write("graph example {\\n")
    ...     _ = f.write("   A -- B []\\n")
    ...     _ = f.write("}\\n")
    >>> try:
    ...     read_file(example_file)
    ... except ValueError as e:
    ...     print(e)
    Incorrect line format: 'A -- B []'

    >>> try:
    ...     read_file("nonexistent.dot")
    ... except FileNotFoundError as e:
    ...     print(e)
    File 'nonexistent.dot' not found.
    """
    # Check file extension
    if not file_name.endswith(".dot"):
        return "File must have a .dot extension."

    # Try to open the file
    try:
        with open(file_name, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except FileNotFoundError:
        return f"File '{file_name}' not found."
    except IOError:
        return f"Error reading the file '{file_name}'."

    # Check the format of the first and last lines
    if not lines[0].strip().startswith("graph ") or not lines[0].strip().endswith("{"):
        return "File must start with 'graph <name> {'."
    if not lines[-1].strip() == "}":
        return "File must end with '}'."
    graph = {}

    # Process graph lines
    for line in lines[1:-1]:
        line = line.strip()
        if (
            not line
            or "--" not in line
            or "[distance=" not in line
            or not line.endswith("];")
        ):
            return f"Incorrect line format: '{line}'"

        # Parse the line
        try:
            parts = line.split("--")
            node1 = parts[0].strip()
            right_part = parts[1].strip()
            node2, distance_part = right_part.split("[distance=")
            node2 = node2.strip()
            distance = float(distance_part.rstrip("];").strip())
        except (ValueError, TypeError, IndexError) as e:
            return f"Error in line '{line}': {e}"
        # Add edges to the graph
        if node1 not in graph:
            graph[node1] = {}
        if node2 not in graph:
            graph[node2] = {}

        # The graph is undirected
        graph[node1][node2] = distance
        graph[node2][node1] = distance

    return graph


def validator(graph: dict[str, dict[str, float]]) -> bool | str:
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


def kmedoids_clustering(graph: dict[str, dict[str, float]],\
                     num_of_clusters: int, max_iter:int=100) -> list[dict[str, dict[str, float]]]:
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
    for _ in range(num_of_clusters-1):
        distance_to_nearest_medoid = [
            min(distance_matrix[node][medoid] for medoid in medoids)**2
            for node in range(n)
        ]
        probabilities = [d / sum(distance_to_nearest_medoid)
                        for d in distance_to_nearest_medoid]
        next_medoid = np.argmax(probabilities)
        medoids.append(next_medoid)

    for _ in range(max_iter):

        clusters = {medoid: [] for medoid in medoids}
        labels = [0]*n

        for i in range(n):
            medoids_distance=[distance_matrix[i][medoid] for medoid in medoids]
            closest_medoid = medoids[np.argmin(medoids_distance)]
            clusters[closest_medoid].append(i)
            labels[i] = closest_medoid

        new_medoids = []
        for cluster in clusters.values():
            total_distance=[
                sum(distance_matrix[node][other] for other in cluster)  for node in cluster
            ]
            new_medoids.append(cluster[np.argmin(total_distance)])

        if set(medoids) == set(new_medoids):
            break

        medoids = new_medoids

    output_clustering = []
    for medoid in clusters:
        cluster = {}
        cluster['center']=nodes[medoid]
        cluster['nodes'] = {nodes[index] for index,j in enumerate(labels) if j == medoid}
        output_clustering.append(cluster)

    return output_clustering


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
            distances_sum += sum(d**2 for d in dijkstra(cluster_nodes, cluster['center']).values())

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


def print_clusters(clustered_graph):
    """
    visualisation of clustering in terminal
    """
    result = ""
    result += "=" * 40
    result += "\nClustering Results:\n"
    result += "=" * 40
    num = 1
    for cluster in clustered_graph:
        if 'center' in cluster:
            centre = cluster['center']
            result += f"\n\nCluster {num} (central node: {centre})\n"
        else:
            result += f"\n\nCluster {num}:\n"
        nodes = cluster['nodes']
        for node in nodes:
            result += f"\n  - {node}"
        num += 1
        result += "\n\n"
        result += "=" * 40
        return result

def command_line_interface():
    """
    The function for handling interaction with the user. For example: how many clusters
    should there be in the result, or blank.

    Launches all other functions.

    :return: None
    """
    default_time_delay = 0.035
    def smooth_text(text:str,delay, ending = False): #function for smoothly appearing text
        """
        function that smooth_texts text in terminal not instantly but with small time sleep
        so that this text smooth_texting process can be beautiful
        """
        for i, char in enumerate(text):
            if i == len(text)-1:
                if ending == True:
                    print(char,flush=True, end = "")
                    time.sleep(delay)
            else:
                print(char,end="",flush=True)
                time.sleep(delay)
    #introduction of the program to user
    smooth_text("Hello, my dear friend!!\nThis program can help if you want to do some clustering with your data",default_time_delay)
    smooth_text("Here is some main requirements:\nFile(with data to cluster) format should be .dot",default_time_delay)
    smooth_text("Also we need your data to be like this:\ngraph\nA — B [distance]",default_time_delay)
    smooth_text("B — C [distance]",default_time_delay)
    smooth_text("A — C [distance]",default_time_delay)
    smooth_text("Where 'A','B','C' - names of the nodes\ndistance - distance between those nodes\nIn addition, whenever you want to quit - just type \"quit\"",default_time_delay)
    smooth_text("write path to file here",default_time_delay)
    #check file
    path_name = input("path to file:")
    graph = read_file(path_name)
    check_file_content = validator(graph)
    while isinstance(check_file_content, str):
        smooth_text("Please try again",default_time_delay)
        path_name = input("path to file:")
        if path_name == "quit":
            exit()
        graph = read_file(path_name)
        check_file_content = validator(graph)
        if check_file_content == "quit":
            exit()

    smooth_text("Also please type number of clusters that you want to recieve. Number can be from 1 to 100. If it doesn't matter for you type \"0\"",default_time_delay)
    number_of_clusters = input("Please type the number of clusters and 0 if it doesn't matter:")
    while number_of_clusters != 0 and number_of_clusters < 1 and number_of_clusters > 100 and number_of_clusters != "quit" :
        smooth_text("sorry but it seems you put invalid number of clusters, please try again",default_time_delay)
        number_of_clusters = input("Please type the number of clusters and 0 if it doesn't matter:")
    if number_of_clusters == "quit":
        exit()
    #choosing algorithm depending on number of clusters
    if number_of_clusters == 0:
        smooth_text("Also you may choose which algorithm to choose\n>>>>>>k_medoids\n>>>>>>louvain",default_time_delay)
        smooth_text("type \"k\" to use k_medoids, \"l\" to use louvain and \"n\" if it doesn't matter",default_time_delay)
        alg_choice = input(smooth_text("choose algorithm:",default_time_delay))
        while alg_choice != "l" and alg_choice != "n" and alg_choice != "k" and alg_choice != "quit":
                smooth_text("sorry but your input is invalid, choose the algorithm by typing one english letter: l,k,n or quit to exit")
                alg_choice = input(smooth_text("choose algorithm:",default_time_delay))
        if alg_choice == "k":
            result_of_clustering = kmedoids_clustering(graph)
        elif alg_choice == "l":
            result_of_clustering = louvain_algorithm(graph)
        elif alg_choice == "n":
            alg = 0
            alg = random.randint(1,2)
            if alg == 1:
                result_of_clustering = kmedoids_clustering(graph)
            else:
                result_of_clustering = louvain_algorithm(graph)
        elif alg_choice == "quit":
            exit()
    if number_of_clusters > 1 and number_of_clusters < 100:
        result_of_clustering = []
        result_of_clustering = kmedoids_clustering((graph), number_of_clusters)
    # visualisation of result
    smooth_text("Clustering is done! type \"term\" if you want to see the results in terminal and \"browser\"", default_time_delay)
    smooth_text("if you want to see the results using browser",default_time_delay)
    vis_choice = input(smooth_text("way of visualisation:",default_time_delay))
    while vis_choice != "browser" and vis_choice != "term":
        vis_choice = input(smooth_text("command not found, please choose the way to see the results:",default_time_delay))
    if vis_choice == "quit":
        exit()
    elif vis_choice == "term":
        smooth_text(print_clusters(result_of_clustering), default_time_delay)
    elif vis_choice == "browser":
        webbrowser.open(f"{os.getcwd()}\\graph.html")
    return smooth_text("Thank you for using our program! Have a great day!",default_time_delay)

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
