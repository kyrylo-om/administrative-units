"""
A module for handling user interaction.
"""


from pyvis.network import Network
import argparse
import os
import random
import webbrowser
import k_medoids
import louvain
import utilities


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
    if not isinstance(graph, dict) or \
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

    distances = utilities.dijkstra(graph, node)
    if any(dist == float('inf') for dist in distances.values()):
        return 'The graph is not connected - some nodes cannot be reached'

    return True


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

    net = Network(notebook=True, cdn_resources="remote")

    added_nodes = []
    for node, edges in graph.items():
        title = f"Name: {node}" + (
            f"\nCluster: {node_clusters[node]}" if clusters else "") + f"\nConnections: {graph[node]}"
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
    """
    The main function which handles user interaction using argparse.
    Launches all other functions.

    :return: None
    """

    def print_clusters(clustered_graph: list):
        """
        Prints the clusters to the command line.

        :param clustered_graph: list, A list of nodes divided to clusters.
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

    parser = argparse.ArgumentParser(
        prog="main.py",
        description='A module for clustering weighted graphs.', )
    parser.add_argument('file', help="path to the file to read your graph from",
                        metavar="PATH_TO_FILE", type=str)
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
    parser.add_argument('-w', help="specify a file name to write the clustering results into",
                        metavar="FILENAME", type=str, default=None)
    parser.add_argument('-v', '--visualize', help="use to visualize the clustered graph",
                        default=None, action='store_true')

    args = parser.parse_args()

    if args.n is not None:
        if args.n < 0:
            parser.error("argument -n: number of clusters cannot be less than zero.")
        elif args.n == 0:
            parser.error("argument -n: number of clusters cannot be zero. "
                         "Use without this argument to determine the number of clusters automatically.")

    if args.a and args.a not in (1, 2):
        parser.error("argument -a: value must be 1 or 2")

    if args.a == 2 and args.n is not None:
        parser.error("argument -n must be void if the chosen algorithm is Louvain")

    print("\nReading file...")
    graph = read_file(args.file)

    if isinstance(graph, str):
        print(f"Reading failed: {graph}")
        exit()

    print("Reading successful.")

    print("\nValidating...")

    validating_result = validator(graph)
    if validating_result is not True:
        print(f"Validating failed: {validating_result}")
        exit()

    print("Validating successful.")

    if args.n is not None and args.n > len(graph):
        parser.error("argument -n: number of clusters cannot be greater than node count.")

    if args.s:
        random.seed(args.s)

    if args.n is None:
        if args.a == 1:
            print("\nFinding optimal cluster count...")
            optimal_cluster_count = k_medoids.find_optimal_cluster_count(graph)
            print(f"Optimal cluster count: {optimal_cluster_count}")
            print(f"\nRunning k-medoids algorithm for {optimal_cluster_count} clusters...")
            clusters = k_medoids.kmedoids_clustering(graph, optimal_cluster_count)
        else:
            print("\nRunning Louvain algorithm...")
            clusters = louvain.louvain_algorithm(graph)
    else:
        print(f"\nRunning k-medoids algorithm for {args.n} clusters...")
        clusters = k_medoids.kmedoids_clustering(graph, args.n)

    print("\n")
    text_clusters = print_clusters(clusters)
    print(text_clusters)

    if args.w is not None:
        with open(args.w, 'w', encoding="utf-8") as file:
            file.write(text_clusters)
        print(f"\nThe result has been recorded to {args.w}")

    if args.visualize:
        print("\nVisualizing result...")
        visualize(graph, clusters)
        print("The output HTML file has been recorded to graph.html")
        print("\nOpening in browser...")
        webbrowser.open(f"{os.getcwd()}\\graph.html")
        print("Done")


if __name__ == "__main__":
    main()
