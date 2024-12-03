"""
A module for clustering settlements into administrative units.
"""


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
