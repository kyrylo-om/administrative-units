"""
A module for clustering settlements into administrative units.
"""


import argparse
import random


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
        usage="settlement_clustering.py -f PATH [-n NUM_OF_CLUSTERS] [-s SEED] [-v]")
    parser.add_argument('-f', help="path to the file to read your graph from (required)",
                        metavar="PATH", type=str)
    # parser.add_argument('-a', metavar='ALGORITHM', type=int,
    #                     help="specifies the algorithm you want to use for clustering. "
    #                          "1 for k-medoids and 2 for Louvain method (-n must be void)",
    #                     default=None)
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

    if not args.file and not args.n and not args.visualize:
        command_line_interface()
    else:
        if not args.file:
            parser.error("the following argument is required: -f/--file")

        if args.s:
            random.seed(args.s)

        graph = read_file(args.file)

        if not args.n:
            clusters = dbscan(graph)
        else:
            if args.n < 0:
                parser.error("argument -n: number of clusters cannot be less than zero.")
            clusters = kmedoids_clustering(graph, args.n)

        if args.visualize:
            visualize(graph, clusters)
        else:
            pass
            # print_clusters(clusters)


if __name__ == "__main__":
    main()
