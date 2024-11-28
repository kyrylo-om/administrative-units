"""
A module for clustering settlements into administrative units.
"""
import time
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
    default_time_delay = 0.035
    def smooth_text(text:str,delay):
        """
        function that smooth_texts text in terminal not instantly but with small time sleep
        so that this text smooth_texting process can be beautiful
        """
        for i, char in enumerate(text):
            if i == len(text)-1:
                print(char,flush=True)
                time.sleep(delay)
            else:
                print(char,end="",flush=True)
                time.sleep(delay)
    smooth_text("Hello, my dear friend!!\nThis program can help if you want to do some clustering with your data",default_time_delay)
    smooth_text("Here is some main requirements:\nFile(with data to cluster) format should be .dot",default_time_delay)
    smooth_text("Also we need your data to be like this:\ngraph\nA — B [distance]",default_time_delay)
    smooth_text("B — C [distance]",default_time_delay)
    smooth_text("A — C [distance]",default_time_delay)
    smooth_text("Where 'A','B','C' - names of the nodes\ndistance - distance between those nodes",default_time_delay)
    smooth_text("write path to file here",default_time_delay)
    path_name = input("path to file:",)
    file_format = path_name[-3:]
    if file_format != "dot":
        if_wrong_format_text = ("Sorry but format of file doesn't match the requirements, please pay attention to format of it and try again",)
        return smooth_text(if_wrong_format_text,default_time_delay)
    else:
        check_file = check_file_content(path_name)
        if check_file != "Wrong file (return of that function)":
            smooth_text("Also please type number of clusters that you want to recieve. Number can be from 1 to 100. If it doesn't matter for you type \"0\"",default_time_delay)
            number_of_clusters = input("Please type the number of clusters and 0 if it doesn't matter:")
            if number_of_clusters == 0:
                result_of_clustering = dbscan(read_file(path_name))
                
            elif number_of_clusters > 1 and number_of_clusters < 100:
                result_of_clustering = kmedoids_clustering(read_file(path_name), number_of_clusters)
        else:
            if_wrong_file_text = "Sorry but your file does not match requirements for clustering, you may choose another file and try again"
            return smooth_text(if_wrong_file_text,default_time_delay)


def visualize(clusters: list):
    """
    Visualizes a weighted graph using the pyvis library.

    :param clusters: list, The nodes divided to clusters.
    :return: None
    """
    pass
