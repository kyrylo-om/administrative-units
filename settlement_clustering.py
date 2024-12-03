"""
A module for clustering settlements into administrative units.
"""
import time
import os
import webbrowser
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
            result_of_clustering == louvain_algorithm(graph)
        elif alg_choice == "n":
            alg = 0
            alg = random.randint(1,2)
            if alg == 1:
                result_of_clustering = kmedoids_clustering(graph)
            else:
                result_of_clustering == louvain_algorithm(graph)
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



print(command_line_interface())


def visualize(clusters: list):
    """
    Visualizes a weighted graph using the pyvis library.

    :param clusters: list, The nodes divided to clusters.
    :return: None
    """
    pass
