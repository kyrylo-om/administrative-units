"""
A module for clustering settlements into administrative units.
"""


def read_file(file_name: str) -> dict[str, dict[str, float]]:
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


def dbscan(
    graph: dict[str, dict[str, float]], eps: float, min_points: int
) -> list[dict[str, dict[str, float]]]:
    """
    An algorithm for clustering without a predetermined number of clusters - DBSCAN.

    :param graph: dict, The graph of nodes.
    :param eps: float, The maximum distance between two points for them to be considered neighbours.
    (from the same cluster)
    :param min_points: int, The minimum number of points required to form a cluster.
    :return: list, The nodes divided to clusters (each cluster is an element of the list).
    """
    pass


def kmedoids_clustering(
    graph: dict[str, dict[str, float]], num_of_clusters: int
) -> list[dict[str, dict[str, float]]]:
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

