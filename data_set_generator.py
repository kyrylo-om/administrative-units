import random
from faker import Faker


def make_demo_graph(amount: int = 5, minimum: float = 0.1, maximum: float = 10.0,
                    extra_edges_prob: float = 0.0001, node_labels: bool = False, seed: int = None) -> dict[str, dict[str, float]]:
    """
    Generate a connected, undirected weighted graph with optional extra edges.

    Args:
        amount (int): The number of nodes the graph will have.
        minimum (float): The minimum value (distance) an edge can have.
        maximum (float): The maximum value (distance) an edge can have.
        extra_edges_prob (float): Probability of adding an additional edge beyond the minimum spanning tree.
        node_labels (bool): Determines whether nodes should have generated labels.
        If False, nodes will be represented as numbers, if True - as randomly generated strings.
        seed (int): A seed value for reproducibility. If None, randomness will vary on each run.

    Returns:
        dict: A dictionary where keys are node names (strings) and values are dictionaries
              representing edges and their weights.
    """
    if amount <= 0:
        raise ValueError("The number of nodes must be greater than 0.")
    if minimum < 0:
        raise ValueError("Minimum value must be greater than 0")
    if maximum < 0:
        raise ValueError("Minimum value must be greater than 0")
    if minimum >= maximum:
        raise ValueError("Minimum value must be less than the maximum value.")
    if not 0 <= extra_edges_prob <= 1:
        raise ValueError("Connectivity must be a value between 0 and 1.")
    if amount > 1800 and node_labels is True:
        raise ValueError("There are not enough unique cities' names.")

    # Set the random seed
    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)

    # Initialize graph
    if node_labels:
        fake = Faker('uk_UA')
        graph = {fake.unique.city(): {} for _ in range(amount)}
    else:
        graph = {f"{i}": {} for i in range(amount)}

    nodes = list(graph.keys())

    # Ensure graph is connected using a minimum spanning tree approach
    available_nodes = nodes
    connected_nodes = [nodes.pop(0)]  # Start with the first node

    while available_nodes:
        from_node = random.choice(connected_nodes)
        to_node = random.choice(available_nodes)

        # Create a random weight for the edge
        weight = round(random.uniform(minimum, maximum), 2)
        if from_node == to_node:
            continue
        # Add the edge
        graph[from_node][to_node] = weight
        graph[to_node][from_node] = weight

        # Move the node to the connected set
        connected_nodes.append(to_node)
        available_nodes.remove(to_node)

    # Add random extra edges based on probability
    for node_a in graph:
        for node_b in graph:
            if node_a == node_b or node_b in graph[node_a]:
                continue
            if random.random() <= extra_edges_prob:
                weight = round(random.uniform(minimum, maximum), 2)
                graph[node_a][node_b] = weight
                graph[node_b][node_a] = weight

    return graph


def convert_to_dot(graph: dict[str, dict[str, float]], filename: str) -> None:
    """
    Generate and save a Graphviz DOT representation of the given graph to a file.

    This function converts a graph (represented as an adjacency dictionary) into a DOT file format, 
    which can be used for visualizing the graph using tools like Graphviz.

    Args:
        graph (dict[str, dict[str, float]]): A dictionary representing the graph, where:
            - Keys are node names (strings).
            - Values are dictionaries mapping neighboring nodes to their edge weights.
        filename (str): The path to the file to write into.

    Side Effects:
        Creates a file named `demo_graph.dot` in the current working directory 
        containing the DOT representation of the graph.
    """
    with open(filename, 'w', encoding='UTF-8') as output:
        output.write("graph " + filename.capitalize()[:-4] + " {\n")

        written_edges = set()

        for node, edges in graph.items():
            for neighbor, weight in edges.items():
                edge = tuple(sorted((node, neighbor)))
                if edge not in written_edges:
                    output.write(f'    {edge[0]} -- {edge[1]} [distance={weight}];\n')
                    written_edges.add(edge)

        output.write("}\n")
