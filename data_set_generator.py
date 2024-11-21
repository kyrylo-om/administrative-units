'''Module to generate data sets'''
import random

def generator(amount: int=5, minimum: float=0.1, maximum: float=10.0, extra_edges_prob: float = 0.2) -> dict[str, dict[str, float]]:
    """
    Generate a connected, undirected weighted graph with optional extra edges.

    Args:
        amount (int): The number of nodes the graph will have.
        minimum (float): The minimum value (distance) an edge can have.
        maximum (float): The maximum value (distance) an edge can have.
        extra_edges_prob (float): Probability of adding an additional edge beyond the minimum spanning tree.

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
    if not (0 <= extra_edges_prob <= 1):
        raise ValueError("Connectivity must be a value between 0 and 1.")

    # Initialize graph
    graph = {f"{i}": {} for i in range(amount)}
    nodes = list(graph.keys())

    # Step 1: Ensure graph is connected using a minimum spanning tree approach
    available_nodes = set(nodes)
    connected_nodes = {nodes.pop(0)}  # Start with the first node

    while available_nodes:
        from_node = random.choice(list(connected_nodes))
        to_node = random.choice(list(available_nodes))

        # Create a random weight for the edge
        weight = round(random.uniform(minimum, maximum), 2)

        # Add the edge
        graph[from_node][to_node] = weight
        graph[to_node][from_node] = weight

        # Move the node to the connected set
        connected_nodes.add(to_node)
        available_nodes.remove(to_node)

    # Step 2: Add random extra edges based on probability
    for i in range(amount):
        for j in range(i + 1, amount):
            node_a, node_b = f"{i}", f"{j}"
            if random.random() <= extra_edges_prob and node_b not in graph[node_a]:
                weight = round(random.uniform(minimum, maximum), 2)
                graph[node_a][node_b] = weight
                graph[node_b][node_a] = weight

    return graph

print(generator())