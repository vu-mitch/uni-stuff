import networkx as nx
import itertools
import random
from math import factorial, log2, ceil
from tqdm import tqdm
from functools import lru_cache
import pickle

def graph_generator(n_nodes, density, weight_range=(1, 10), seed=2333):
    """Generate a random graph based on the density.

    Args:
        n_nodes (int): Number of nodes.
        density (float): Density of the graph (0-1).
        weight_range (tuple, optional): Range of edge weights. Defaults to (1, 10).
        seed (int, optional): Random seed for reproducibility. Defaults to 2333.

    Returns:
        nx.Graph: Generated graph.
    """
    random.seed(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))  
    max_edges = n_nodes * (n_nodes - 1) // 2
    n_edges = int(max_edges * density)

    # Create a list of all possible edges
    all_possible_edges = list(itertools.combinations(range(n_nodes), 2))
    # Shuffle and select the first n_edges
    random.shuffle(all_possible_edges)
    selected_edges = all_possible_edges[:n_edges]
    
    # Add the selected edges with weights
    for u, v in selected_edges:
        if weight_range is None:
            G.add_edge(u, v)
        else:
            weight = random.randint(*weight_range)
            G.add_edge(u, v, weight=weight)

    return G

def coalition_degree(G, S):
    """Calculate the characteristic function of a coalition in a graph.
    
    This function computes the sum of weighted degrees for nodes in coalition S.

    Args:
        G (nx.Graph): The graph.
        S (set/list): Set or list of nodes forming the coalition.

    Returns:
        float: The characteristic value of the coalition.
    """
    if not S:  # Handle empty coalition
        return 0
    
    subgraph = G.subgraph(S)
    return sum(dict(subgraph.degree(weight='weight')).values()) / 2

def shapley_value(G: nx.Graph, f=coalition_degree, verbose=False):
    """Calculate exact Shapley values for all nodes without sampling.

    Args:
        G (nx.Graph): Graph.
        f (function, optional): Characteristic function. Defaults to coalition_degree.
        verbose (bool, optional): Whether to show progress bar. Defaults to False.

    Returns:
        dict: Dictionary of Shapley values for each node.
    """
    # Ensure nodes are sorted for consistent ordering
    nodes = sorted(list(G.nodes()))
    n_nodes = len(nodes)
    shapley_values = {node: 0 for node in nodes}
    
    # Precompute factorials and coefficients to improve efficiency
    fact = [factorial(i) for i in range(n_nodes + 1)]
    
    coefficients = [
        (fact[s] * fact[n_nodes - s - 1]) / fact[n_nodes]
        for s in range(n_nodes)
    ]
    
    # Cache for function evaluations to avoid redundant calculations
    @lru_cache(maxsize=2**15)
    def cached_f(coalition_tuple):
        return f(G, set(coalition_tuple))
    
    # Process coalitions size by size to avoid storing all of them at once
    if verbose:
        # Set up a progress bar for all coalitions
        total_combinations = 2**n_nodes
        pbar = tqdm(total=total_combinations, desc="Processing coalitions")
    
    # Prepare nodes set for faster lookups
    nodes_set = set(nodes)
    
    # Process each coalition size separately to save memory
    for r in range(n_nodes + 1):
        # Generate combinations on-the-fly
        for coalition in itertools.combinations(nodes, r):
            if verbose:
                pbar.update(1)
                
            coalition_value = cached_f(coalition)
            
            # Get coefficient for this coalition size
            coeff = coefficients[r - 1] if r > 0 and r <= n_nodes else 0
            
            # Use set difference for nodes not in coalition
            coalition_set = set(coalition)
            remaining_nodes = nodes_set - coalition_set
            
            # For each node not in the coalition, compute its marginal contribution
            for node in remaining_nodes:
                new_coalition = tuple(sorted(coalition_set.union({node})))
                
                new_coalition_value = cached_f(new_coalition)
                
                # Calculate marginal contribution
                marginal_contribution = new_coalition_value - coalition_value
                
                # Update Shapley value
                shapley_values[node] += coeff * marginal_contribution
    
    if verbose:
        pbar.close()
    
    return shapley_values

def get_reachable_nodes_at_depth(G, node, depth):
    """Get all nodes at exactly k-hop distance from a given node.

    Args:
        G (nx.Graph): Graph.
        node: The source node.
        depth (int): Hop distance (k).

    Returns:
        set: Nodes that are exactly 'depth' hops away from the source node.
    """
    path_lengths = nx.single_source_shortest_path_length(G, node, cutoff=depth)
    return {n for n, d in path_lengths.items() if d == depth}

def cis(G, f=coalition_degree):
    """Calculate CIS-values for all nodes in a graph.

    Args:
        G (nx.Graph): The graph.
        f (function, optional): Characteristic function. Defaults to coalition_degree.

    Returns:
        dict: Dictionary of CIS-values for each node.
    """
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    grand_coalition_value = f(G, set(nodes))
    individual_values = {node: f(G, {node}) for node in nodes}
    total_individual_value = sum(individual_values.values())

    surplus = grand_coalition_value - total_individual_value
    equal_share = surplus / n_nodes if n_nodes > 0 else 0

    cis_values = {
        node: individual_values[node] + equal_share
        for node in nodes
    }
    return cis_values

def shapG(G: nx.Graph, f=coalition_degree, depth=1, m=15, approximate_by_ratio=True, scale=True, verbose=False):
    """Approximate Shapley values using local neighborhood sampling (ShapG algorithm).

    Args:
        G (nx.Graph): Graph.
        f (function, optional): Characteristic function. Defaults to coalition_degree.
        depth (int, optional): Maximum neighborhood depth. Defaults to 1.
        m (int, optional): Maximum number of reachable_nodes to sample. Defaults to 15.
        approximate_by_ratio (bool, optional): Whether to scale values by the ratio
            of the full coalition value to the sum of approximated values. Defaults to True.
        scale (bool, optional): Whether to apply scaling factor based on neighborhood size. Defaults to True.
        verbose (bool, optional): Whether to show progress information. Defaults to False.

    Returns:
        dict: Dictionary of approximated Shapley values for each node.
    """
    shapley_values = {node: 0 for node in G.nodes()}
    
    # Precompute full coalition value if we'll need it for scaling
    full_coalition_value = f(G, set(G.nodes())) if approximate_by_ratio else None
    
    # Use tqdm for progress tracking if verbose
    node_iterator = tqdm(G.nodes(), desc="Computing Shapley approximations") if verbose else G.nodes()
    
    # Cache for function evaluations
    @lru_cache(maxsize=2**15)
    def cached_f(coalition_tuple):
        return f(G, set(coalition_tuple))
    
    all_subsets = list()
    
    for node in node_iterator:
        # Collect all reachable_nodes within specified depth
        reachable_nodes_at_depth = set()
        for d in range(1, depth + 1):
            reachable_nodes_at_depth.update(get_reachable_nodes_at_depth(G, node, d))
        
        # Determine if we need sampling or can process the full neighborhood
        if len(reachable_nodes_at_depth) < m:
            # Small enough neighborhood - process all subsets
            reachable_nodes_at_depth.add(node)  # Add the node itself
            
            coeff = 1 / 2 ** (len(reachable_nodes_at_depth) - 1)            
            for S_size in range(len(reachable_nodes_at_depth)):
                for S in itertools.combinations(reachable_nodes_at_depth - {node}, S_size):
                    S_tuple = tuple(sorted(S))  # Sort for better cache performance
                    S_with_node_tuple = tuple(sorted(S + (node,)))
                    
                    all_subsets.append(S_with_node_tuple)# + ("smaller",))
                    
                    marginal_contribution = (
                        cached_f(S_with_node_tuple) - 
                        cached_f(S_tuple)
                    )
                    shapley_values[node] += marginal_contribution
            
            # Apply scaling factor
            shapley_values[node] *= coeff
        else:
            # Large neighborhood - use sampling
            # Determine number of samples based on neighborhood size
            # Eine Wahrscheinlichkeitsaufgabe in der Kundenwerbung Equation 18
            # sample_nums = ceil(((len(reachable_nodes_at_depth) + 1/2) / m - 1/2) * (log2(len(reachable_nodes_at_depth)) + 0.5772156649) + 1/2) # original formula
            sample_nums = ceil(len(reachable_nodes_at_depth) / m * (log2(len(reachable_nodes_at_depth)) + 0.5772156649))
            
            # Precompute coefficient outside of loops
            coeff = 1 / 2 ** (m) / sample_nums
            if scale:
                # Scale proportionally to the ratio of full neighborhood size to sample size
                coeff *= ((len(reachable_nodes_at_depth) + 1) / (m + 1))
            
            reachable_nodes_list = list(reachable_nodes_at_depth)  # Convert to list for sampling
            
            for _ in range(sample_nums):
                # Sample a subset of reachable_nodes
                reachable_nodes_sampled = set(random.sample(reachable_nodes_list, min(m, len(reachable_nodes_list))))
                reachable_nodes_sampled.add(node)  # Add the node itself
                
                for S_size in range(len(reachable_nodes_sampled)):
                    for S in itertools.combinations(reachable_nodes_sampled - {node}, S_size):
                        S_tuple = tuple(sorted(S))  # Sort for better cache performance
                        S_with_node_tuple = tuple(sorted(S + (node,)))
                        
                        all_subsets.append(S_with_node_tuple)# + ("bigger",))

                        marginal_contribution = (
                            cached_f(S_with_node_tuple) - 
                            cached_f(S_tuple)
                        )
                        shapley_values[node] += marginal_contribution
                #         all_subsets.append(("next subset",))
                #     all_subsets.append(("next size",))
                # all_subsets.append(("next sampling",))

            
            # Apply scaling factors
            shapley_values[node] *= coeff
        # all_subsets.append(("next node",))

    
    # Optional: scale all values to match the full coalition value
    if approximate_by_ratio:
        if full_coalition_value is None:  # If we didn't precompute it
            full_coalition_value = f(G, set(G.nodes()))
            
        approximate_sum = sum(shapley_values.values())
        
        if approximate_sum != 0:  # Avoid division by zero
            correction_factor = full_coalition_value / approximate_sum
            shapley_values = {k: v * correction_factor for k, v in shapley_values.items()}
    
    with open(r'C:\Users\admin\Desktop\master_ai\master\shapG-master\shapG\shapG_subsets.pkl', 'wb') as f:
        pickle.dump(all_subsets, f)
    print(all_subsets)
    return shapley_values