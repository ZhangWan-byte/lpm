# import torch
# from ogb.nodeproppred import PygNodePropPredDataset
# from torch_geometric.utils import subgraph
# import numpy as np
# import powerlaw
# import matplotlib.pyplot as plt
# import argparse

# def parse_args():
#     parser = argparse.ArgumentParser(description="Check Power-Law Fit for OGBN-Arxiv Degree Distribution")
#     parser.add_argument('--target_year', type=int, default=2015, help='The target year for the subgraph (default: 2015)')
#     return parser.parse_args()

# def get_subgraph_by_year(data, target_year):
#     years = data.node_year.squeeze()
#     node_mask = (years == target_year)
#     selected_nodes = node_mask.nonzero(as_tuple=True)[0]

#     if selected_nodes.numel() == 0:
#         raise ValueError(f"No nodes found for year {target_year}.")

#     # Relabel nodes in the edge_index to range 0..len(selected_nodes)-1
#     sub_edge_index, _ = subgraph(selected_nodes, data.edge_index, relabel_nodes=True)

#     # Create a subgraph data object
#     sub_data = data.clone()
#     sub_data.edge_index = sub_edge_index
#     sub_data.x = data.x[selected_nodes]
#     sub_data.y = data.y[selected_nodes]
#     sub_data.node_year = data.node_year[selected_nodes]
#     sub_data.num_nodes = selected_nodes.size(0)

#     return sub_data

# def compute_degrees(edge_index, num_nodes):
#     degrees = torch.zeros(num_nodes, dtype=torch.long)
#     for i in range(edge_index.size(1)):
#         degrees[edge_index[0, i]] += 1
#         degrees[edge_index[1, i]] += 1  # As the graph is undirected
#     return degrees

# def fit_power_law(degrees):
#     # Fit the degree distribution to a power-law using the powerlaw package
#     degree_values = degrees.numpy()
    
#     # We only care about degrees greater than 0
#     degree_values = degree_values[degree_values > 0]
    
#     # Fit a power law
#     fit = powerlaw.Fit(degree_values, discrete=True)
    
#     # Return the alpha exponent
#     return fit.alpha, fit.power_law

# def plot_degree_distribution(degrees):
#     degree_values = degrees.numpy()
#     degree_values = degree_values[degree_values > 0]
    
#     plt.hist(degree_values, bins=30, density=True, alpha=0.6, color='b', edgecolor='black')
#     plt.xlabel('Degree')
#     plt.ylabel('Density')
#     plt.title('Degree Distribution')
#     plt.show()

# if __name__ == "__main__":
#     # Parse arguments for target_year
#     args = parse_args()
#     target_year = args.target_year
#     print(f"Using target year: {target_year}")

#     # Load the ogbn-arxiv dataset
#     dataset = PygNodePropPredDataset(name='ogbn-arxiv')
#     data = dataset[0]

#     # Get subgraph for target year
#     sub_data = get_subgraph_by_year(data, target_year=target_year)

#     # Compute the degree distribution
#     degrees = compute_degrees(sub_data.edge_index, sub_data.num_nodes)

#     # Fit the degree distribution to a power-law
#     alpha, power_law = fit_power_law(degrees)

#     print(f"Power-law exponent (alpha): {alpha}")

#     # Plot the degree distribution
#     plot_degree_distribution(degrees)

#     # Check if the degree distribution fits a power-law
#     print(f"Does the degree distribution fit a power law? {power_law.is_power_law}")


import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import subgraph
import numpy as np
import powerlaw
import matplotlib.pyplot as plt
import argparse

# Argument parsing for target_year and variational distribution
def parse_args():
    parser = argparse.ArgumentParser(description="Check Power-Law Fit for OGBN-Arxiv Degree Distribution")
    parser.add_argument('--target_year', type=int, default=2015, help='The target year for the subgraph (default: 2015)')
    return parser.parse_args()

def get_subgraph_by_year(data, target_year):
    years = data.node_year.squeeze()
    node_mask = (years == target_year)
    selected_nodes = node_mask.nonzero(as_tuple=True)[0]

    if selected_nodes.numel() == 0:
        raise ValueError(f"No nodes found for year {target_year}.")

    # Relabel nodes in the edge_index to range 0..len(selected_nodes)-1
    sub_edge_index, _ = subgraph(selected_nodes, data.edge_index, relabel_nodes=True)

    # Create a subgraph data object
    sub_data = data.clone()
    sub_data.edge_index = sub_edge_index
    sub_data.x = data.x[selected_nodes]
    sub_data.y = data.y[selected_nodes]
    sub_data.node_year = data.node_year[selected_nodes]
    sub_data.num_nodes = selected_nodes.size(0)

    return sub_data

def compute_degrees(edge_index, num_nodes):
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    for i in range(edge_index.size(1)):
        degrees[edge_index[0, i]] += 1
        degrees[edge_index[1, i]] += 1  # As the graph is undirected
    return degrees

def fit_power_law(degrees):
    # Fit the degree distribution to a power-law using the powerlaw package
    degree_values = degrees.numpy()
    
    # We only care about degrees greater than 0
    degree_values = degree_values[degree_values > 0]
    
    # Fit a power law
    fit = powerlaw.Fit(degree_values, discrete=True)
    
    # Perform a distribution comparison (power-law vs exponential)
    log_likelihood_ratio, p_value = fit.distribution_compare('power_law', 'exponential')

    # Return the alpha exponent and the log-likelihood ratio result
    return fit.alpha, log_likelihood_ratio, p_value

def plot_degree_distribution_log_log(degrees):
    degree_values = degrees.numpy()
    degree_values = degree_values[degree_values > 0]
    
    # Compute the frequency of each degree
    unique_degrees, counts = np.unique(degree_values, return_counts=True)
    p_k = counts / len(degree_values)  # Normalize to get probabilities

    # Log-log plot
    plt.figure(figsize=(8, 6))
    plt.loglog(unique_degrees, p_k, marker='o', linestyle='None', color='b', alpha=0.7)
    plt.xlabel('Degree (k)')
    plt.ylabel('P(k)')
    plt.title('Degree Distribution (Log-Log scale)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Parse arguments for target_year
    args = parse_args()
    target_year = args.target_year
    print(f"Using target year: {target_year}")

    # Load the ogbn-arxiv dataset
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]

    # Get subgraph for target year
    sub_data = get_subgraph_by_year(data, target_year=target_year)

    # Compute the degree distribution
    degrees = compute_degrees(sub_data.edge_index, sub_data.num_nodes)

    # Fit the degree distribution to a power-law
    alpha, log_likelihood_ratio, p_value = fit_power_law(degrees)

    print(f"Power-law exponent (alpha): {alpha}")
    print(f"Log-likelihood ratio: {log_likelihood_ratio}")
    print(f"P-value for comparison: {p_value}")

    # Check if the degree distribution fits a power-law
    if p_value < 0.05:
        print("The degree distribution follows a power-law.")
    else:
        print("The degree distribution does NOT follow a power-law.")

    # Plot the degree distribution in log-log scale
    plot_degree_distribution_log_log(degrees)
