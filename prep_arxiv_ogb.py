import torch
from torch_geometric.utils import subgraph
import os
import random
import pickle
from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm  # Import tqdm for the progress bar

# Function to get subgraph by year
def get_subgraph_by_year(data, target_year):
    years = data.node_year.squeeze()
    node_mask = (years == target_year)
    selected_nodes = node_mask.nonzero(as_tuple=True)[0]

    if selected_nodes.numel() == 0:
        raise ValueError(f"No nodes found for year {target_year}.")

    # Relabel nodes in the edge_index to range 0..len(selected_nodes)-1
    sub_edge_index, _ = subgraph(selected_nodes, data.edge_index, relabel_nodes=True)

    # Build subgraph data object
    sub_data = data.clone()
    sub_data.edge_index = sub_edge_index
    sub_data.x = data.x[selected_nodes]
    sub_data.y = data.y[selected_nodes]
    sub_data.node_year = data.node_year[selected_nodes]
    sub_data.num_nodes = selected_nodes.size(0)

    return sub_data

# Load the OGBN-Arxiv dataset
dataset = PygNodePropPredDataset(name='ogbn-arxiv')
data = dataset[0]

# Output directory for preprocessed data
output_dir = "preprocessed_data/"
os.makedirs(output_dir, exist_ok=True)

# Sampling 200 subgraphs per class with 100 nodes each
def sample_graphs_for_binary_classification(data, target_year, num_graphs_per_class=200, nodes_per_graph=100):
    graphs = []
    labels = []
    
    # Get the subgraph for the target year
    sub_data1 = get_subgraph_by_year(data, target_year)
    sub_data2 = get_subgraph_by_year(data, target_year + 5)  # For the next 5 years
    
    # Filter for classes
    class_1_nodes = [i for i, label in enumerate(sub_data1.y)]
    class_2_nodes = [i for i, label in enumerate(sub_data2.y)]
    
    # Define binary labels
    class_1_label = 0  # Label earlier year as 0
    class_2_label = 1  # Label later year as 1
    
    # Sample from class 1
    for _ in range(num_graphs_per_class):
        sampled_nodes = random.sample(class_1_nodes, nodes_per_graph)
        sampled_nodes_tensor = torch.tensor(sampled_nodes)
        sampled_subgraph_data = sub_data1.subgraph(sampled_nodes_tensor)
        graphs.append(sampled_subgraph_data)
        labels.append(class_1_label)
    
    # Sample from class 2
    for _ in range(num_graphs_per_class):
        sampled_nodes = random.sample(class_2_nodes, nodes_per_graph)
        sampled_nodes_tensor = torch.tensor(sampled_nodes)
        sampled_subgraph_data = sub_data2.subgraph(sampled_nodes_tensor)
        graphs.append(sampled_subgraph_data)
        labels.append(class_2_label)
    
    return graphs, labels

# Pre-process and save training, validation, and test data (yearly splits for test data)
def pre_process_and_save(data):
    # For training and validation, we will sample from 2010
    print("Sampling training and validation graphs from 2010...")
    train_val_graphs, train_val_labels = sample_graphs_for_binary_classification(data, target_year=2010, num_graphs_per_class=200, nodes_per_graph=100)

    # Save training and validation data
    with open(os.path.join(output_dir, "train_val_graphs_2010_2015.pkl"), 'wb') as f:
        pickle.dump((train_val_graphs, train_val_labels), f)

    # For testing, we will sample from 2011 to 2014
    print("Sampling testing graphs from 2011 to 2014...")

    for year in range(2011, 2015):
        year_graphs, year_labels = sample_graphs_for_binary_classification(data, target_year=year, num_graphs_per_class=200, nodes_per_graph=100)

        # Save test data for each year
        with open(os.path.join(output_dir, f"test_graphs_{year}_{year+5}.pkl"), 'wb') as f:
            pickle.dump((year_graphs, year_labels), f)
    
    print("Data pre-processing and saving complete!")

# Pre-process and save the data
pre_process_and_save(data)
