import numpy as np

import torch
import pickle
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import argparse

# Define the GCN model for graph classification
class GCNGraphClass(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNGraphClass, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.fc = torch.nn.Linear(out_channels, 1)  # Intermediate layer

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv(x, edge_index))
        x = global_mean_pool(x, batch)  # Pooling to obtain a graph-level embedding
        x = self.fc(x)
        return torch.sigmoid(x)  # Apply sigmoid for binary classification

# Function to evaluate the model on the validation set
def evaluate_model(model, val_graphs, val_labels):
    model.eval()
    
    all_preds = []
    all_labels = []
    all_losses = []

    loss_fn = torch.nn.BCELoss()

    with torch.no_grad():
        for graph, label in zip(val_graphs, val_labels):
            x, edge_index, label = graph.x.to(device), graph.edge_index.to(device), torch.tensor([label]).to(device)
            batch = graph.batch if graph.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=device)
            out = model(x, edge_index, batch)

            loss = loss_fn(out[0], label.float())

            all_preds.append(out)
            all_labels.append(label)
            all_losses.append(loss.item())

    all_preds = torch.stack(all_preds).cpu().squeeze()
    all_labels = torch.tensor(all_labels).cpu().numpy()
    # Get the predicted class (with the highest logit)
    # all_preds_class = all_preds.argmax(axis=1)
    all_preds_class = np.round(all_preds.numpy()).astype(int)  # Assuming binary classification with logits
    
    # Ensure that all_labels is in the correct format (flatten to a 1D array)
    accuracy = accuracy_score(all_labels, all_preds_class)
    return accuracy, np.mean(all_losses)

# Argument parsing for target_year (to loop through 2011 to 2020)
def parse_args():
    parser = argparse.ArgumentParser(description="Test GCN on OGBN-Arxiv Dataset (2011-2016, 2012-2017, 2013-2018, 2014-2019)")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Set device for testing (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model (the same architecture used during training)
    model = GCNGraphClass(in_channels=128, hidden_channels=16, out_channels=16).to(device)  # Adjust in_channels as per data

    # Load the best pre-trained model
    model.load_state_dict(torch.load("best_gcn_model.pth"))  # Load the best model saved during training
    model.eval()

    test_losses = []
    test_accuracies = []

    # Iterate through the years 2011 to 2014
    for year in range(2011, 2015):
        # Load the test data for the current year
        test_graphs_file = f'preprocessed_data/test_graphs_{year}_{year+5}.pkl'
        with open(test_graphs_file, 'rb') as f:
            test_graphs, test_labels = pickle.load(f)

        print(f"Testing on year {year}...")
        print(f"Test set size: {len(test_graphs)}")

        # Evaluate on the test set
        test_accuracy, test_loss = evaluate_model(model, test_graphs, test_labels)
        print(f"Test Accuracy for {year}: {test_accuracy:.4f}\tTest Loss for {year}: {test_loss:.4f}")

        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    np.save('test_accuracies.npy', np.array(test_accuracies))
    np.save('test_losses.npy', np.array(test_losses))

    print("Testing complete!")
