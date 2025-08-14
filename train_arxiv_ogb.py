import numpy as np
import torch
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import pickle
from sklearn.model_selection import train_test_split  # For data splitting

# Load the pre-processed training and validation graphs
with open('preprocessed_data/train_val_graphs_2010_2015.pkl', 'rb') as f:
    train_val_graphs, train_val_labels = pickle.load(f)

# Split the data into training and validation sets (80:20 split)
train_graphs, val_graphs, train_labels, val_labels = train_test_split(
    train_val_graphs, train_val_labels, test_size=0.2, random_state=42
)

print(f"Training set size: {len(train_graphs)}")
print(f"Validation set size: {len(val_graphs)}")

print(f"Graph size: {train_graphs[0].num_nodes} nodes, {train_graphs[0].num_edges} edges")

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
        return torch.sigmoid(x)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = GCNGraphClass(in_channels=128, hidden_channels=16, out_channels=16).to(device)  # Adjust in_channels as per data
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# loss_fn = torch.nn.CrossEntropyLoss()  # For multi-class classification
loss_fn = torch.nn.BCELoss()  # For binary classification

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

# Training loop
num_epochs = 10
best_val_loss = 100.0
best_model_state_dict = None

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    all_preds = []
    all_labels = []

    epoch_loss = 0.0

    for graph, label in zip(train_graphs, train_labels):

        # Extract the features and edge_index
        x, edge_index, label = graph.x, graph.edge_index, torch.tensor([label]).to(device)
        # Ensure the features and edge_index are on the same device as the model
        x = x.to(device)
        edge_index = edge_index.to(device)
        # Ensure the batch is defined (if using batch processing)
        batch = graph.batch if graph.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=device)

        # Forward pass
        optimizer.zero_grad()
        out = model(x, edge_index, batch)

        # Compute the loss (CrossEntropyLoss automatically handles labels as integers)
        # loss = loss_fn(out, label)  # Long type for multi-class labels
        loss = loss_fn(out[0], label.float())  # For binary classification with logits
        # print(out[0], label)

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
        all_preds.append(out)
        all_labels.append(label)

    epoch_loss /= len(train_graphs)

    # Calculate accuracy (or other metrics)
    all_preds_detached = torch.stack(all_preds).detach().cpu().squeeze().numpy()
    all_labels = torch.tensor(all_labels).cpu().numpy()

    # Get the predicted class (with the highest logit)
    # all_preds_class = np.argmax(all_preds_detached, axis=1)
    all_preds_class = np.round(all_preds_detached).astype(int)  # For binary classification
    
    # Ensure that all_labels is in the correct format (flatten to a 1D array)
    train_accuracy = accuracy_score(all_labels, all_preds_class)

    # Evaluate on validation set
    val_accuracy, val_loss = evaluate_model(model, val_graphs, val_labels)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Save the best model based on validation accuracy
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state_dict = model.state_dict()  # Save the best model state

# Save the trained model after training
torch.save(best_model_state_dict, "best_gcn_model.pth")
print("Model saved!")