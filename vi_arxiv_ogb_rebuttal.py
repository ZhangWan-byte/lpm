import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import subgraph
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import argparse

# Argument parsing for target_year and variational distribution
def parse_args():
    parser = argparse.ArgumentParser(description="Latent Position Model Training")
    parser.add_argument('--target_year', type=int, default=2015, help='The target year for the subgraph (default: 2015)')
    parser.add_argument('--variational_distribution', type=str, default='Gaussian', choices=['Gaussian', 'Laplace', 'MoG', 'StudentT'],
                        help='Choose the variational distribution (default: Gaussian)')
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

class LatentPositionModel(torch.nn.Module):
    def __init__(self, N, d, variational_distribution='Gaussian', num_components=2):
        super(LatentPositionModel, self).__init__()
        self.N = N
        self.d = d
        self.variational_distribution = variational_distribution
        self.num_components = num_components
        
        # Initialize latent means and log std dev
        self.mu = torch.randn(N, d, requires_grad=True)  # Latent means (for latent positions)
        self.log_sigma = torch.randn(N, d, requires_grad=True)  # Log of std dev to ensure positivity
        
        # For MoG, we add the mixture components (weights and component means)
        if self.variational_distribution == 'MoG':
            self.mixture_weights = torch.nn.Parameter(torch.ones(self.num_components) / self.num_components, requires_grad=True)
            self.component_means = torch.randn(self.num_components, d, requires_grad=True)
            self.component_covariances = torch.nn.Parameter(torch.eye(d).unsqueeze(0).repeat(self.num_components, 1, 1), requires_grad=True)

    def get_variational_params(self):
        """Returns the variational parameters (latent positions and standard deviations)."""
        sigma = torch.exp(self.log_sigma)  # Convert log std dev to std dev
        return self.mu, sigma

    def likelihood(self, edge_index, mu, sigma):
        """Compute the log-likelihood for the observed directed edges."""
        edge_index = edge_index.to(mu.device)  # Move edge_index to the same device as mu
        
        pairwise_distances = torch.norm(mu[edge_index[0]] - mu[edge_index[1]], dim=1) ** 2
        p_ij = torch.sigmoid(-pairwise_distances)  # Apply sigmoid to the negative squared distances
        log_p = torch.sum(torch.log(p_ij))  # Sum log probabilities for all edges
        return log_p

    def kl_divergence(self, mu, sigma):
        """Compute the KL divergence between the variational distribution and the prior (Gaussian prior)."""
        if self.variational_distribution == 'Gaussian':
            # KL divergence between Gaussian variational distribution and Gaussian prior
            kl = 0.5 * torch.sum(sigma ** 2 + mu ** 2 - torch.log(sigma) - 1)  # Closed-form KL for Gaussian prior
        elif self.variational_distribution == 'Laplace':
            # KL divergence between Laplace and Gaussian
            kl = torch.sum(torch.log(2 * sigma) - sigma - 0.5)  # Closed-form KL for Laplace prior
        elif self.variational_distribution == 'MoG':
            # KL divergence between Mixture of Gaussians and Gaussian prior
            kl = 0
            for k in range(self.num_components):
                weight = self.mixture_weights[k]
                mean = self.component_means[k]
                cov = self.component_covariances[k]
                # Add terms related to the mixture components
                kl += weight * (torch.trace(torch.inverse(cov)) + torch.matmul(mean, torch.matmul(torch.inverse(cov), mean)) - self.d)
            return kl
        elif self.variational_distribution == 'StudentT':
            # KL divergence between Student t-distribution and Gaussian prior
            # A more complex calculation based on the properties of the t-distribution
            kl = torch.sum(torch.log(1 + mu**2 / (2 * sigma**2)))
            return kl

        return kl

    def elbo(self, edge_index):
        """Compute the negative Evidence Lower Bound (ELBO)."""
        mu, sigma = self.get_variational_params()
        log_likelihood = self.likelihood(edge_index, mu, sigma)
        kl = self.kl_divergence(mu, sigma)
        return - log_likelihood + kl

if __name__ == "__main__":
    # Parse arguments for target_year and variational distribution
    args = parse_args()
    target_year = args.target_year
    variational_distribution = args.variational_distribution
    print(f"Using target year: {target_year}")
    print(f"Using variational distribution: {variational_distribution}")

    # Set device for training (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories if they don't exist
    output_dir = 'output_rebuttal_vi'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the ogbn-arxiv dataset
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]

    # Get subgraph for target year
    sub_data = get_subgraph_by_year(data, target_year=target_year)

    print(f"Year {target_year} subgraph:")
    print(f"- Nodes: {sub_data.num_nodes}")
    print(f"- Edges: {sub_data.edge_index.size(1)}")
    print(f"- Node features shape: {sub_data.x.shape}")
    print(f"- Node labels shape: {sub_data.y.shape}")

    # Prepare data for training
    edge_index = sub_data.edge_index  # Directed edges in the graph
    N, d = sub_data.num_nodes, 2  # Set latent space dimension (e.g., 2D)

    lr = 0.01
    num_epochs = 3000

    # Initialize LatentPositionModel with node features and move model to GPU
    model = LatentPositionModel(N=N, d=d, variational_distribution=variational_distribution).to(device)
    optimizer = torch.optim.Adam([model.mu, model.log_sigma], lr=lr)

    # To store intermediate variables
    training_log = []
    all_mu = []  # Store all `mu` values across epochs

    # Move data to GPU (if using GPU)
    edge_index = edge_index.to(device)  # Move edge index to the device

    # Optimize for the ELBO (maximizing ELBO, so we minimize the negative ELBO)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        elbo = model.elbo(edge_index)

        # Check for NaN in ELBO
        if torch.isnan(elbo).any():
            print("NaN detected in ELBO!")
            break  # Stop training if NaN is detected

        # Backpropagate the ELBO
        elbo.backward()
        
        # Apply gradient clipping to avoid gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Save intermediate ELBO values and latent positions for debugging/analysis
        if epoch % 10 == 0:
            training_log.append((epoch, elbo.item()))
            print(f'Epoch {epoch}, ELBO: {elbo.item()}')

            # Store `mu` for later visualization
            all_mu.append(model.mu.cpu().detach().numpy())  # Save as numpy array for easy saving

    # Save the final latent positions and training log
    torch.save(all_mu, os.path.join(output_dir, f"{target_year}_{args.variational_distribution}_all_mu.pt"))  # Save all latent positions as a single file
    torch.save(training_log, os.path.join(output_dir, f"{target_year}_{args.variational_distribution}_training_log.pt"))

    # Visualize training log (ELBO values over epochs)
    epochs, elbo_values = zip(*training_log)
    plt.plot(epochs, elbo_values)
    plt.xlabel("Epochs")
    plt.ylabel("negative ELBO")
    plt.title("Training negative ELBO over Epochs")
    plt.savefig(os.path.join(output_dir, f"{target_year}_{args.variational_distribution}_training_elbo_plot.png"))  # Save plot to file
    plt.show()

    print("Training complete. Results saved.")
