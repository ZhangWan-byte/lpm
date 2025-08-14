import pickle
import random
from matplotlib.pylab import laplace
import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import subgraph
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import argparse

# Argument parsing for target_year
def parse_args():
    parser = argparse.ArgumentParser(description="Latent Position Model Training")
    parser.add_argument('--target_year', type=int, default=2010, help='The target year for the subgraph (default: 2010)')
    parser.add_argument('--variational_distribution', type=str, default='Gaussian', choices=['Gaussian', 'Laplace', 'MoG', 'StudentT'],
                        help='Choose the variational distribution (default: Gaussian)')
    parser.add_argument('--class_label', type=int, default=0, help='Class label for binary classification (default: 0)')
    parser.add_argument('--latent_dim', type=int, default=2, help='Dimensionality of the latent space (default: 2)')
    return parser.parse_args()

class LatentPositionModel(torch.nn.Module):
    def __init__(self, N, d=2, prior_distribution='Gaussian', variational_distribution='Gaussian', num_components=2):
        super(LatentPositionModel, self).__init__()
        self.N = N
        self.d = d
        self.prior_distribution = prior_distribution
        self.variational_distribution = variational_distribution
        self.num_components = num_components
        
        # Initialize latent means and log std dev
        if variational_distribution == 'Gaussian':
            self.mu = torch.randn(N, d, requires_grad=True)  # Latent means (for latent positions)
            self.log_sigma = torch.randn(N, d, requires_grad=True)  # Log of std dev to ensure positivity
        elif variational_distribution == 'Laplace':
            self.mu = torch.zeros(N, d, requires_grad=True)  # Latent means (for latent positions)
            self.b = torch.ones(N, d, requires_grad=True)  # Scale parameter for Laplace
            laplace = torch.distributions.Laplace(loc=self.mu, scale=self.b)
            self.Z = laplace.sample((N, 2))
        # For MoG, we add the mixture components (weights and component means)
        elif self.variational_distribution == 'MoG':
            self.mixture_weights = torch.nn.Parameter(torch.ones(self.num_components) / self.num_components, requires_grad=True)
            self.component_means = torch.randn(self.num_components, d, requires_grad=True)
            self.component_covariances = torch.nn.Parameter(torch.eye(d).unsqueeze(0).repeat(self.num_components, 1, 1), requires_grad=True)
        else:
            raise ValueError("Unsupported prior distribution. Use 'Gaussian' or 'Laplace'.")

        
    def get_variational_params(self):
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

        if self.prior_distribution == 'Gaussian':

            """Compute the KL divergence between the variational distribution (Gaussian/Laplace/MoG) and the prior (Gaussian N(0,I))."""
            if self.variational_distribution == 'Gaussian':
                kl = 0.5 * torch.sum(sigma ** 2 + mu ** 2 - torch.log(sigma) - 1)  # Closed-form KL for Gaussian prior N(0,I)
            elif self.variational_distribution == 'Laplace':
                # KL divergence between Laplace and Gaussian
                kl = torch.sum(torch.log(2 * sigma) - sigma - 0.5)  # Closed-form KL for Laplace prior
            elif self.variational_distribution == 'MoG':
                # KL divergence between Mixture of Gaussians and Gaussian prior
                kl = 0
                # Ensure that all tensors are on the correct device
                for k in range(self.num_components):
                    weight = self.mixture_weights[k]
                    mean = self.component_means[k].to(mu.device)  # Ensure mean is on the same device as mu
                    cov = self.component_covariances[k].to(mu.device)  # Ensure covariance is on the same device as mu
                    # Add terms related to the mixture components
                    kl += weight * (torch.trace(torch.inverse(cov)) + torch.matmul(mean, torch.matmul(torch.inverse(cov), mean)) - self.d)
            elif self.variational_distribution == 'StudentT':
                # KL divergence between Student t-distribution and Gaussian prior
                # A more complex calculation based on the properties of the t-distribution
                kl = torch.sum(torch.log(1 + mu**2 / (2 * sigma**2)))
            else:
                raise ValueError("Unsupported variational distribution. Use 'Gaussian', 'Laplace', 'MoG', or 'StudentT'.")
        # elif self.prior == 'Laplace':
        #     """Compute the KL divergence between the variational distribution (Laplace) and the prior (Gaussian/Laplace/MoG prior)."""
        #     if self.variational_distribution == 'Gaussian':
        #         # KL divergence between Gaussian and Laplace prior
        #         kl = -torch.log(2 * b) - 1 + 0.5 * torch.log(2 * np.pi * sigma**2) + 0.5 * (mu**2 + 2 * b**2) / sigma**2

        return kl

    def elbo(self, edge_index):
        """Compute the negative Evidence Lower Bound (ELBO)."""
        mu, sigma = self.get_variational_params()
        log_likelihood = self.likelihood(edge_index, mu, sigma)
        # kl = self.kl_divergence(mu, sigma)
        # return - log_likelihood + kl
        return -log_likelihood

if __name__ == "__main__":
    # Parse arguments for target_year
    args = parse_args()
    target_year = args.target_year
    variational_distribution = args.variational_distribution
    print(f"Using target year: {target_year}")
    print(f"Using variational distribution: {variational_distribution}")

    # Set device for training (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories if they don't exist
    output_dir = 'output_C3_new_0810_mle'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the data from the pickle file
    if target_year == 2010:
        file_path = 'preprocessed_data/train_val_graphs_2010_2015.pkl'
    else:
        file_path = f'preprocessed_data/test_graphs_{target_year}_{target_year+5}.pkl'

    with open(file_path, 'rb') as f:
        graphs, labels = pickle.load(f)
    
    print(f"Loaded {len(graphs)} graphs.")
    print(np.unique(labels, return_counts=True))

    # Filter graphs for class label
    graphs = [g for g, l in zip(graphs, labels) if l == args.class_label]
    labels = [l for l in labels if l == args.class_label]

    # Now you can use the loaded data
    print(f"After filtering, loaded {len(graphs)} graphs.")
    print(np.unique(labels, return_counts=True))
    print(f"Graph average edges: {np.mean([g.num_edges for g in graphs])}")

    # Prepare data for training
    N, d = graphs[0].num_nodes, args.latent_dim  # Set latent space dimension (e.g., 2D)

    lr = 0.01
    num_epochs = 20

    # Initialize LatentPositionModel with node features and move model to GPU
    model = LatentPositionModel(N=N, d=d, variational_distribution=variational_distribution).to(device)
    optimizer = torch.optim.Adam([model.mu, model.log_sigma], lr=lr)

    # To store intermediate variables
    training_log = []
    all_mu = []  # Store all `mu` values across epochs

    # Optimize for the ELBO (maximizing ELBO, so we minimize the negative ELBO)
    for epoch in range(num_epochs):

        for graph, label in tqdm(zip(graphs, labels), total=len(graphs), desc=f"Epoch {epoch+1}/{num_epochs}"):

            model.train()
            optimizer.zero_grad()

            # Move graph data to the same device as the model
            edge_index = graph.edge_index.to(device)

            # Forward pass: compute the negative ELBO
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
        if epoch % 1 == 0:
            training_log.append((epoch, elbo.item()))
            print(f'Epoch {epoch}, ELBO: {elbo.item()}')

            # Store `mu` for later visualization
            all_mu.append(model.mu.cpu().detach().numpy())  # Save as numpy array for easy saving

    # Save the final latent positions and training log
    torch.save(all_mu, os.path.join(output_dir, f"{target_year+5*args.class_label}_{args.variational_distribution}_{args.latent_dim}_all_mu.pt"))  # Save all latent positions as a single file
    torch.save(training_log, os.path.join(output_dir, f"{target_year+5*args.class_label}_{args.variational_distribution}_{args.latent_dim}_training_log.pt"))

    # Visualize training log (ELBO values over epochs)
    epochs, elbo_values = zip(*training_log)
    plt.plot(epochs, elbo_values)
    plt.xlabel("Epochs")
    plt.ylabel("negative ELBO")
    plt.title("Training negative ELBO over Epochs")
    plt.savefig(os.path.join(output_dir, f"{target_year+5*args.class_label}_{args.variational_distribution}_{args.latent_dim}_training_elbo_plot.png"))  # Save plot to file
    plt.show()

    print("Training complete. Results saved.")
