import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal
from torch.utils.data import Dataset, DataLoader, TensorDataset


# Create an artificial dataset
n_samples = 1000
n_features = 20
n_clusters = 5

# Create random cluster centers
cluster_centers = torch.randn(n_clusters, n_features)

# Generate data points around cluster centers
X_artificial = torch.cat([cluster_centers[c].unsqueeze(0) + 0.1 * torch.randn(n_samples // n_clusters, n_features)
                          for c in range(n_clusters)], dim=0)

# Convert data to PyTorch tensors
X_tensor = torch.FloatTensor(X_artificial)


class AutoencoderVBGMM(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_components):
        super(AutoencoderVBGMM, self).__init__()
        
        # Autoencoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            # nn.Sigmoid()  # Output values between 0 and 1
        )
        
        # VBGMM parameters
        self.num_components = num_components
        self.pi = nn.Parameter(torch.ones(num_components) / num_components, requires_grad=True)
        self.mu = nn.Parameter(torch.randn(num_components, latent_dim), requires_grad=True)
        self.log_var = nn.Parameter(torch.zeros(num_components, latent_dim), requires_grad=True)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded, encoded
    
    def calculate_loss(self, x):
        decoded, encoded = self.forward(x)
        
        # Reconstruction loss (autoencoder)
        recon_loss = nn.functional.mse_loss(decoded, x, reduction='sum')
        
        # VBGMM log likelihood
        log_likelihoods = []
        for c in range(self.num_components):
            var = torch.exp(self.log_var[c])
            dist = MultivariateNormal(self.mu[c], torch.diag(var))
            log_likelihoods.append(dist.log_prob(encoded))
        log_likelihoods = torch.stack(log_likelihoods, dim=1)
        log_likelihoods = torch.logsumexp(log_likelihoods + torch.log(self.pi), dim=1)
        
        # Negative log likelihood loss
        neg_log_likelihood = -torch.sum(log_likelihoods)
        
        # Total loss
        total_loss = recon_loss + neg_log_likelihood
        
        return total_loss

# Instantiate the model
input_dim = n_features
hidden_dim = 128
latent_dim = 2
num_components = 5

model = AutoencoderVBGMM(input_dim, hidden_dim, latent_dim, num_components)

# Create DataLoader for batching
batch_size = 32
data_loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=True)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    for batch in data_loader:
        x_batch = batch[0]
        optimizer.zero_grad()
        loss = model.calculate_loss(x_batch)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

print("Training complete!")
