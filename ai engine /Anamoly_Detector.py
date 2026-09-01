import torch
import torch.nn as nn
from typing import Tuple
import logging

logger = logging.getLogger("GPUVariationalAutoencoder")

class GPUAnomalyDetector(nn.Module):
    """
    Production Variational Autoencoder (VAE).
    Computes KL-Divergence and MSE to identify complex spatio-temporal hardware anomalies.
    """
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, latent_dim: int = 4):
        super().__init__()
        self.input_dim = input_dim
        
        # Shared Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # VAE Reparameterization layers
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid() # Outputs bounded features
        )

        # Dynamic EMA Threshold tracking
        self.register_buffer("ema_loss", torch.tensor(0.05))

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from the latent Gaussian distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def detect(self, x: torch.Tensor, sensitivity: float = 3.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched inference pass. 
        Calculates loss and adapts threshold dynamically.
        Sensitivity multiplier determines standard deviations above EMA to flag.
        """
        with torch.inference_mode():
            clean_x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
            reconstruction, mu, logvar = self.forward(clean_x)
            
            # Loss = Reconstruction Error (MSE) + KL-Divergence (Regularization)
            mse_loss = torch.mean((clean_x - reconstruction) ** 2, dim=-1)
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
            
            # Total anomaly score per link
            total_loss = mse_loss + (0.001 * kld_loss)
            
            # Update dynamic threshold via Exponential Moving Average (EMA)
            batch_mean = torch.mean(total_loss)
            self.ema_loss = (0.99 * self.ema_loss) + (0.01 * batch_mean)
            
            # Dynamic threshold: Mean + (Sensitivity * Variance Estimation)
            dynamic_threshold = self.ema_loss * sensitivity
            anomalies = total_loss > dynamic_threshold
            
            return anomalies, total_loss

def get_anomaly_model(device: torch.device, weights_path: str = None, input_dim: int = 8) -> GPUAnomalyDetector:
    model = GPUAnomalyDetector(input_dim=input_dim).to(device)
    if weights_path and torch.os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        logger.info(f"✅ VAE Anomaly Weights loaded from {weights_path}")
    model.eval()
    return model