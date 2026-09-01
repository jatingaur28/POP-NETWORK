"""
Enterprise Production AI Training Pipeline for TATA SDN Controller.
Explicitly loads 'Dataset.csv' and 'Label.csv', aligns them, filters
normal baseline traffic, trains the Spatio-Temporal VAE, and saves weights.
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Directory path configurations
CURRENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = CURRENT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

WEIGHTS_PATH_VAE = MODELS_DIR / "prod_vae.pth"
WEIGHTS_PATH_BRAIN = CURRENT_DIR / "oracle_brain.pth"

DATASET_PATH = CURRENT_DIR / "Dataset.csv"
LABEL_PATH = CURRENT_DIR / "Label.csv"


# ==============================================================================
# 1. VAE ANOMALY DETECTOR ARCHITECTURE
# ==============================================================================
class GPUAnomalyDetector(nn.Module):
    """
    Variational Autoencoder (VAE) for Network Telemetry Anomaly Detection.
    Learns normal multi-dimensional network operational bounds.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, latent_dim: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder Network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # Latent Space Projections
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder Network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # Normalizes reconstructed features between [0.0, 1.0]
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Applies reparameterization trick during training."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar


# ==============================================================================
# 2. DATASET INGESTION & LABEL FILTERING PIPELINE
# ==============================================================================
def load_custom_datasets(max_rows: int = 50000) -> torch.Tensor:
    """
    Loads Dataset.csv and Label.csv, aligns records, filters normal baseline data,
    normalizes features, and returns a PyTorch Tensor.
    """
    if DATASET_PATH.exists():
        print(f"📁 Loading feature set from: {DATASET_PATH.name}...")
        try:
            df_data = pd.read_csv(DATASET_PATH, nrows=max_rows, low_memory=False)
            
            # Keep only numeric columns
            numeric_df = df_data.select_dtypes(include=[np.number])
            if "id" in numeric_df.columns:
                numeric_df = numeric_df.drop(columns=["id"])

            # Use Label.csv if present to isolate normal baseline records
            if LABEL_PATH.exists():
                print(f"📁 Loading classification labels from: {LABEL_PATH.name}...")
                df_labels = pd.read_csv(LABEL_PATH, nrows=max_rows, low_memory=False)
                
                # Automatically detect label column name
                label_col = None
                for col in df_labels.columns:
                    if any(kw in col.lower() for kw in ['label', 'class', 'target', 'attack']):
                        label_col = col
                        break
                
                if label_col and len(df_labels) == len(numeric_df):
                    # Filter for normal traffic (e.g., label == 0 or 'Normal')
                    normal_mask = (df_labels[label_col] == 0) | (df_labels[label_col].astype(str).str.lower() == 'normal')
                    if normal_mask.sum() > 50:
                        numeric_df = numeric_df[normal_mask]
                        print(f"   ↳ Filtered and retained {normal_mask.sum()} normal baseline training rows using {LABEL_PATH.name}.")

            raw_data = numeric_df.dropna().values
            if raw_data.shape[0] > 50:
                print(f"   ↳ Successfully extracted {raw_data.shape[0]} records and {raw_data.shape[1]} features.")
                # Min-Max Normalization to [0, 1]
                min_vals = raw_data.min(axis=0)
                max_vals = raw_data.max(axis=0)
                denom = np.where((max_vals - min_vals) == 0, 1.0, max_vals - min_vals)
                normalized = (raw_data - min_vals) / denom
                return torch.tensor(normalized, dtype=torch.float32)
        except Exception as e:
            print(f"⚠️ Error parsing Dataset.csv / Label.csv: {e}. Falling back to simulation.")

    # Fallback synthetic generator if files are missing
    print("📊 Dataset.csv not found. Generating fallback diurnal telemetry samples...")
    num_samples = 5000
    time_steps = np.linspace(0, 4 * np.pi, num_samples)
    utilization = np.clip(0.40 + 0.25 * np.sin(time_steps) + np.random.normal(0, 0.05, num_samples), 0.05, 0.90)
    latency = np.clip(0.20 + (utilization * 0.3) + np.random.normal(0, 0.03, num_samples), 0.05, 0.95)
    ospf_cost = np.random.choice([0.1, 0.2, 0.5, 0.8], size=num_samples)
    temp = np.clip(0.35 + (utilization * 0.25) + np.random.normal(0, 0.02, num_samples), 0.2, 0.85)
    errors = np.clip(np.random.exponential(scale=0.01, size=num_samples), 0.0, 0.2)
    velocity = np.clip(np.gradient(utilization) * 5.0 + 0.5, 0.0, 1.0)
    jitter = np.clip(0.10 + (errors * 0.5) + np.random.normal(0, 0.02, num_samples), 0.02, 0.8)
    rx_power = np.clip(0.70 - (temp * 0.15) + np.random.normal(0, 0.02, num_samples), 0.1, 0.95)

    dataset = np.stack([utilization, latency, ospf_cost, temp, errors, velocity, jitter, rx_power], axis=1)
    return torch.tensor(dataset, dtype=torch.float32)


# ==============================================================================
# 3. TRAINING EXECUTION BLOCK
# ==============================================================================
def train_vae(epochs: int = 40, batch_size: int = 64, lr: float = 0.002):
    # Select Apple Silicon MPS GPU if available, else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"🚀 Training Device: {device}")

    tensor_data = load_custom_datasets()
    input_dim = tensor_data.shape[1]

    model = GPUAnomalyDetector(input_dim=input_dim, hidden_dim=32, latent_dim=4).to(device)
    model.train()

    dataset = TensorDataset(tensor_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"\n🧠 Starting training loop ({epochs} epochs across {len(tensor_data)} samples)...")
    for epoch in range(1, epochs + 1):
        total_loss, total_recon = 0.0, 0.0

        for batch in loader:
            x = batch[0].to(device)
            optimizer.zero_grad()

            reconstruction, mu, logvar = model(x)

            recon_loss = torch.mean((x - reconstruction) ** 2)
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
            loss = recon_loss + (0.001 * kl_loss)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_loss / len(loader)
            avg_recon = total_recon / len(loader)
            print(f"   Epoch [{epoch:02d}/{epochs:02d}] ➔ Loss: {avg_loss:.6f} | Reconstruction MSE: {avg_recon:.6f}")

    # Serialize and save trained weights
    torch.save(model.state_dict(), str(WEIGHTS_PATH_VAE))
    torch.save(model.state_dict(), str(WEIGHTS_PATH_BRAIN))

    print(f"\n✅ Training Complete Successfully!")
    print(f"   ↳ Saved VAE weights to   : {WEIGHTS_PATH_VAE}")
    print(f"   ↳ Saved Oracle brain to  : {WEIGHTS_PATH_BRAIN}")


if __name__ == "__main__":
    train_vae(epochs=40, batch_size=64, lr=0.002)