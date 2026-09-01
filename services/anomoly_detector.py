"""
Enterprise Spatio-Temporal Variational Autoencoder (VAE) for Network Telemetry.
Ingests live router metrics (utilization, latency, OSPF cost, optical DOM temperature,
CRC errors, traffic velocity, jitter, optical Rx power) and computes multi-dimensional
reconstruction loss with dynamic EMA thresholding and feature-level root-cause profiling.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

# Ensure root path resolution for sdn_kdn module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

try:
    from sdn_kdn.core.exceptions import AIInferenceError, DataShapeError
except ImportError:
    class DataShapeError(ValueError): pass
    class AIInferenceError(RuntimeError): pass

logger = logging.getLogger("GPUVAEAnomalyDetector")

# Canonical feature indexing for telemetry vectors
FEATURE_NAMES = [
    "utilization",
    "norm_latency",
    "ospf_cost",
    "optical_temp",
    "crc_error_rate",
    "traffic_velocity",
    "jitter",
    "optical_rx_power"
]


class GPUAnomalyDetector(nn.Module):
    """
    Production-grade Variational Autoencoder (VAE) for Carrier Telemetry.
    Computes Mean Squared Error (MSE) + KL-Divergence to identify anomalies
    and provides feature attribution to explain root-cause degradation.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, latent_dim: int = 4):
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

        # Latent Space Projections (Mean and Log-Variance)
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
            nn.Sigmoid(),  # Outputs bounded normalized telemetry [0.0, 1.0]
        )

        # Dynamic EMA baseline buffers (synced across CPU/GPU devices)
        self.register_buffer("ema_mean", torch.tensor(0.02))
        self.register_buffer("ema_std", torch.tensor(0.01))
        self.register_buffer("warmup_steps", torch.tensor(0, dtype=torch.long))

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Applies the reparameterization trick to sample from N(mu, var)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # Deterministic evaluation during inference

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: Encode -> Reparameterize -> Decode."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

    def _update_ema_baseline(self, batch_losses: torch.Tensor, alpha: float = 0.05) -> None:
        """Updates online moving average of reconstruction loss to track diurnal cycles."""
        with torch.no_grad():
            batch_mean = torch.mean(batch_losses)
            batch_std = torch.std(batch_losses) if batch_losses.numel() > 1 else torch.tensor(0.005, device=batch_losses.device)
            
            self.ema_mean.mul_(1.0 - alpha).add_(batch_mean * alpha)
            self.ema_std.mul_(1.0 - alpha).add_(batch_std * alpha)
            self.warmup_steps.add_(1)

    def detect(
        self, x: torch.Tensor, sensitivity: float = 3.0
    ) -> Tuple[Union[bool, torch.Tensor], Union[float, torch.Tensor]]:
        """
        Runs batched or single-tensor anomaly inference.
        Returns boolean anomaly masks and raw anomaly score tensors.
        """
        if x.shape[-1] != self.input_dim:
            raise DataShapeError(
                f"Expected trailing tensor dimension of size {self.input_dim}, "
                f"but received shape {tuple(x.shape)}."
            )

        try:
            with torch.inference_mode():
                # Scrub malformed metrics (NaN/Inf) resulting from dropped SNMP/gNMI packets
                clean_x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

                reconstruction, mu, logvar = self.forward(clean_x)

                # Composite Loss: Reconstruction Error (MSE) + KL-Divergence penalty
                reconstruction_err = torch.mean((clean_x - reconstruction) ** 2, dim=-1)
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
                anomaly_scores = reconstruction_err + (0.001 * kl_div)

                # Update streaming statistical baselines
                self._update_ema_baseline(anomaly_scores)

                # Compute dynamic anomaly cutoff threshold
                dynamic_threshold = self.ema_mean + (sensitivity * self.ema_std)
                dynamic_threshold = torch.clamp(dynamic_threshold, min=0.03, max=0.85)

                if clean_x.dim() == 1:
                    loss_val = float(anomaly_scores.item())
                    is_anomalous = bool(loss_val > dynamic_threshold.item())
                    return is_anomalous, loss_val
                else:
                    anomalies = anomaly_scores > dynamic_threshold
                    return anomalies, anomaly_scores

        except Exception as e:
            logger.error("VAE Anomaly Detector forward pass failed: %s", str(e), exc_info=True)
            raise AIInferenceError(f"VAE Inference failure: {e!s}")

    def explain_anomaly(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Computes feature-level attribution to determine the root cause of an alert.
        Returns a sorted mapping of features to their percentage contribution to the error.
        """
        if x.dim() > 1:
            x = x[0]

        with torch.inference_mode():
            clean_x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).unsqueeze(0)
            reconstruction, _, _ = self.forward(clean_x)
            
            sq_err_per_feature = ((clean_x - reconstruction) ** 2).squeeze(0)
            total_err = torch.sum(sq_err_per_feature).item()
            
            if total_err == 0.0:
                return {feat: 0.0 for feat in FEATURE_NAMES[:self.input_dim]}

            contributions = {}
            for i, feat in enumerate(FEATURE_NAMES[:self.input_dim]):
                ratio = (sq_err_per_feature[i].item() / total_err) * 100.0
                contributions[feat] = round(ratio, 2)

            return dict(sorted(contributions.items(), key=lambda item: item[1], reverse=True))


def get_anomaly_model(
    device: torch.device, weights_path: Optional[str] = None, input_dim: int = 8
) -> GPUAnomalyDetector:
    """
    Factory helper to initialize and load the VAE anomaly detector checkpoint.
    """
    model = GPUAnomalyDetector(input_dim=input_dim).to(device)

    if weights_path and os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            logger.info("✅ VAE Checkpoint loaded successfully from: %s", weights_path)
        except Exception as e:
            logger.warning(
                "⚠️ Checkpoint '%s' could not be loaded: %s. Operating with warm-initialized baseline.",
                weights_path, str(e)
            )

    model.eval()
    return model