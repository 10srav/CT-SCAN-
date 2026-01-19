"""
Variational Quantum Circuit (VQC) for CT Sinogram Denoising

This module implements the quantum denoising pipeline:
1. Classical preprocessing (patch extraction, normalization)
2. Quantum encoding (amplitude/angle embedding)
3. Variational circuit (RY/RZ rotations + CZ entanglement)
4. Measurement and classical post-processing

Mathematical Foundation:
- Radon Transform: Rf(θ,s) = ∫∫ f(x,y)δ(x·cosθ + y·sinθ - s) dx dy
- VQC Loss: L = ||y - ŷ||² + λ||∇ŷ - ∇y||² (MSE + Edge preservation)
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import logging

# Quantum Libraries
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    pnp = np

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit.circuit import Parameter
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# Classical ML
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Image Processing
from skimage.transform import radon, iradon
from skimage.restoration import denoise_tv_chambolle
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import wiener

logger = logging.getLogger(__name__)


@dataclass
class VQCConfig:
    """Configuration for VQC Denoiser"""
    num_qubits: int = 16
    num_layers: int = 6
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs: int = 100
    patch_size: int = 16
    lambda_edge: float = 0.1  # Edge loss weight
    lambda_lpips: float = 0.05  # Perceptual loss weight
    use_gpu: bool = True
    backend: str = "pennylane"  # "pennylane" or "qiskit"
    noise_model: Optional[str] = None  # "depolarizing", "amplitude_damping"


class QuantumLayer(nn.Module):
    """
    Hybrid Quantum-Classical Layer using PennyLane

    Architecture:
    - Input: Classical feature vector (flattened patch)
    - Embedding: Amplitude or Angle encoding
    - Ansatz: Hardware-efficient with RY, RZ, CZ gates
    - Output: Expectation values of Pauli-Z measurements
    """

    def __init__(self, config: VQCConfig):
        super().__init__()
        self.config = config
        self.n_qubits = config.num_qubits
        self.n_layers = config.num_layers

        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane is required for quantum layer")

        # Initialize quantum device
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # Trainable parameters: RY and RZ for each qubit in each layer
        self.params = nn.Parameter(
            torch.randn(self.n_layers, self.n_qubits, 2) * 0.1
        )

        # Create quantum node
        self.qnode = qml.QNode(
            self._circuit,
            self.dev,
            interface="torch",
            diff_method="backprop"
        )

    def _circuit(self, inputs: torch.Tensor, weights: torch.Tensor) -> List[float]:
        """
        Variational Quantum Circuit

        Args:
            inputs: Normalized input features [n_qubits]
            weights: Trainable parameters [n_layers, n_qubits, 2]

        Returns:
            Expectation values of Pauli-Z on all qubits
        """
        # Amplitude embedding (requires 2^n features, we pad/truncate)
        # Using angle embedding for simplicity (more practical)
        for i in range(self.n_qubits):
            qml.RY(inputs[i] * np.pi, wires=i)

        # Variational layers
        for layer in range(self.n_layers):
            # Single-qubit rotations
            for qubit in range(self.n_qubits):
                qml.RY(weights[layer, qubit, 0], wires=qubit)
                qml.RZ(weights[layer, qubit, 1], wires=qubit)

            # Entangling layer (CZ gates in ring topology)
            for qubit in range(self.n_qubits - 1):
                qml.CZ(wires=[qubit, qubit + 1])
            # Close the ring
            qml.CZ(wires=[self.n_qubits - 1, 0])

        # Measure all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantum layer

        Args:
            x: Input tensor [batch, n_qubits]

        Returns:
            Output tensor [batch, n_qubits]
        """
        batch_size = x.shape[0]
        outputs = []
        original_device = x.device

        # Move to CPU for quantum simulation (PennyLane requirement)
        x_cpu = x.cpu()
        params_cpu = self.params.cpu()

        for i in range(batch_size):
            # Normalize input to [-1, 1]
            inp = x_cpu[i]
            inp = 2 * (inp - inp.min()) / (inp.max() - inp.min() + 1e-8) - 1

            # Run quantum circuit (on CPU)
            out = self.qnode(inp, params_cpu)
            outputs.append(torch.stack(out))

        # Move result back to original device
        result = torch.stack(outputs).to(original_device)
        return result


class ClassicalEncoder(nn.Module):
    """Classical preprocessing network before quantum layer"""

    def __init__(self, input_dim: int, quantum_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, quantum_dim),
            nn.Tanh()  # Bound to [-1, 1] for quantum encoding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassicalDecoder(nn.Module):
    """Classical postprocessing network after quantum layer"""

    def __init__(self, quantum_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(quantum_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, output_dim),
            nn.Sigmoid()  # Output in [0, 1] for image values
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridVQCDenoiser(nn.Module):
    """
    Complete Hybrid Quantum-Classical Denoiser

    Pipeline:
    1. Extract patches from noisy sinogram
    2. Classical encoder compresses to quantum dimension
    3. VQC processes the quantum features
    4. Classical decoder expands back to patch size
    5. Reconstruct denoised sinogram from patches
    """

    def __init__(self, config: VQCConfig):
        super().__init__()
        self.config = config

        patch_dim = config.patch_size ** 2
        quantum_dim = config.num_qubits

        self.encoder = ClassicalEncoder(patch_dim, quantum_dim)
        self.quantum_layer = QuantumLayer(config)
        self.decoder = ClassicalDecoder(quantum_dim, patch_dim)

        # Residual connection weight (learnable)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input patches [batch, patch_size^2]

        Returns:
            Denoised patches [batch, patch_size^2]
        """
        # Store input for residual
        identity = x

        # Classical encoding
        encoded = self.encoder(x)

        # Quantum processing
        quantum_out = self.quantum_layer(encoded)

        # Convert to float32 (quantum layer outputs float64)
        quantum_out = quantum_out.float()

        # Classical decoding
        decoded = self.decoder(quantum_out)

        # Residual connection (learned weighted sum)
        output = decoded + self.residual_weight * identity

        return output


class EdgePreservingLoss(nn.Module):
    """
    Combined loss function for sinogram denoising

    L = MSE(y, ŷ) + λ_edge * EdgeLoss + λ_lpips * PerceptualLoss

    Edge loss uses Sobel gradient matching
    """

    def __init__(self, lambda_edge: float = 0.1, lambda_lpips: float = 0.05):
        super().__init__()
        self.lambda_edge = lambda_edge
        self.lambda_lpips = lambda_lpips

        # Sobel kernels for edge detection
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

    def compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude using Sobel filter"""
        device = x.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)

        # Reshape for convolution [batch, 1, H, W]
        if x.dim() == 3:
            x = x.unsqueeze(1)

        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)

        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        compute_breakdown: bool = False
    ) -> torch.Tensor:
        """
        Compute combined loss

        Args:
            pred: Predicted denoised sinogram
            target: Ground truth clean sinogram
            compute_breakdown: If True, return loss components

        Returns:
            Total loss (and components if requested)
        """
        # MSE Loss
        mse_loss = F.mse_loss(pred, target)

        # Edge Loss (gradient matching)
        if pred.dim() >= 3:  # Need 2D for edge computation
            pred_edges = self.compute_edges(pred)
            target_edges = self.compute_edges(target)
            edge_loss = F.mse_loss(pred_edges, target_edges)
        else:
            edge_loss = torch.tensor(0.0, device=pred.device)

        # Total loss
        total_loss = mse_loss + self.lambda_edge * edge_loss

        if compute_breakdown:
            return total_loss, {
                'mse': mse_loss.item(),
                'edge': edge_loss.item(),
                'total': total_loss.item()
            }

        return total_loss


class SinogramDenoiser:
    """
    High-level API for sinogram denoising

    Handles:
    - Patch extraction and reconstruction
    - Training loop with logging
    - Inference with batching
    - Model saving/loading
    """

    def __init__(self, config: Optional[VQCConfig] = None):
        self.config = config or VQCConfig()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config.use_gpu
            else "cpu"
        )

        self.model = HybridVQCDenoiser(self.config).to(self.device)
        self.loss_fn = EdgePreservingLoss(
            self.config.lambda_edge,
            self.config.lambda_lpips
        )
        self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)

        self.training_history = []

    def extract_patches(
        self,
        sinogram: np.ndarray,
        patch_size: int
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Extract non-overlapping patches from sinogram

        Args:
            sinogram: Input sinogram [H, W]
            patch_size: Size of square patches

        Returns:
            patches: Array of patches [N, patch_size^2]
            shape: Original sinogram shape for reconstruction
        """
        H, W = sinogram.shape

        # Pad to fit patches exactly
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size

        padded = np.pad(sinogram, ((0, pad_h), (0, pad_w)), mode='reflect')

        # Extract patches
        patches = []
        for i in range(0, padded.shape[0], patch_size):
            for j in range(0, padded.shape[1], patch_size):
                patch = padded[i:i+patch_size, j:j+patch_size]
                patches.append(patch.flatten())

        return np.array(patches), (H, W, padded.shape[0], padded.shape[1])

    def reconstruct_from_patches(
        self,
        patches: np.ndarray,
        shape_info: Tuple[int, int, int, int],
        patch_size: int
    ) -> np.ndarray:
        """
        Reconstruct sinogram from patches

        Args:
            patches: Array of patches [N, patch_size^2]
            shape_info: (orig_H, orig_W, padded_H, padded_W)
            patch_size: Size of square patches

        Returns:
            Reconstructed sinogram [H, W]
        """
        orig_h, orig_w, pad_h, pad_w = shape_info

        # Reshape patches
        n_rows = pad_h // patch_size
        n_cols = pad_w // patch_size

        reconstructed = np.zeros((pad_h, pad_w))
        idx = 0

        for i in range(n_rows):
            for j in range(n_cols):
                patch = patches[idx].reshape(patch_size, patch_size)
                reconstructed[
                    i*patch_size:(i+1)*patch_size,
                    j*patch_size:(j+1)*patch_size
                ] = patch
                idx += 1

        # Remove padding
        return reconstructed[:orig_h, :orig_w]

    def train_step(
        self,
        noisy_patches: torch.Tensor,
        clean_patches: torch.Tensor
    ) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        denoised = self.model(noisy_patches)

        # Compute loss
        loss, breakdown = self.loss_fn(denoised, clean_patches, compute_breakdown=True)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return breakdown

    def train(
        self,
        noisy_sinograms: np.ndarray,
        clean_sinograms: np.ndarray,
        validation_split: float = 0.1
    ) -> Dict[str, List[float]]:
        """
        Full training loop

        Args:
            noisy_sinograms: Array of noisy sinograms [N, H, W]
            clean_sinograms: Array of clean sinograms [N, H, W]
            validation_split: Fraction for validation

        Returns:
            Training history dictionary
        """
        # Extract all patches
        all_noisy_patches = []
        all_clean_patches = []

        for noisy, clean in zip(noisy_sinograms, clean_sinograms):
            n_patches, _ = self.extract_patches(noisy, self.config.patch_size)
            c_patches, _ = self.extract_patches(clean, self.config.patch_size)
            all_noisy_patches.append(n_patches)
            all_clean_patches.append(c_patches)

        noisy_patches = np.vstack(all_noisy_patches)
        clean_patches = np.vstack(all_clean_patches)

        # Train/validation split
        n_samples = len(noisy_patches)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)

        train_idx, val_idx = indices[n_val:], indices[:n_val]

        train_noisy = torch.FloatTensor(noisy_patches[train_idx]).to(self.device)
        train_clean = torch.FloatTensor(clean_patches[train_idx]).to(self.device)
        val_noisy = torch.FloatTensor(noisy_patches[val_idx]).to(self.device)
        val_clean = torch.FloatTensor(clean_patches[val_idx]).to(self.device)

        history = {'train_loss': [], 'val_loss': [], 'lr': []}

        for epoch in range(self.config.epochs):
            # Training
            epoch_losses = []
            for i in range(0, len(train_noisy), self.config.batch_size):
                batch_noisy = train_noisy[i:i+self.config.batch_size]
                batch_clean = train_clean[i:i+self.config.batch_size]

                loss_info = self.train_step(batch_noisy, batch_clean)
                epoch_losses.append(loss_info['total'])

            avg_train_loss = np.mean(epoch_losses)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(val_noisy)
                val_loss = self.loss_fn(val_pred, val_clean).item()

            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(current_lr)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs} - "
                    f"Train: {avg_train_loss:.6f}, Val: {val_loss:.6f}, "
                    f"LR: {current_lr:.6f}"
                )

        self.training_history = history
        return history

    def denoise(self, sinogram: np.ndarray) -> np.ndarray:
        """
        Denoise a single sinogram

        Args:
            sinogram: Noisy sinogram [H, W]

        Returns:
            Denoised sinogram [H, W]
        """
        self.model.eval()

        # Normalize
        sino_min, sino_max = sinogram.min(), sinogram.max()
        normalized = (sinogram - sino_min) / (sino_max - sino_min + 1e-8)

        # Extract patches
        patches, shape_info = self.extract_patches(normalized, self.config.patch_size)

        # Convert to tensor
        patches_tensor = torch.FloatTensor(patches).to(self.device)

        # Denoise
        with torch.no_grad():
            denoised_patches = self.model(patches_tensor)

        # Reconstruct
        denoised_patches = denoised_patches.cpu().numpy()
        denoised = self.reconstruct_from_patches(
            denoised_patches, shape_info, self.config.patch_size
        )

        # Denormalize
        denoised = denoised * (sino_max - sino_min) + sino_min

        return denoised

    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.training_history
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model checkpoint"""
        # Always load to CPU first, then move to target device
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.config = checkpoint['config']

        # Recreate model with loaded config
        self.model = HybridVQCDenoiser(self.config).to(self.device)

        # Load state dict (handle potential device mismatches)
        state_dict = checkpoint['model_state_dict']
        self.model.load_state_dict(state_dict)

        # Reinitialize optimizer for current device
        self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception:
            # Optimizer state may not be compatible, use fresh optimizer
            pass

        self.training_history = checkpoint.get('history', [])
        logger.info(f"Model loaded from {path}")


# Classical baseline implementations for comparison
class ClassicalDenoisers:
    """Classical denoising baselines for benchmarking"""

    @staticmethod
    def gaussian(sinogram: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Gaussian filter denoising"""
        return gaussian_filter(sinogram, sigma=sigma)

    @staticmethod
    def median(sinogram: np.ndarray, size: int = 3) -> np.ndarray:
        """Median filter denoising"""
        return median_filter(sinogram, size=size)

    @staticmethod
    def wiener_filter(sinogram: np.ndarray, noise: float = None) -> np.ndarray:
        """Wiener filter denoising"""
        return wiener(sinogram, noise=noise)

    @staticmethod
    def tv_denoise(sinogram: np.ndarray, weight: float = 0.1) -> np.ndarray:
        """Total Variation denoising"""
        return denoise_tv_chambolle(sinogram, weight=weight)


class CTReconstructor:
    """CT image reconstruction from sinograms"""

    @staticmethod
    def radon_transform(image: np.ndarray, theta: np.ndarray = None) -> np.ndarray:
        """
        Forward Radon transform: Image -> Sinogram

        Rf(θ,s) = ∫∫ f(x,y)δ(x·cosθ + y·sinθ - s) dx dy
        """
        if theta is None:
            theta = np.linspace(0., 180., max(image.shape), endpoint=False)
        return radon(image, theta=theta, circle=True)

    @staticmethod
    def fbp_reconstruct(
        sinogram: np.ndarray,
        theta: np.ndarray = None,
        filter_name: str = 'ramp'
    ) -> np.ndarray:
        """
        Filtered Back Projection reconstruction: Sinogram -> Image

        Args:
            sinogram: Input sinogram
            theta: Projection angles
            filter_name: 'ramp', 'shepp-logan', 'cosine', 'hamming', 'hann'

        Returns:
            Reconstructed image
        """
        if theta is None:
            theta = np.linspace(0., 180., sinogram.shape[1], endpoint=False)
        return iradon(sinogram, theta=theta, filter_name=filter_name, circle=True)


# Utility functions for adding noise
def add_poisson_noise(sinogram: np.ndarray, dose_fraction: float = 0.25) -> np.ndarray:
    """
    Add Poisson noise to simulate low-dose CT

    Args:
        sinogram: Clean sinogram
        dose_fraction: Fraction of full dose (0.25 = 25% dose)

    Returns:
        Noisy sinogram
    """
    # Scale to photon counts
    max_counts = 1e5 * dose_fraction
    scaled = (sinogram - sinogram.min()) / (sinogram.max() - sinogram.min())
    scaled = scaled * max_counts + 1  # Avoid zero

    # Add Poisson noise
    noisy_counts = np.random.poisson(scaled)

    # Convert back to sinogram scale
    noisy = noisy_counts / max_counts * (sinogram.max() - sinogram.min()) + sinogram.min()

    return noisy


def add_gaussian_noise(sinogram: np.ndarray, std: float = 0.05) -> np.ndarray:
    """Add Gaussian noise to sinogram"""
    noise = np.random.normal(0, std * sinogram.std(), sinogram.shape)
    return sinogram + noise


def add_combined_noise(
    sinogram: np.ndarray,
    dose_fraction: float = 0.25,
    gaussian_std: float = 0.02
) -> np.ndarray:
    """Add realistic combined Poisson + Gaussian noise"""
    noisy = add_poisson_noise(sinogram, dose_fraction)
    noisy = add_gaussian_noise(noisy, gaussian_std)
    return noisy


if __name__ == "__main__":
    # Quick test
    from skimage.data import shepp_logan_phantom

    # Generate test data
    phantom = shepp_logan_phantom()
    phantom = phantom / phantom.max()

    # Create sinogram
    theta = np.linspace(0., 180., 180, endpoint=False)
    clean_sinogram = CTReconstructor.radon_transform(phantom, theta)

    # Add noise
    noisy_sinogram = add_combined_noise(clean_sinogram, dose_fraction=0.25)

    # Initialize denoiser
    config = VQCConfig(num_qubits=8, num_layers=4, epochs=10)
    denoiser = SinogramDenoiser(config)

    print(f"Clean sinogram shape: {clean_sinogram.shape}")
    print(f"Noisy sinogram shape: {noisy_sinogram.shape}")
    print(f"Device: {denoiser.device}")
    print("VQC Denoiser initialized successfully!")
