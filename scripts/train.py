#!/usr/bin/env python3
"""
Quantum CT Sinogram Denoising - Training Script

Trains the VQC-based denoising model on CT sinogram data.

Usage:
    python train.py --epochs 100 --batch-size 32 --output model.pt

Features:
    - Hybrid quantum-classical training loop
    - MLflow experiment tracking
    - Checkpoint saving
    - Learning rate scheduling
    - Early stopping
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import h5py
from tqdm import tqdm

# Optional MLflow tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

from app.quantum.vqc_denoiser import (
    SinogramDenoiser, VQCConfig, add_combined_noise
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CTDataset:
    """CT Sinogram Dataset Loader"""

    def __init__(self, data_path: str, patch_size: int = 16):
        self.data_path = Path(data_path)
        self.patch_size = patch_size

    def load_hdf5(self) -> tuple:
        """Load preprocessed HDF5 dataset"""
        h5_path = self.data_path / 'sinogram_dataset.h5'

        if not h5_path.exists():
            logger.warning(f"HDF5 dataset not found at {h5_path}")
            return self._generate_synthetic()

        with h5py.File(h5_path, 'r') as f:
            train_clean = f['train/clean'][:]
            train_noisy = f['train/noisy'][:]
            val_clean = f['val/clean'][:]
            val_noisy = f['val/noisy'][:]

        logger.info(f"Loaded dataset from {h5_path}")
        logger.info(f"  Train: {len(train_clean)} sinograms")
        logger.info(f"  Val: {len(val_clean)} sinograms")

        return (train_clean, train_noisy), (val_clean, val_noisy)

    def _generate_synthetic(self) -> tuple:
        """Generate synthetic training data"""
        from skimage.data import shepp_logan_phantom
        from skimage.transform import radon

        logger.info("Generating synthetic training data...")

        theta = np.linspace(0., 180., 180, endpoint=False)
        n_train, n_val = 80, 20

        train_clean, train_noisy = [], []
        val_clean, val_noisy = [], []

        for i in tqdm(range(n_train + n_val), desc="Generating data"):
            # Base phantom with variations
            phantom = shepp_logan_phantom()
            phantom += np.random.randn(*phantom.shape) * 0.02
            phantom = np.clip(phantom, 0, 1)

            # Create sinogram
            clean = radon(phantom, theta=theta, circle=True)
            noisy = add_combined_noise(clean, dose_fraction=0.25, gaussian_std=0.02)

            # Normalize
            max_val = max(clean.max(), noisy.max())
            clean = clean / max_val
            noisy = noisy / max_val

            if i < n_train:
                train_clean.append(clean)
                train_noisy.append(noisy)
            else:
                val_clean.append(clean)
                val_noisy.append(noisy)

        return (
            (np.array(train_clean), np.array(train_noisy)),
            (np.array(val_clean), np.array(val_noisy))
        )

    def extract_patches(self, sinograms: np.ndarray) -> np.ndarray:
        """Extract patches from sinograms"""
        patches = []
        ps = self.patch_size

        for sino in sinograms:
            H, W = sino.shape
            for i in range(0, H - ps + 1, ps):
                for j in range(0, W - ps + 1, ps):
                    patch = sino[i:i+ps, j:j+ps]
                    patches.append(patch.flatten())

        return np.array(patches)


class Trainer:
    """VQC Training Manager"""

    def __init__(
        self,
        config: VQCConfig,
        output_dir: str,
        use_mlflow: bool = True
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE

        # Initialize denoiser
        self.denoiser = SinogramDenoiser(config)
        self.device = self.denoiser.device

        logger.info(f"Trainer initialized on device: {self.device}")

        # Training state
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def train(
        self,
        train_data: tuple,
        val_data: tuple,
        epochs: int,
        batch_size: int,
        early_stopping_patience: int = 20
    ) -> dict:
        """
        Main training loop

        Args:
            train_data: (clean_sinograms, noisy_sinograms)
            val_data: (clean_sinograms, noisy_sinograms)
            epochs: Number of training epochs
            batch_size: Batch size
            early_stopping_patience: Epochs before early stopping

        Returns:
            Training history dictionary
        """
        # Setup MLflow
        if self.use_mlflow:
            mlflow.set_experiment("quantum-ct-denoising")
            mlflow.start_run(run_name=f"vqc_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            mlflow.log_params({
                "num_qubits": self.config.num_qubits,
                "vqc_layers": self.config.num_layers,
                "learning_rate": self.config.learning_rate,
                "batch_size": batch_size,
                "epochs": epochs
            })

        # Extract patches
        dataset = CTDataset(self.output_dir, self.config.patch_size)
        train_clean_patches = dataset.extract_patches(train_data[0])
        train_noisy_patches = dataset.extract_patches(train_data[1])
        val_clean_patches = dataset.extract_patches(val_data[0])
        val_noisy_patches = dataset.extract_patches(val_data[1])

        logger.info(f"Training patches: {train_clean_patches.shape}")
        logger.info(f"Validation patches: {val_clean_patches.shape}")

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(train_noisy_patches),
            torch.FloatTensor(train_clean_patches)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_noisy = torch.FloatTensor(val_noisy_patches).to(self.device)
        val_clean = torch.FloatTensor(val_clean_patches).to(self.device)

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        logger.info(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            # Training phase
            self.denoiser.model.train()
            epoch_losses = []

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for batch_noisy, batch_clean in pbar:
                batch_noisy = batch_noisy.to(self.device)
                batch_clean = batch_clean.to(self.device)

                loss_info = self.denoiser.train_step(batch_noisy, batch_clean)
                epoch_losses.append(loss_info['total'])
                pbar.set_postfix({'loss': f"{loss_info['total']:.6f}"})

            avg_train_loss = np.mean(epoch_losses)

            # Validation phase
            self.denoiser.model.eval()
            with torch.no_grad():
                val_pred = self.denoiser.model(val_noisy)
                val_loss = self.denoiser.loss_fn(val_pred, val_clean).item()

            # Update scheduler
            self.denoiser.scheduler.step()
            current_lr = self.denoiser.scheduler.get_last_lr()[0]

            # Record history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rate'].append(current_lr)

            # Log to MLflow
            if self.use_mlflow:
                mlflow.log_metrics({
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'learning_rate': current_lr
                }, step=epoch)

            # Logging
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train: {avg_train_loss:.6f}, Val: {val_loss:.6f}, "
                    f"LR: {current_lr:.6f}"
                )

            # Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

            # Periodic checkpoint
            if (epoch + 1) % 50 == 0:
                self._save_checkpoint(epoch, val_loss, is_best=False)

        # Final save
        final_path = self.output_dir / 'vqc_model_final.pt'
        self.denoiser.save(str(final_path))

        if self.use_mlflow:
            mlflow.log_artifact(str(final_path))
            mlflow.end_run()

        logger.info(f"Training complete. Best val loss: {self.best_val_loss:.6f}")

        return history

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        """Save model checkpoint"""
        if is_best:
            path = self.output_dir / 'vqc_model_best.pt'
        else:
            path = self.output_dir / f'vqc_model_epoch{epoch+1}.pt'

        self.denoiser.save(str(path))
        logger.info(f"Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser(description='Train VQC Denoiser')

    # Data arguments
    parser.add_argument('--data-dir', type=str, default='./data/processed',
                        help='Path to processed data directory')

    # Model arguments
    parser.add_argument('--num-qubits', type=int, default=16,
                        help='Number of qubits (default: 16)')
    parser.add_argument('--vqc-layers', type=int, default=6,
                        help='Number of VQC layers (default: 6)')
    parser.add_argument('--patch-size', type=int, default=16,
                        help='Patch size (default: 16)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--early-stopping', type=int, default=20,
                        help='Early stopping patience (default: 20)')

    # Output arguments
    parser.add_argument('--output', type=str, default='./data/models',
                        help='Output directory for model')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU if available')
    parser.add_argument('--no-mlflow', action='store_true',
                        help='Disable MLflow tracking')

    args = parser.parse_args()

    # Create config
    config = VQCConfig(
        num_qubits=args.num_qubits,
        num_layers=args.vqc_layers,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patch_size=args.patch_size,
        use_gpu=args.use_gpu
    )

    # Load data
    dataset = CTDataset(args.data_dir, args.patch_size)
    train_data, val_data = dataset.load_hdf5()

    # Initialize trainer
    trainer = Trainer(
        config=config,
        output_dir=args.output,
        use_mlflow=not args.no_mlflow
    )

    # Train
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.early_stopping
    )

    # Save history
    import json
    history_path = Path(args.output) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
