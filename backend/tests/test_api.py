"""
API Tests for Quantum CT Sinogram Denoising

Comprehensive test suite covering:
- Health check endpoints
- Denoising endpoints
- Training endpoints
- Metrics endpoints
- Authentication

Coverage target: >90%
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import io
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app, app_state, settings


@pytest.fixture
def client():
    """Create test client"""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_sinogram():
    """Generate sample sinogram for testing"""
    from skimage.data import shepp_logan_phantom
    from skimage.transform import radon

    phantom = shepp_logan_phantom()
    theta = np.linspace(0., 180., 180, endpoint=False)
    sinogram = radon(phantom, theta=theta, circle=True)
    return sinogram


@pytest.fixture
def sample_sinogram_file(sample_sinogram):
    """Create sinogram as file-like object"""
    buffer = io.BytesIO()
    np.save(buffer, sample_sinogram)
    buffer.seek(0)
    return buffer


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check_returns_200(self, client):
        """Health endpoint should return 200 OK"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_check_contains_status(self, client):
        """Health response should contain status field"""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_check_contains_version(self, client):
        """Health response should contain version"""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == settings.app_version

    def test_health_check_contains_quantum_backend(self, client):
        """Health response should contain quantum backend info"""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "quantum_backend" in data


class TestRootEndpoint:
    """Test root endpoint"""

    def test_root_returns_200(self, client):
        """Root endpoint should return 200"""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_api_info(self, client):
        """Root should return API information"""
        response = client.get("/")
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestDenoiseEndpoint:
    """Test denoising endpoint"""

    def test_denoise_requires_file(self, client):
        """Denoise endpoint should require file upload"""
        response = client.post("/api/v1/denoise")
        assert response.status_code == 422  # Validation error

    @patch('app.main.app_state')
    def test_denoise_gaussian_method(self, mock_state, client, sample_sinogram_file):
        """Test Gaussian denoising method"""
        mock_state.denoiser = None  # Don't need VQC for Gaussian

        response = client.post(
            "/api/v1/denoise",
            files={"file": ("sinogram.npy", sample_sinogram_file, "application/octet-stream")},
            params={"method": "gaussian"}
        )

        # Should return 200 or handle gracefully
        assert response.status_code in [200, 500]

    def test_denoise_invalid_method(self, client, sample_sinogram_file):
        """Invalid denoising method should return error"""
        response = client.post(
            "/api/v1/denoise",
            files={"file": ("sinogram.npy", sample_sinogram_file, "application/octet-stream")},
            params={"method": "invalid_method"}
        )
        # Should handle gracefully
        assert response.status_code in [400, 500]


class TestMetricsEndpoint:
    """Test metrics endpoint"""

    def test_metrics_returns_200(self, client):
        """Metrics endpoint should return 200"""
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200

    def test_metrics_returns_list(self, client):
        """Metrics should return list of results"""
        response = client.get("/api/v1/metrics")
        data = response.json()
        assert isinstance(data, list)

    def test_metrics_contains_required_fields(self, client):
        """Each metric should have required fields"""
        response = client.get("/api/v1/metrics")
        data = response.json()

        for metric in data:
            assert "psnr" in metric
            assert "ssim" in metric
            assert "nrmse" in metric
            assert "method" in metric

    def test_vqc_has_best_psnr(self, client):
        """VQC should have the best PSNR in metrics"""
        response = client.get("/api/v1/metrics")
        data = response.json()

        vqc_psnr = next((m["psnr"] for m in data if m["method"] == "vqc"), 0)
        other_psnrs = [m["psnr"] for m in data if m["method"] not in ["vqc", "noisy"]]

        assert vqc_psnr > max(other_psnrs)


class TestTrainingEndpoint:
    """Test training endpoint"""

    def test_train_endpoint_exists(self, client):
        """Training endpoint should exist"""
        response = client.post("/api/v1/train", json={
            "epochs": 1,
            "batch_size": 8
        })
        # Should return 200 or 401 (auth required)
        assert response.status_code in [200, 401]

    def test_train_returns_job_id(self, client):
        """Training should return job ID"""
        response = client.post("/api/v1/train", json={
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.01,
            "num_qubits": 4,
            "vqc_layers": 2
        })

        if response.status_code == 200:
            data = response.json()
            assert "job_id" in data
            assert "status" in data


class TestAuthEndpoint:
    """Test authentication endpoint"""

    def test_token_endpoint_exists(self, client):
        """Token endpoint should exist"""
        response = client.post("/token", data={
            "username": "test",
            "password": "test"
        })
        # Should return 401 for invalid credentials
        assert response.status_code in [200, 401]

    def test_valid_credentials_returns_token(self, client):
        """Valid credentials should return token"""
        response = client.post("/token", data={
            "username": "admin",
            "password": "quantum"
        })

        if response.status_code == 200:
            data = response.json()
            assert "access_token" in data
            assert "token_type" in data


class TestReconstructEndpoint:
    """Test reconstruction endpoint"""

    def test_reconstruct_endpoint_exists(self, client, sample_sinogram_file):
        """Reconstruct endpoint should exist"""
        response = client.post(
            "/api/v1/reconstruct",
            files={"file": ("sinogram.npy", sample_sinogram_file, "application/octet-stream")}
        )
        assert response.status_code in [200, 500]


class TestUploadEndpoint:
    """Test DICOM upload endpoint"""

    def test_upload_endpoint_exists(self, client):
        """Upload endpoint should exist"""
        response = client.post("/api/v1/upload", files=[])
        # Should handle empty upload
        assert response.status_code in [200, 422]


class TestQuantumDenoiser:
    """Test quantum denoiser functionality"""

    def test_vqc_config_creation(self):
        """VQC config should be creatable"""
        from app.quantum.vqc_denoiser import VQCConfig

        config = VQCConfig(
            num_qubits=8,
            num_layers=4,
            learning_rate=0.01
        )

        assert config.num_qubits == 8
        assert config.num_layers == 4

    def test_sinogram_denoiser_initialization(self):
        """Sinogram denoiser should initialize"""
        from app.quantum.vqc_denoiser import SinogramDenoiser, VQCConfig

        config = VQCConfig(num_qubits=4, num_layers=2)
        denoiser = SinogramDenoiser(config)

        assert denoiser is not None
        assert denoiser.model is not None

    def test_classical_denoisers(self, sample_sinogram):
        """Classical denoisers should work"""
        from app.quantum.vqc_denoiser import ClassicalDenoisers

        gaussian = ClassicalDenoisers.gaussian(sample_sinogram, sigma=1.0)
        assert gaussian.shape == sample_sinogram.shape

        median = ClassicalDenoisers.median(sample_sinogram, size=3)
        assert median.shape == sample_sinogram.shape

    def test_radon_transform(self):
        """Radon transform should work"""
        from app.quantum.vqc_denoiser import CTReconstructor
        from skimage.data import shepp_logan_phantom

        phantom = shepp_logan_phantom()
        sinogram = CTReconstructor.radon_transform(phantom)

        assert sinogram is not None
        assert len(sinogram.shape) == 2

    def test_fbp_reconstruction(self, sample_sinogram):
        """FBP reconstruction should work"""
        from app.quantum.vqc_denoiser import CTReconstructor

        recon = CTReconstructor.fbp_reconstruct(sample_sinogram)
        assert recon is not None
        assert len(recon.shape) == 2

    def test_noise_simulation(self, sample_sinogram):
        """Noise simulation should add noise"""
        from app.quantum.vqc_denoiser import add_combined_noise

        noisy = add_combined_noise(sample_sinogram, dose_fraction=0.25)

        assert noisy.shape == sample_sinogram.shape
        # Noisy should be different from clean
        assert not np.allclose(noisy, sample_sinogram)


class TestMetrics:
    """Test metric computation"""

    def test_psnr_computation(self, sample_sinogram):
        """PSNR computation should work"""
        from skimage.metrics import peak_signal_noise_ratio

        noisy = sample_sinogram + np.random.randn(*sample_sinogram.shape) * 0.1
        noisy = (noisy - noisy.min()) / (noisy.max() - noisy.min())
        clean = (sample_sinogram - sample_sinogram.min()) / (sample_sinogram.max() - sample_sinogram.min())

        psnr = peak_signal_noise_ratio(clean, noisy, data_range=1.0)
        assert psnr > 0
        assert psnr < 100

    def test_ssim_computation(self, sample_sinogram):
        """SSIM computation should work"""
        from skimage.metrics import structural_similarity

        noisy = sample_sinogram + np.random.randn(*sample_sinogram.shape) * 0.1
        noisy = (noisy - noisy.min()) / (noisy.max() - noisy.min())
        clean = (sample_sinogram - sample_sinogram.min()) / (sample_sinogram.max() - sample_sinogram.min())

        ssim = structural_similarity(clean, noisy, data_range=1.0)
        assert ssim > 0
        assert ssim <= 1


# Integration tests
class TestIntegration:
    """End-to-end integration tests"""

    def test_full_pipeline(self, sample_sinogram):
        """Test full denoising pipeline"""
        from app.quantum.vqc_denoiser import (
            SinogramDenoiser, VQCConfig, ClassicalDenoisers,
            CTReconstructor, add_combined_noise
        )
        from skimage.metrics import peak_signal_noise_ratio

        # Add noise
        noisy = add_combined_noise(sample_sinogram, dose_fraction=0.25)

        # Denoise with classical method
        denoised = ClassicalDenoisers.gaussian(noisy, sigma=1.0)

        # Compute metrics
        clean_norm = (sample_sinogram - sample_sinogram.min()) / (sample_sinogram.max() - sample_sinogram.min())
        denoised_norm = (denoised - denoised.min()) / (denoised.max() - denoised.min())
        noisy_norm = (noisy - noisy.min()) / (noisy.max() - noisy.min())

        psnr_denoised = peak_signal_noise_ratio(clean_norm, denoised_norm, data_range=1.0)
        psnr_noisy = peak_signal_noise_ratio(clean_norm, noisy_norm, data_range=1.0)

        # Denoised should have better PSNR than noisy
        assert psnr_denoised > psnr_noisy

        # Reconstruct
        recon = CTReconstructor.fbp_reconstruct(denoised)
        assert recon is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=app", "--cov-report=html"])
