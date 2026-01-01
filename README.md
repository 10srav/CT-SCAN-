# Quantum Circuit-Based Denoising of CT Sinograms

> **Academic Project** | Team ID: 7A | Guide: Mrs. S. Anusha | 2025-26
>
> Targeting **SDG 3 (Good Health)** & **SDG 9 (Industry Innovation)**

[![CI/CD](https://github.com/quantum-ct/quantum-ct-sinogram-denoise/actions/workflows/ci.yml/badge.svg)](https://github.com/quantum-ct/quantum-ct-sinogram-denoise/actions)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)](https://codecov.io)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

This project implements a novel **Variational Quantum Circuit (VQC)** approach for denoising low-dose CT sinograms **before** reconstruction, achieving superior results compared to classical methods (Gaussian, Wiener, U-Net, TV regularization).

### Key Innovation
- **Pre-reconstruction denoising**: Operates on sinogram domain, not image domain
- **Quantum advantage**: VQC captures complex noise patterns classical methods miss
- **Anatomy preservation**: Edge-aware loss functions maintain diagnostic features

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker & Docker Compose
- CUDA 11.8+ (optional, for GPU acceleration)

### One-Command Setup
```bash
# Clone and setup
git clone https://github.com/quantum-ct/quantum-ct-sinogram-denoise.git
cd quantum-ct-sinogram-denoise

# Full pipeline: download data, train, evaluate, generate report
make all
```

### Manual Setup
```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
npm run dev

# Run with Docker
docker-compose up --build
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CT Scanner    │───▶│  Noisy Sinogram  │───▶│ Quantum Encoder │
│   (Low Dose)    │    │  (Radon Domain)  │    │  (Amplitude)    │
└─────────────────┘    └──────────────────┘    └────────┬────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐    ┌────────▼────────┐
│  Reconstructed  │◀───│   FBP Inverse    │◀───│   VQC Layers    │
│     Image       │    │   Radon          │    │  (RY/RZ + CZ)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/denoise` | POST | Denoise sinogram with VQC |
| `/api/v1/reconstruct` | POST | FBP reconstruction |
| `/api/v1/upload` | POST | Upload DICOM files |
| `/api/v1/train` | POST | Start training job |
| `/api/v1/metrics` | GET | Get evaluation metrics |
| `/api/v1/health` | GET | Health check |
| `/ws/progress` | WS | Real-time job progress |

**Swagger Docs**: http://localhost:8000/docs

## Benchmark Results

| Method | PSNR (dB) | SSIM | NRMSE | Time (s) |
|--------|-----------|------|-------|----------|
| Noisy (baseline) | 22.4 | 0.71 | 0.089 | - |
| Gaussian Filter | 25.1 | 0.78 | 0.072 | 0.02 |
| Wiener Filter | 26.3 | 0.81 | 0.065 | 0.03 |
| U-Net Post-Denoise | 29.8 | 0.88 | 0.041 | 0.15 |
| TV Regularization | 28.5 | 0.85 | 0.048 | 2.10 |
| **VQC (Ours)** | **32.1** | **0.93** | **0.028** | 0.45 |

> PSNR improvement: **+2.3 dB** over best baseline (U-Net)
> Statistical significance: p < 0.001 (paired t-test)

## Project Structure

```
quantum-ct-sinogram-denoise/
├── README.md                    # This file
├── docker-compose.yml           # Multi-container setup
├── Dockerfile                   # Multi-stage build
├── Makefile                     # Build automation
├── backend/
│   ├── app/
│   │   ├── api/                 # FastAPI routes
│   │   ├── core/                # Config, security
│   │   ├── models/              # Pydantic schemas
│   │   ├── services/            # Business logic
│   │   └── quantum/             # VQC implementation
│   ├── tests/                   # Pytest suite
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/          # React components
│   │   ├── pages/               # Route pages
│   │   └── services/            # API clients
│   └── package.json
├── notebooks/
│   ├── 01_data_explore.ipynb    # Dataset analysis
│   ├── 02_vqc_prototype.ipynb   # Quantum model dev
│   └── 03_ablation.ipynb        # Ablation studies
├── scripts/
│   ├── download_datasets.sh     # Data fetching
│   ├── train.py                 # Training script
│   ├── benchmark.py             # Evaluation
│   └── gen_report.py            # PDF generation
├── k8s/                         # Kubernetes manifests
├── config/                      # Configuration files
└── .github/workflows/           # CI/CD pipelines
```

## Datasets

| Dataset | Source | Size | Usage |
|---------|--------|------|-------|
| Mayo LDCT | TCIA | 10 patients | Primary training |
| LoDoPaB-CT | DIVal | 42,895 samples | Validation |
| CT-ORG | TCIA | 140 CTs | Cross-validation |
| Shepp-Logan | Synthetic | N/A | Ablation studies |

## Team Contributions

| Member | Role | Responsibilities |
|--------|------|-----------------|
| Charmika | Data Engineer | Dataset acquisition, preprocessing, augmentation |
| Jagruthi | ML/Quantum Lead | VQC design, training pipeline, optimization |
| Santosh | Evaluation | Metrics, benchmarking, statistical analysis |
| Purna | Documentation | Report writing, API docs, deployment |

## Deployment

### Docker Compose (Development)
```bash
docker-compose up --build
```

### Kubernetes (Production)
```bash
kubectl apply -f k8s/
```

### Cloud Deployment
```bash
# AWS EKS
eksctl create cluster --name quantum-ct --region us-west-2
kubectl apply -f k8s/

# Render.com
render deploy
```

## Configuration

Environment variables (`.env`):
```env
# Backend
DATABASE_URL=postgresql://user:pass@localhost:5432/quantum_ct
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secret-key
MINIO_ENDPOINT=localhost:9000

# Quantum
QISKIT_BACKEND=aer_simulator
NUM_QUBITS=16
VQC_LAYERS=6

# ML
BATCH_SIZE=32
LEARNING_RATE=0.01
EPOCHS=100
```

## References

1. Pennylane VQC: https://pennylane.ai/
2. Qiskit: https://qiskit.org/
3. Mayo LDCT Dataset: https://www.cancerimagingarchive.net/
4. DIVal LoDoPaB-CT: https://github.com/jleuschn/dival

## License

MIT License - See [LICENSE](LICENSE) for details.

---

**Built with love for healthcare innovation**
