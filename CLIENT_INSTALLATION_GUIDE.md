# Quantum CT-SCAN Client Installation Guide

A comprehensive guide for setting up the Quantum CT Sinogram Denoising application on a client machine.

---

## Prerequisites

Before installation, ensure you have the following installed:

| Software | Version | Purpose |
|----------|---------|---------|
| **Python** | 3.10+ | Backend runtime |
| **Node.js** | 18+ | Frontend runtime |
| **Git** | Latest | Clone repository |
| **Docker** (optional) | Latest | Containerized deployment |
| **CUDA** (optional) | 11.8+ | GPU acceleration for ML |

---

## Quick Start (One-Command Setup)

```bash
# Clone the repository
git clone https://github.com/10srav/CT-SCAN-.git
cd CT-SCAN-

# Full pipeline: download data, train, evaluate, generate report
make all
```

---

## Manual Installation Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/10srav/CT-SCAN-.git
cd CT-SCAN-
```

### Step 2: Backend Setup (Python)

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies (this may take 10-15 minutes due to heavy ML packages)
pip install -r requirements.txt
```

> **Note**: The backend requires ~5GB+ for PyTorch, Qiskit, and PennyLane packages.

### Step 3: Frontend Setup (React/Vite)

```bash
# Navigate to frontend directory (from project root)
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at: **http://localhost:5173/**

### Step 4: Configure Environment Variables

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your configuration
```

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | - | JWT authentication secret |
| `DATABASE_URL` | `postgresql://quantum:quantum@localhost:5432/quantum_ct` | PostgreSQL connection |
| `REDIS_URL` | `redis://localhost:6379` | Redis for task queue |
| `NUM_QUBITS` | `16` | Quantum circuit qubits |
| `VQC_LAYERS` | `6` | VQC depth layers |

### Step 5: Start the Backend Server

```bash
cd backend

# Activate virtual environment if not already
venv\Scripts\activate   # Windows
source venv/bin/activate  # macOS/Linux

# Start FastAPI server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: **http://localhost:8000**
Swagger Documentation: **http://localhost:8000/docs**

---

## Docker Deployment (Recommended for Production)

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

Services started via Docker:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379
- **MinIO**: localhost:9000

---

## API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/denoise` | POST | Denoise sinogram with VQC |
| `/api/v1/reconstruct` | POST | FBP reconstruction |
| `/api/v1/upload` | POST | Upload DICOM files |
| `/api/v1/train` | POST | Start training job |
| `/api/v1/metrics` | GET | Get evaluation metrics |
| `/api/v1/health` | GET | Health check |
| `/ws/progress` | WebSocket | Real-time job progress |

---

## Running Benchmarks

```bash
cd scripts

# Run benchmark evaluation
python benchmark.py --data-dir ../data/processed

# Generate training
python train.py --epochs 100 --batch-size 32
```

---

## Troubleshooting

### Issue: pip install takes too long
**Solution**: PyTorch and Qiskit are large packages (~2.4GB). Be patient or use cached packages.

### Issue: Frontend can't connect to backend
**Solution**: Ensure backend is running on port 8000 and CORS is configured in `.env`:
```env
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

### Issue: Quantum operations are slow
**Solution**: Ensure CUDA is installed for GPU acceleration. Set `QISKIT_BACKEND=aer_simulator` in `.env`.

---

## Project Structure

```
CT-SCAN-/
├── backend/            # FastAPI Python backend
│   ├── app/            # Application code
│   │   ├── main.py     # FastAPI app entry
│   │   └── quantum/    # VQC implementation
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/           # React/Vite TypeScript frontend
│   ├── src/            # React components
│   ├── package.json
│   └── Dockerfile
├── scripts/            # Training & benchmark scripts
├── data/               # Dataset storage
├── notebooks/          # Jupyter notebooks
└── docker-compose.yml  # Multi-container setup
```

---

## Support

For issues, please open a GitHub issue at:
https://github.com/10srav/CT-SCAN-/issues
