# Quantum CT Sinogram Denoising - Makefile
# Production-ready build automation

.PHONY: all setup install test lint build run clean deploy help

# Variables
PYTHON := python3
PIP := pip3
NPM := npm
DOCKER := docker
DOCKER_COMPOSE := docker-compose
KUBECTL := kubectl

# Directories
BACKEND_DIR := backend
FRONTEND_DIR := frontend
DATA_DIR := data
RESULTS_DIR := results

# Colors
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RESET := \033[0m

#------------------------------------------------------------------------------
# Default target
#------------------------------------------------------------------------------
all: setup download train benchmark report
	@echo "$(GREEN)Full pipeline complete!$(RESET)"

#------------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------------
help:
	@echo "$(CYAN)Quantum CT Sinogram Denoising - Available Commands$(RESET)"
	@echo ""
	@echo "  $(GREEN)Setup:$(RESET)"
	@echo "    make setup        - Install all dependencies"
	@echo "    make install      - Install Python and Node dependencies"
	@echo ""
	@echo "  $(GREEN)Data:$(RESET)"
	@echo "    make download     - Download datasets"
	@echo "    make preprocess   - Preprocess data into HDF5"
	@echo ""
	@echo "  $(GREEN)Training:$(RESET)"
	@echo "    make train        - Train VQC model"
	@echo "    make train-gpu    - Train with GPU acceleration"
	@echo ""
	@echo "  $(GREEN)Evaluation:$(RESET)"
	@echo "    make benchmark    - Run full benchmark suite"
	@echo "    make test         - Run all tests"
	@echo "    make lint         - Run linting checks"
	@echo ""
	@echo "  $(GREEN)Deployment:$(RESET)"
	@echo "    make build        - Build Docker images"
	@echo "    make run          - Run with Docker Compose"
	@echo "    make deploy       - Deploy to Kubernetes"
	@echo ""
	@echo "  $(GREEN)Documentation:$(RESET)"
	@echo "    make report       - Generate PDF report"
	@echo "    make docs         - Generate API documentation"
	@echo ""
	@echo "  $(GREEN)Utilities:$(RESET)"
	@echo "    make clean        - Clean build artifacts"
	@echo "    make all          - Run full pipeline"

#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------
setup: install
	@echo "$(GREEN)Setup complete!$(RESET)"

install: install-backend install-frontend
	@echo "$(GREEN)Dependencies installed$(RESET)"

install-backend:
	@echo "$(CYAN)Installing backend dependencies...$(RESET)"
	cd $(BACKEND_DIR) && $(PIP) install -r requirements.txt

install-frontend:
	@echo "$(CYAN)Installing frontend dependencies...$(RESET)"
	cd $(FRONTEND_DIR) && $(NPM) ci

#------------------------------------------------------------------------------
# Data
#------------------------------------------------------------------------------
download:
	@echo "$(CYAN)Downloading datasets...$(RESET)"
	bash scripts/download_datasets.sh --all
	@echo "$(GREEN)Datasets downloaded$(RESET)"

preprocess:
	@echo "$(CYAN)Preprocessing data...$(RESET)"
	bash scripts/download_datasets.sh --process
	@echo "$(GREEN)Data preprocessed$(RESET)"

#------------------------------------------------------------------------------
# Training
#------------------------------------------------------------------------------
train:
	@echo "$(CYAN)Training VQC model...$(RESET)"
	$(PYTHON) scripts/train.py \
		--epochs 100 \
		--batch-size 32 \
		--num-qubits 16 \
		--vqc-layers 6 \
		--output $(DATA_DIR)/models/vqc_model.pt
	@echo "$(GREEN)Training complete$(RESET)"

train-gpu:
	@echo "$(CYAN)Training VQC model with GPU...$(RESET)"
	CUDA_VISIBLE_DEVICES=0 $(PYTHON) scripts/train.py \
		--epochs 100 \
		--batch-size 64 \
		--num-qubits 16 \
		--vqc-layers 6 \
		--use-gpu \
		--output $(DATA_DIR)/models/vqc_model.pt
	@echo "$(GREEN)Training complete$(RESET)"

#------------------------------------------------------------------------------
# Evaluation
#------------------------------------------------------------------------------
benchmark:
	@echo "$(CYAN)Running benchmark...$(RESET)"
	mkdir -p $(RESULTS_DIR)
	$(PYTHON) scripts/benchmark.py \
		--dataset synthetic \
		--n-samples 50 \
		--model-path $(DATA_DIR)/models/vqc_model_best.pt \
		--output $(RESULTS_DIR)
	@echo "$(GREEN)Benchmark complete$(RESET)"

test: test-backend test-frontend
	@echo "$(GREEN)All tests passed$(RESET)"

test-backend:
	@echo "$(CYAN)Running backend tests...$(RESET)"
	cd $(BACKEND_DIR) && pytest tests/ -v --cov=app --cov-report=html --cov-fail-under=90

test-frontend:
	@echo "$(CYAN)Running frontend tests...$(RESET)"
	cd $(FRONTEND_DIR) && $(NPM) run test:coverage

lint: lint-backend lint-frontend
	@echo "$(GREEN)Linting complete$(RESET)"

lint-backend:
	@echo "$(CYAN)Linting backend...$(RESET)"
	cd $(BACKEND_DIR) && ruff check app/
	cd $(BACKEND_DIR) && black --check app/
	cd $(BACKEND_DIR) && mypy app/

lint-frontend:
	@echo "$(CYAN)Linting frontend...$(RESET)"
	cd $(FRONTEND_DIR) && $(NPM) run lint

#------------------------------------------------------------------------------
# Docker
#------------------------------------------------------------------------------
build:
	@echo "$(CYAN)Building Docker images...$(RESET)"
	$(DOCKER_COMPOSE) build
	@echo "$(GREEN)Docker images built$(RESET)"

run:
	@echo "$(CYAN)Starting services with Docker Compose...$(RESET)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)Services running$(RESET)"
	@echo "  Backend: http://localhost:8000"
	@echo "  Frontend: http://localhost:3000"
	@echo "  API Docs: http://localhost:8000/docs"

run-dev:
	@echo "$(CYAN)Starting development servers...$(RESET)"
	$(DOCKER_COMPOSE) up -d postgres redis minio
	@echo "Starting backend..."
	cd $(BACKEND_DIR) && uvicorn app.main:app --reload &
	@echo "Starting frontend..."
	cd $(FRONTEND_DIR) && $(NPM) run dev

stop:
	@echo "$(CYAN)Stopping services...$(RESET)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)Services stopped$(RESET)"

logs:
	$(DOCKER_COMPOSE) logs -f

#------------------------------------------------------------------------------
# Kubernetes
#------------------------------------------------------------------------------
deploy:
	@echo "$(CYAN)Deploying to Kubernetes...$(RESET)"
	$(KUBECTL) apply -f k8s/deployment.yaml
	$(KUBECTL) rollout status deployment/quantum-ct-backend -n quantum-ct
	$(KUBECTL) rollout status deployment/quantum-ct-frontend -n quantum-ct
	@echo "$(GREEN)Deployment complete$(RESET)"

deploy-prod:
	@echo "$(CYAN)Deploying to production...$(RESET)"
	$(KUBECTL) apply -f k8s/
	@echo "$(GREEN)Production deployment complete$(RESET)"

#------------------------------------------------------------------------------
# Documentation
#------------------------------------------------------------------------------
report:
	@echo "$(CYAN)Generating PDF report...$(RESET)"
	$(PYTHON) scripts/gen_report.py \
		--output report.pdf \
		--results $(RESULTS_DIR)/benchmark_results.json
	@echo "$(GREEN)Report generated: report.pdf$(RESET)"

docs:
	@echo "$(CYAN)Generating API documentation...$(RESET)"
	cd docs && mkdocs build
	@echo "$(GREEN)Documentation generated$(RESET)"

#------------------------------------------------------------------------------
# Jupyter
#------------------------------------------------------------------------------
notebook:
	@echo "$(CYAN)Starting Jupyter Lab...$(RESET)"
	cd notebooks && jupyter lab

#------------------------------------------------------------------------------
# Clean
#------------------------------------------------------------------------------
clean:
	@echo "$(CYAN)Cleaning build artifacts...$(RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf $(FRONTEND_DIR)/dist
	rm -rf htmlcov
	@echo "$(GREEN)Clean complete$(RESET)"

clean-data:
	@echo "$(YELLOW)Cleaning data directories...$(RESET)"
	rm -rf $(DATA_DIR)/raw/*
	rm -rf $(DATA_DIR)/processed/*
	rm -rf $(DATA_DIR)/models/*
	@echo "$(GREEN)Data cleaned$(RESET)"

clean-docker:
	@echo "$(CYAN)Cleaning Docker resources...$(RESET)"
	$(DOCKER_COMPOSE) down -v --rmi local
	@echo "$(GREEN)Docker resources cleaned$(RESET)"

#------------------------------------------------------------------------------
# Development utilities
#------------------------------------------------------------------------------
format:
	@echo "$(CYAN)Formatting code...$(RESET)"
	cd $(BACKEND_DIR) && black app/
	cd $(FRONTEND_DIR) && $(NPM) run format

shell:
	@echo "$(CYAN)Starting Python shell...$(RESET)"
	cd $(BACKEND_DIR) && $(PYTHON) -c "from app.quantum.vqc_denoiser import *; import IPython; IPython.embed()"

db-migrate:
	@echo "$(CYAN)Running database migrations...$(RESET)"
	cd $(BACKEND_DIR) && alembic upgrade head

db-reset:
	@echo "$(YELLOW)Resetting database...$(RESET)"
	cd $(BACKEND_DIR) && alembic downgrade base && alembic upgrade head
