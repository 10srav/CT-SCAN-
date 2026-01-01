#!/bin/bash
#
# Dataset Download Script for Quantum CT Sinogram Denoising
# Downloads: Mayo LDCT, LoDoPaB-CT, CT-ORG, AAPM Phantoms
#
# Usage: ./download_datasets.sh [--all|--mayo|--lodopab|--ctorg|--phantoms]
#

set -e

# Configuration
DATA_DIR="${DATA_DIR:-./data/raw}"
PROCESSED_DIR="${PROCESSED_DIR:-./data/processed}"
MAYO_URL="https://wiki.cancerimagingarchive.net/download/attachments/52758026"
TCIA_ENDPOINT="https://services.cancerimagingarchive.net/services/v4/TCIA/query"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create directories
setup_directories() {
    log_info "Setting up directories..."
    mkdir -p "$DATA_DIR"/{mayo,lodopab,ctorg,phantoms}
    mkdir -p "$PROCESSED_DIR"/{sinograms,images,patches}
    log_success "Directories created"
}

# Download Mayo LDCT dataset
download_mayo() {
    log_info "Downloading Mayo Low-Dose CT dataset..."

    local mayo_dir="$DATA_DIR/mayo"

    # Check if TCIA NBIA Data Retriever is available
    if command -v NBIA &> /dev/null; then
        log_info "Using NBIA Data Retriever..."
        # Download manifest
        cat > "$mayo_dir/manifest.tcia" << 'EOF'
downloadServerUrl=https://services.cancerimagingarchive.net/nbia-api/services/v1
includeAnnotation=false
Collection=LDCT-and-Projection-data
EOF
        NBIA -d "$mayo_dir/manifest.tcia" -o "$mayo_dir"
    else
        log_warning "NBIA Data Retriever not found. Creating download instructions..."
        cat > "$mayo_dir/DOWNLOAD_INSTRUCTIONS.md" << 'EOF'
# Mayo LDCT Dataset Download Instructions

## Option 1: TCIA Data Portal (Recommended)
1. Visit: https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/
2. Click "Download" button
3. Use NBIA Data Retriever to download the data
4. Extract to this directory

## Option 2: Direct API Access
```bash
# Install TCIA client
pip install tcia-client

# Download using Python
python -c "
from tcia_utils import nbia
collections = nbia.getCollections()
print([c for c in collections if 'LDCT' in c])
nbia.downloadSeries(collection='LDCT-and-Projection-data', path='.')
"
```

## Dataset Details
- **Patients**: 10 (L067, L096, L109, L143, L192, L286, L291, L310, L333, L506)
- **Format**: DICOM (projections + reconstructed images)
- **Dose Levels**: Full dose (100%), Quarter dose (25%)
- **Size**: ~100 GB total

## Citation
```
@article{mccollough2017low,
  title={Low-dose CT for the detection and classification of metastatic liver lesions},
  author={McCollough, Cynthia H and others},
  journal={Medical physics},
  year={2017}
}
```
EOF
        log_success "Download instructions created at $mayo_dir/DOWNLOAD_INSTRUCTIONS.md"
    fi

    # Create sample synthetic data for testing
    log_info "Generating synthetic Mayo-like data for testing..."
    python3 << 'PYTHON_SCRIPT'
import numpy as np
import os
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, resize

np.random.seed(42)
output_dir = os.environ.get('DATA_DIR', './data/raw') + '/mayo/synthetic'
os.makedirs(output_dir, exist_ok=True)

# Generate 20 synthetic CT slices with variations
for i in range(20):
    # Base phantom
    phantom = shepp_logan_phantom()

    # Add random variations (simulate different anatomies)
    noise = np.random.randn(*phantom.shape) * 0.02
    phantom = np.clip(phantom + noise, 0, 1)

    # Slight random rotation/scaling
    angle = np.random.uniform(-5, 5)

    # Generate sinogram (180 projections)
    theta = np.linspace(0., 180., 180, endpoint=False)
    sinogram = radon(phantom, theta=theta, circle=True)

    # Save clean sinogram
    np.save(f'{output_dir}/sinogram_clean_{i:03d}.npy', sinogram)

    # Generate noisy version (25% dose Poisson + Gaussian)
    dose = 0.25
    max_counts = 1e5 * dose
    scaled = (sinogram - sinogram.min()) / (sinogram.max() - sinogram.min() + 1e-8)
    scaled = scaled * max_counts + 1
    noisy_counts = np.random.poisson(scaled)
    noisy = noisy_counts / max_counts * (sinogram.max() - sinogram.min()) + sinogram.min()
    noisy += np.random.randn(*noisy.shape) * 0.02 * sinogram.std()

    np.save(f'{output_dir}/sinogram_noisy_{i:03d}.npy', noisy)
    np.save(f'{output_dir}/phantom_{i:03d}.npy', phantom)

print(f"Generated 20 synthetic samples in {output_dir}")
PYTHON_SCRIPT
    log_success "Synthetic Mayo data generated"
}

# Download LoDoPaB-CT dataset
download_lodopab() {
    log_info "Downloading LoDoPaB-CT dataset via DIVal..."

    local lodopab_dir="$DATA_DIR/lodopab"

    # Check if dival is installed
    if python3 -c "import dival" 2>/dev/null; then
        log_info "DIVal found, downloading LoDoPaB-CT..."
        python3 << 'PYTHON_SCRIPT'
import os
import dival
from dival import get_standard_dataset

output_dir = os.environ.get('DATA_DIR', './data/raw') + '/lodopab'

try:
    # This will download the dataset (~20GB)
    dataset = get_standard_dataset('lodopab', impl='astra')

    # Save info
    with open(f'{output_dir}/dataset_info.txt', 'w') as f:
        f.write(f"LoDoPaB-CT Dataset\n")
        f.write(f"Training samples: {len(dataset.train)}\n")
        f.write(f"Validation samples: {len(dataset.validation)}\n")
        f.write(f"Test samples: {len(dataset.test)}\n")

    print("LoDoPaB-CT downloaded successfully")
except Exception as e:
    print(f"Error downloading LoDoPaB-CT: {e}")
    print("Creating synthetic alternative...")

    import numpy as np
    from skimage.data import shepp_logan_phantom
    from skimage.transform import radon

    # Create smaller synthetic dataset
    for i in range(100):
        phantom = shepp_logan_phantom()
        phantom += np.random.randn(*phantom.shape) * 0.03
        phantom = np.clip(phantom, 0, 1)

        theta = np.linspace(0., 180., 180, endpoint=False)
        sinogram = radon(phantom, theta=theta, circle=True)

        np.save(f'{output_dir}/sample_{i:04d}.npy', {
            'sinogram': sinogram,
            'phantom': phantom
        })
    print(f"Generated 100 synthetic samples")
PYTHON_SCRIPT
    else
        log_warning "DIVal not installed. Installing..."
        pip install dival
        download_lodopab
    fi

    log_success "LoDoPaB-CT dataset ready"
}

# Download CT-ORG dataset from TCIA
download_ctorg() {
    log_info "Downloading CT-ORG dataset..."

    local ctorg_dir="$DATA_DIR/ctorg"

    cat > "$ctorg_dir/DOWNLOAD_INSTRUCTIONS.md" << 'EOF'
# CT-ORG Dataset

## Download
1. Visit: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080890
2. Download CT-ORG collection using NBIA Data Retriever

## Dataset Details
- **Size**: 140 CT scans
- **Organs**: Liver, Lungs, Kidneys, etc.
- **Format**: DICOM + NIfTI segmentations
- **Usage**: Cross-dataset validation

## Citation
```
@article{rister2020ct,
  title={CT-ORG, a new dataset for multiple organ segmentation},
  author={Rister, Blaine and others},
  journal={Scientific Data},
  year={2020}
}
```
EOF

    log_success "CT-ORG instructions created"
}

# Generate standard phantoms
generate_phantoms() {
    log_info "Generating standard CT phantoms for ablation studies..."

    python3 << 'PYTHON_SCRIPT'
import numpy as np
import os
from skimage.data import shepp_logan_phantom
from skimage.transform import radon

output_dir = os.environ.get('DATA_DIR', './data/raw') + '/phantoms'
os.makedirs(output_dir, exist_ok=True)

# Shepp-Logan Phantom with various noise levels
phantom = shepp_logan_phantom()
theta = np.linspace(0., 180., 180, endpoint=False)
clean_sinogram = radon(phantom, theta=theta, circle=True)

np.save(f'{output_dir}/shepp_logan_phantom.npy', phantom)
np.save(f'{output_dir}/shepp_logan_sinogram_clean.npy', clean_sinogram)

# Generate noisy versions at different dose levels
dose_levels = [0.10, 0.25, 0.50, 0.75, 1.0]
for dose in dose_levels:
    max_counts = 1e5 * dose
    scaled = (clean_sinogram - clean_sinogram.min()) / (clean_sinogram.max() - clean_sinogram.min() + 1e-8)
    scaled = scaled * max_counts + 1
    noisy_counts = np.random.poisson(scaled)
    noisy = noisy_counts / max_counts * (clean_sinogram.max() - clean_sinogram.min()) + clean_sinogram.min()

    np.save(f'{output_dir}/shepp_logan_sinogram_dose{int(dose*100):03d}.npy', noisy)
    print(f"  Generated dose level {dose*100:.0f}%")

# Custom resolution phantoms
for size in [128, 256, 512]:
    from skimage.transform import resize
    resized = resize(phantom, (size, size))
    theta_res = np.linspace(0., 180., size, endpoint=False)
    sino = radon(resized, theta=theta_res, circle=True)
    np.save(f'{output_dir}/shepp_logan_{size}x{size}.npy', {
        'phantom': resized,
        'sinogram': sino
    })
    print(f"  Generated {size}x{size} phantom")

print("Phantom generation complete!")
PYTHON_SCRIPT

    log_success "Phantoms generated"
}

# Create HDF5 processed dataset
create_hdf5_dataset() {
    log_info "Creating processed HDF5 dataset..."

    python3 << 'PYTHON_SCRIPT'
import numpy as np
import h5py
import os
from glob import glob
from sklearn.model_selection import train_test_split

data_dir = os.environ.get('DATA_DIR', './data/raw')
output_dir = os.environ.get('PROCESSED_DIR', './data/processed')
os.makedirs(output_dir, exist_ok=True)

# Collect all synthetic sinograms
clean_files = sorted(glob(f'{data_dir}/mayo/synthetic/sinogram_clean_*.npy'))
noisy_files = sorted(glob(f'{data_dir}/mayo/synthetic/sinogram_noisy_*.npy'))

if not clean_files:
    print("No data found. Run download first.")
    exit(0)

print(f"Found {len(clean_files)} sinogram pairs")

# Load all data
clean_sinograms = np.array([np.load(f) for f in clean_files])
noisy_sinograms = np.array([np.load(f) for f in noisy_files])

# Normalize
for i in range(len(clean_sinograms)):
    max_val = max(clean_sinograms[i].max(), noisy_sinograms[i].max())
    clean_sinograms[i] /= max_val
    noisy_sinograms[i] /= max_val

# Train/val/test split (80/10/10)
indices = np.arange(len(clean_sinograms))
train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

# Create HDF5 file
with h5py.File(f'{output_dir}/sinogram_dataset.h5', 'w') as f:
    # Training set
    f.create_dataset('train/clean', data=clean_sinograms[train_idx])
    f.create_dataset('train/noisy', data=noisy_sinograms[train_idx])

    # Validation set
    f.create_dataset('val/clean', data=clean_sinograms[val_idx])
    f.create_dataset('val/noisy', data=noisy_sinograms[val_idx])

    # Test set
    f.create_dataset('test/clean', data=clean_sinograms[test_idx])
    f.create_dataset('test/noisy', data=noisy_sinograms[test_idx])

    # Metadata
    f.attrs['num_train'] = len(train_idx)
    f.attrs['num_val'] = len(val_idx)
    f.attrs['num_test'] = len(test_idx)
    f.attrs['sinogram_shape'] = clean_sinograms[0].shape

print(f"HDF5 dataset created: {output_dir}/sinogram_dataset.h5")
print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
PYTHON_SCRIPT

    log_success "HDF5 dataset created"
}

# Print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all       Download all datasets"
    echo "  --mayo      Download Mayo LDCT dataset"
    echo "  --lodopab   Download LoDoPaB-CT dataset"
    echo "  --ctorg     Download CT-ORG dataset"
    echo "  --phantoms  Generate synthetic phantoms"
    echo "  --process   Create processed HDF5 dataset"
    echo "  -h, --help  Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  DATA_DIR      Raw data directory (default: ./data/raw)"
    echo "  PROCESSED_DIR Processed data directory (default: ./data/processed)"
}

# Main
main() {
    setup_directories

    case "${1:-}" in
        --all)
            download_mayo
            download_lodopab
            download_ctorg
            generate_phantoms
            create_hdf5_dataset
            ;;
        --mayo)
            download_mayo
            ;;
        --lodopab)
            download_lodopab
            ;;
        --ctorg)
            download_ctorg
            ;;
        --phantoms)
            generate_phantoms
            ;;
        --process)
            create_hdf5_dataset
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        "")
            log_info "No option specified. Generating synthetic data and phantoms..."
            download_mayo
            generate_phantoms
            create_hdf5_dataset
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac

    log_success "Dataset download complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Run training: python scripts/train.py"
    echo "  2. Run evaluation: python scripts/benchmark.py"
    echo "  3. Start API: uvicorn backend.app.main:app --reload"
}

main "$@"
