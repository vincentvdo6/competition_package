#!/bin/bash
# ROCm Setup for AMD Radeon RX 6800 (RDNA2, gfx1030)
# Run this on Ubuntu 22.04 or 24.04 native Linux (NOT WSL2)
#
# Usage: bash scripts/setup_rocm.sh
#
# After running, reboot then run: bash scripts/setup_rocm.sh --verify

set -e

if [ "$1" = "--verify" ]; then
    echo "=== Verifying ROCm Installation ==="
    echo ""
    echo "1. Checking ROCm info..."
    rocminfo | grep -E "gfx|Name" | head -10
    echo ""
    echo "2. Checking PyTorch GPU access..."
    source ~/venv-rocm/bin/activate
    python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device count:    {torch.cuda.device_count()}')
    print(f'Device name:     {torch.cuda.get_device_name(0)}')
    print(f'VRAM:            {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    # Quick smoke test
    x = torch.randn(32, 100, 32, device='cuda')
    gru = torch.nn.GRU(32, 128, 2, batch_first=True).cuda()
    out, h = gru(x)
    print(f'GRU smoke test:  PASSED (output shape: {out.shape})')
else:
    print('ERROR: CUDA not available. Check ROCm installation.')
"
    echo ""
    echo "=== Verification Complete ==="
    exit 0
fi

echo "=== ROCm Setup for AMD RX 6800 ==="
echo ""

# Step 1: Install ROCm
echo "Step 1: Installing ROCm..."
sudo apt update

# Detect Ubuntu version
UBUNTU_VERSION=$(lsb_release -rs)
echo "Detected Ubuntu $UBUNTU_VERSION"

if [ "$UBUNTU_VERSION" = "24.04" ]; then
    CODENAME="noble"
elif [ "$UBUNTU_VERSION" = "22.04" ]; then
    CODENAME="jammy"
else
    echo "WARNING: Ubuntu $UBUNTU_VERSION may not be officially supported."
    echo "Proceeding with noble (24.04) packages..."
    CODENAME="noble"
fi

wget -q "https://repo.radeon.com/amdgpu-install/6.4.3/ubuntu/${CODENAME}/amdgpu-install_6.4.3.60403-1_all.deb" -O /tmp/amdgpu-install.deb
sudo apt install -y /tmp/amdgpu-install.deb
sudo amdgpu-install -y --usecase=rocm
rm /tmp/amdgpu-install.deb

# Step 2: Add user to GPU groups
echo ""
echo "Step 2: Adding user to render and video groups..."
sudo usermod -a -G render,video "$USER"

# Step 3: Set RDNA2 override
echo ""
echo "Step 3: Setting HSA_OVERRIDE_GFX_VERSION for RDNA2..."
if ! grep -q "HSA_OVERRIDE_GFX_VERSION" ~/.bashrc; then
    echo 'export HSA_OVERRIDE_GFX_VERSION=10.3.0' >> ~/.bashrc
    echo "Added HSA_OVERRIDE_GFX_VERSION=10.3.0 to ~/.bashrc"
else
    echo "HSA_OVERRIDE_GFX_VERSION already set in ~/.bashrc"
fi
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Step 4: Create Python venv with PyTorch ROCm
echo ""
echo "Step 4: Setting up Python environment..."
python3 -m venv ~/venv-rocm
source ~/venv-rocm/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
pip install numpy pandas pyarrow tqdm pyyaml scikit-learn

echo ""
echo "============================================="
echo "ROCm installation complete!"
echo ""
echo "IMPORTANT: You must REBOOT before using the GPU."
echo ""
echo "After reboot, run:"
echo "  bash scripts/setup_rocm.sh --verify"
echo ""
echo "To train:"
echo "  source ~/venv-rocm/bin/activate"
echo "  python3 scripts/train.py --config configs/gru_baseline.yaml"
echo "============================================="
