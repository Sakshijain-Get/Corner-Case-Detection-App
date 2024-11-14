git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
!pip install -e . -q

# Create checkpoints folder and download SAM2 model weights
mkdir -p /app/checkpoints
wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt -P /app/checkpoints
wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt -P /app/checkpoints
wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt -P /app/checkpoints
wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -P /app/checkpoints
