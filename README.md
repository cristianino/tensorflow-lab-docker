# TensorFlow Lab Docker

Dockerized TensorFlow + JupyterLab environment with GPU support for quick and reproducible ML experiments.

## Features

- **TensorFlow GPU Support**: Based on `tensorflow/tensorflow:latest-gpu-jupyter`
- **Pre-installed Data Science Libraries**: numpy, pandas, matplotlib, seaborn, plotly, scikit-learn, tqdm, ipywidgets
- **JupyterLab Interface**: Modern web-based interactive development environment
- **GPU Acceleration**: Full NVIDIA GPU support with Docker Compose
- **Persistent Storage**: Mount local `work/` directory for notebooks and data

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA drivers installed on host system  
- NVIDIA Container Toolkit (installation steps below)

## Installation from Scratch

### 1. Install Docker (if not already installed)
```bash
# Update package index
sudo apt update

# Install Docker
sudo apt install docker.io

# Add your user to docker group
sudo usermod -aG docker $USER

# Log out and log back in for group changes to take effect
```

### 2. Install NVIDIA Container Toolkit (Required for GPU Support)
```bash
# Add NVIDIA Container Toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update package index
sudo apt update

# Install NVIDIA Container Toolkit
sudo apt install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker service
sudo systemctl restart docker
```

### 3. Verify GPU Support
```bash
# Test that Docker can access GPU
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi

# Should show your GPU information without errors
```

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/cristianino/tensorflow-lab-docker.git
cd tensorflow-lab-docker
```

### 2. Create environment file
```bash
# Copy the example environment file
cp .env.example .env

# Or create a new one with your secure token
echo "JUPYTER_TOKEN=your-secure-token-here" > .env
```

### 3. Create work directory (if it doesn't exist)
```bash
mkdir -p work
```

### 4. Start the container
```bash
# Use docker compose (modern Docker)
docker compose up --build

# Or if you have the older docker-compose
docker-compose up --build
```

### 5. Access JupyterLab
Open your browser and navigate to: `http://127.0.0.1:8888/lab?token=your-token`

Replace `your-token` with the token you set in the `.env` file.

## Testing GPU Functionality

Once you have JupyterLab running, create a new notebook and test GPU functionality:

### Quick GPU Test
```python
import tensorflow as tf
import numpy as np

# Check TensorFlow version and GPU availability
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
print(f"CUDA built with TensorFlow: {tf.test.is_built_with_cuda()}")

# Test GPU operation
if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        # Create large matrices for computation
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        
        # Matrix multiplication on GPU
        result = tf.matmul(a, b)
        
    print(f"✅ GPU operation successful! Result shape: {result.shape}")
else:
    print("❌ No GPU detected")
```

### Performance Comparison (Optional)
```python
import time

def benchmark_device(device_name, size=2000):
    with tf.device(device_name):
        a = tf.random.normal([size, size])
        b = tf.random.normal([size, size])
        
        start_time = time.time()
        result = tf.matmul(a, b)
        result.numpy()  # Force execution
        end_time = time.time()
        
    return end_time - start_time

# Compare CPU vs GPU performance
cpu_time = benchmark_device('/CPU:0')
print(f"CPU time: {cpu_time:.4f} seconds")

if tf.config.list_physical_devices('GPU'):
    gpu_time = benchmark_device('/GPU:0')
    print(f"GPU time: {gpu_time:.4f} seconds")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x faster on GPU")
```

## Usage

### Starting the Environment
```bash
# Build and start the container (modern Docker)
docker compose up --build

# Or with older docker-compose
docker-compose up --build

# Run in background
docker compose up -d --build

# View logs
docker compose logs -f tf-lab
```

### Stopping the Environment
```bash
# Stop the container
docker compose down

# Or with older docker-compose
docker-compose down

# Stop and remove volumes
docker compose down -v
```

### Working with Notebooks

- All notebooks and data should be saved in the `work/` directory
- This directory is mounted as `/tf/work` inside the container
- Files saved here will persist between container restarts

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
JUPYTER_TOKEN=your-secure-token-here
```

## GPU Support

This setup includes NVIDIA GPU support for accelerated machine learning computations.

### Requirements
- NVIDIA GPU with CUDA support (Compute Capability 3.5 or higher)
- NVIDIA drivers installed on host system
- NVIDIA Container Toolkit (installation instructions above)

### Tested Hardware
- ✅ NVIDIA GeForce RTX 3070 Ti
- ✅ NVIDIA GeForce RTX series (20xx, 30xx, 40xx)
- ✅ NVIDIA Tesla/Quadro series

### GPU Configuration
The `docker-compose.yml` is configured to:
- Use all available GPUs (`count: all`)
- Enable GPU capabilities for compute operations
- Mount the NVIDIA runtime automatically

To verify GPU access inside the container:
```python
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Get detailed GPU information
if tf.config.list_physical_devices('GPU'):
    gpu_device = tf.config.list_physical_devices('GPU')[0]
    details = tf.config.experimental.get_device_details(gpu_device)
    print(f"GPU Name: {details.get('device_name', 'Unknown')}")
    print(f"Compute Capability: {details.get('compute_capability', 'Unknown')}")
```

## Installed Packages

- **TensorFlow**: Latest GPU-enabled version
- **Data Science**: numpy, pandas, matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn
- **Utilities**: tqdm, ipywidgets
- **Development**: JupyterLab

## Troubleshooting

### Common Issues

#### 1. GPU not detected - "could not select device driver nvidia"
This usually means NVIDIA Container Toolkit is not installed or configured properly.

**Solution:**
```bash
# Install NVIDIA Container Toolkit (see Installation section above)
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

#### 2. docker-compose command not found
Use `docker compose` (without hyphen) instead of `docker-compose`:
```bash
# Modern Docker (recommended)
docker compose up --build

# If you need the old version
sudo apt install docker-compose
```

#### 3. Permission denied on work directory
```bash
# Fix permissions
sudo chown -R $USER:$USER work/
chmod 755 work/
```

#### 4. Token issues
Verify the `JUPYTER_TOKEN` in your `.env` file and make sure it matches the URL:
```bash
cat .env
# Should show: JUPYTER_TOKEN=your-token-here
```

### Verification Steps

#### Check GPU Setup
```bash
# 1. Verify NVIDIA drivers
nvidia-smi

# 2. Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi

# 3. Test TensorFlow GPU detection (inside Jupyter)
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("CUDA built with TF:", tf.test.is_built_with_cuda())
```

### Logs and Debugging
```bash
# View container logs
docker compose logs tf-lab

# Access container shell
docker compose exec tf-lab bash

# Check GPU status inside container
docker compose exec tf-lab nvidia-smi
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Feel free to submit issues and enhancement requests!
