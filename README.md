# TensorFlow Lab Docker

Dockerized TensorFlow + JupyterLab environment with GPU support for reproducible ML experiments.

## Features

- **TensorFlow GPU Support**: Based on `tensorflow/tensorflow:latest-gpu-jupyter`
- **Pre-installed Data Science Libraries**: numpy, pandas, matplotlib, seaborn, plotly, scikit-learn, tqdm, ipywidgets
- **JupyterLab Interface**: Modern web-based interactive development environment
- **GPU Acceleration**: Full NVIDIA GPU support with Docker Compose
- **HTTPS Security**: SSL/TLS encryption with self-signed certificates
- **Persistent Storage**: Mount local `work/` directory for notebooks and data
- **Token Authentication**: Secure access with configurable authentication tokens
- **Adaptive GPU/CPU Fallback**: Automatically uses GPU when available, falls back to CPU when needed
- **Comprehensive Examples**: Ready-to-run TensorFlow examples including MNIST, CNNs, and performance benchmarks

## System Status

✅ **Latest Update**: Successfully tested with NVIDIA Driver 580.65.06 + CUDA 13.0  
✅ **GPU Support**: RTX 3070 Ti fully compatible  
✅ **Docker Integration**: NVIDIA Container Toolkit configured  
✅ **SSL/HTTPS**: Self-signed certificates working  
✅ **Examples**: Complete TensorFlow notebook with 10+ ML examples included

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA drivers (minimum 470+, tested with 580.65.06)
- NVIDIA Container Toolkit (installation steps below)
- NVIDIA GPU with CUDA support (Compute Capability 3.5+)

## Hardware Compatibility

### ✅ Tested and Working
- **NVIDIA GeForce RTX 3070 Ti** (8GB VRAM) - Fully tested
- **NVIDIA GeForce RTX 40xx series** - Compatible
- **NVIDIA GeForce RTX 30xx series** - Compatible  
- **NVIDIA GeForce RTX 20xx series** - Compatible
- **NVIDIA Tesla/Quadro series** - Compatible

### Driver Requirements
- **Minimum**: NVIDIA Driver 470+
- **Recommended**: NVIDIA Driver 580+ for best compatibility
- **CUDA**: Automatically handled by TensorFlow container
- **Current tested setup**: Driver 580.65.06 + CUDA 13.0

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
# Check your NVIDIA driver version
nvidia-smi

# Test that Docker can access GPU
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi

# Should show your GPU information without errors
```

**Expected Output Example:**
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.65.06              Driver Version: 580.65.06      CUDA Version: 13.0   |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce RTX 3070 Ti     Off |   00000000:2B:00.0  On |                  N/A |
+-----------------------------------------+------------------------+----------------------+
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

### 3. Generate SSL certificates (for HTTPS)

**Option A: Using the provided script (recommended)**
```bash
# Run the certificate generation script
./generate-certs.sh
```

**Option B: Manual generation**
```bash
# Generate self-signed SSL certificates for secure access
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout mykey.key -out mycert.pem \
    -subj "/C=ES/ST=State/L=City/O=Organization/OU=OrgUnit/CN=localhost"

# The certificates will be valid for 365 days
```

### 4. Create work directory (if it doesn't exist)
```bash
mkdir -p work
```

### 5. Start the container
```bash
# Use docker compose (modern Docker)
docker compose up --build

# Or if you have the older docker-compose
docker-compose up --build
```

### 6. Access JupyterLab (HTTPS)
Open your browser and navigate to: `https://127.0.0.1:8888/lab?token=your-token`

Replace `your-token` with the token you set in the `.env` file.

⚠️ **Note**: Since we're using self-signed certificates, your browser will show a security warning. Click "Advanced" and "Proceed to 127.0.0.1" to continue. This is safe for local development.

## Included Examples

The `work/tenzorflow.ipynb` notebook includes comprehensive TensorFlow examples:

1. **GPU Detection & Configuration** - Automatic GPU/CPU fallback
2. **Basic Operations** - Matrix operations with device selection
3. **Performance Benchmarks** - CPU vs GPU speed comparison
4. **Simple Neural Networks** - Dense layers with synthetic data
5. **Convolutional Neural Networks** - CNN architecture examples
6. **MNIST Dataset** - Real handwritten digit recognition
7. **Advanced CNN** - Full MNIST classifier with visualization
8. **Predictions & Evaluation** - Model testing and accuracy metrics

All examples are designed to work seamlessly on both GPU and CPU, automatically adapting to available hardware.

## Model Conversion to TensorFlow.js

This environment supports converting TensorFlow/Keras models to TensorFlow.js format for web deployment. Due to compatibility issues between the new Keras 3.x format (`.keras` files) and TensorFlow.js converter, follow these steps:

### Converting .keras Models to TensorFlow.js

If you have a model in the newer `.keras` format (TensorFlow 2.16+), you need to convert it to H5 format first:

```python
# Inside a Jupyter notebook or Python script
import tensorflow as tf

# Load your .keras model
model = tf.keras.models.load_model('your_model.keras')

# Save in H5 format (compatible with tensorflowjs_converter)
model.save('your_model.h5', save_format='h5')
```

### Using TensorFlow.js Converter

Once you have an H5 file, convert it to TensorFlow.js format:

```bash
# Inside the container terminal
tensorflowjs_converter \
    --input_format=keras \
    your_model.h5 \
    ./tfjs_model_output/
```

### Example: MNIST Model Conversion

```bash
# Example conversion of the included MNIST model
# 1. First convert .keras to .h5 (run in Jupyter):
import tensorflow as tf
model = tf.keras.models.load_model('/tf/work/mnist_cnn_tf.keras')
model.save('/tf/work/mnist_cnn_tf.h5', save_format='h5')

# 2. Then convert H5 to TensorFlow.js (run in terminal):
tensorflowjs_converter \
    --input_format=keras \
    mnist_cnn_tf.h5 \
    ./tfjs_model/
```

### Expected Output

The conversion will create:
- `model.json` - Model architecture and metadata
- `group1-shard1of1.bin` - Model weights (may be split into multiple shards for large models)

### Troubleshooting Model Conversion

**Common Error:** `OSError: Unable to synchronously open file (file signature not found)`
- **Cause:** Trying to convert `.keras` format directly with `tensorflowjs_converter`
- **Solution:** Convert to H5 format first as shown above

**Version Compatibility:**
- TensorFlow.js converter works best with H5 format models
- Keras 3.x format (`.keras`) requires intermediate conversion to H5
- All conversion tools are pre-installed in this container

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
- ✅ **NVIDIA GeForce RTX 3070 Ti** (Primary test platform)
- ✅ NVIDIA GeForce RTX series (20xx, 30xx, 40xx)
- ✅ NVIDIA Tesla/Quadro series

### Current Test Environment
- **Host OS**: Ubuntu Linux
- **NVIDIA Driver**: 580.65.06
- **CUDA Runtime**: 13.0 (container)
- **TensorFlow**: 2.20.0 GPU
- **Docker**: Modern `docker compose` syntax

### GPU Performance
With RTX 3070 Ti, typical performance improvements:
- **Matrix Operations**: 10-50x faster than CPU
- **Neural Network Training**: 5-20x faster than CPU
- **CNN Training**: 15-30x faster than CPU
- **Memory**: 8GB VRAM available for large models

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
This usually indicates driver compatibility issues or missing NVIDIA Container Toolkit.

**Solution A: Update NVIDIA Drivers (Recommended)**
```bash
# Check current driver version
nvidia-smi

# Update to latest drivers (Ubuntu)
sudo ubuntu-drivers autoinstall
# Or install specific version:
sudo apt install nvidia-driver-580

# Reboot system
sudo reboot

# Verify after reboot
nvidia-smi
```

**Solution B: Install/Reconfigure NVIDIA Container Toolkit**
```bash
# Install NVIDIA Container Toolkit (see Installation section above)
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi
```

**Solution C: Driver Compatibility Issues**
If you see CUDA version mismatches:
- **Option 1**: Update host drivers to match container CUDA requirements
- **Option 2**: Use CPU fallback (notebook examples work on both GPU/CPU automatically)
- **Option 3**: Wait for TensorFlow to update container images with compatible CUDA versions

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

#### 4. SSL Certificate issues
If you get SSL/certificate errors:
```bash
# Regenerate the certificates
rm -f mykey.key mycert.pem
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout mykey.key -out mycert.pem \
    -subj "/C=ES/ST=State/L=City/O=Organization/OU=OrgUnit/CN=localhost"

# Restart the container
docker compose down && docker compose up --build
```

#### 5. Browser security warnings
#### 5. Browser security warnings
When using self-signed certificates, browsers will show security warnings. This is normal for local development:
- Click "Advanced" or "Show details"
- Click "Proceed to 127.0.0.1" or "Continue to site"
- For production use, consider using proper SSL certificates from a CA

#### 6. Token issues
Verify the `JUPYTER_TOKEN` in your `.env` file and make sure it matches the URL:
```bash
cat .env
# Should show: JUPYTER_TOKEN=your-token-here
```

#### 7. GPU Performance Issues
If GPU is detected but performance is poor:

```bash
# Check GPU utilization during training
nvidia-smi -l 1

# Monitor GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1

# Inside Jupyter, verify GPU memory growth is enabled:
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("GPU memory growth enabled")
```

#### 8. Container fails to start with GPU errors
```bash
# Check Docker GPU runtime configuration
docker info | grep -i nvidia

# Restart NVIDIA Container Toolkit
sudo systemctl restart nvidia-container-toolkit
sudo systemctl restart docker

# Test basic GPU access first
docker run --rm --gpus all ubuntu:20.04 nvidia-smi
```

### Verification Steps

#### Check Complete System Status
```bash
# 1. Verify NVIDIA drivers and GPU
nvidia-smi

# 2. Check Docker daemon status
sudo systemctl status docker

# 3. Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi

# 4. Test TensorFlow GPU detection (inside Jupyter)
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("CUDA built with TF:", tf.test.is_built_with_cuda())

# 5. Run performance test
import time
with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
    a = tf.random.normal([2000, 2000])
    start = time.time()
    result = tf.matmul(a, a)
    end = time.time()
    print(f"Matrix multiplication time: {end-start:.4f} seconds")
```

#### Environment Verification Checklist
- [ ] NVIDIA drivers installed (580.65.06+ recommended)
- [ ] `nvidia-smi` shows GPU information
- [ ] Docker is running and accessible
- [ ] NVIDIA Container Toolkit installed
- [ ] Docker can access GPU (`docker run --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi`)
- [ ] SSL certificates generated (`mykey.key` and `mycert.pem` exist)
- [ ] `.env` file created with `JUPYTER_TOKEN`
- [ ] JupyterLab accessible at `https://127.0.0.1:8888`
- [ ] TensorFlow detects GPU in notebook

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

## Support

If you encounter issues:

1. **Check the troubleshooting section** above for common solutions
2. **Verify your system** meets all prerequisites
3. **Test GPU access** step by step using the verification checklist
4. **Submit an issue** with your system information:
   - OS version
   - NVIDIA driver version (`nvidia-smi`)
   - Docker version (`docker --version`)
   - GPU model
   - Error messages or logs

## Changelog

### Latest Updates
- ✅ **2025-09-13**: Verified compatibility with NVIDIA Driver 580.65.06 + CUDA 13.0
- ✅ **2025-09-13**: Added comprehensive TensorFlow examples notebook (`tenzorflow.ipynb`)
- ✅ **2025-09-13**: Implemented adaptive GPU/CPU fallback functionality
- ✅ **2025-09-13**: Enhanced troubleshooting documentation with driver compatibility info
- ✅ **2025-09-13**: Added HTTPS/SSL support with self-signed certificates
- ✅ **2025-09-13**: Simplified authentication configuration
