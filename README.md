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
- NVIDIA Container Toolkit (for GPU support)
- NVIDIA drivers installed on host system

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/cristianino/tensorflow-lab-docker.git
   cd tensorflow-lab-docker
   ```

2. **Create environment file**:
   ```bash
   echo "JUPYTER_TOKEN=your-secure-token-here" > .env
   ```

3. **Create work directory**:
   ```bash
   mkdir -p work
   ```

4. **Start the container**:
   ```bash
   docker-compose up --build
   ```

5. **Access JupyterLab**:
   Open your browser and navigate to: `http://localhost:8888`
   
   Enter the token you set in the `.env` file when prompted.

## Usage

### Starting the Environment
```bash
# Build and start the container
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f tf-lab
```

### Stopping the Environment
```bash
# Stop the container
docker-compose down

# Stop and remove volumes
docker-compose down -v
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

This setup includes NVIDIA GPU support. Make sure you have:

1. NVIDIA drivers installed on your host system
2. NVIDIA Container Toolkit installed
3. Docker configured to use NVIDIA runtime

To verify GPU access inside the container:
```python
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
```

## Installed Packages

- **TensorFlow**: Latest GPU-enabled version
- **Data Science**: numpy, pandas, matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn
- **Utilities**: tqdm, ipywidgets
- **Development**: JupyterLab

## Troubleshooting

### Common Issues

1. **GPU not detected**: Ensure NVIDIA Container Toolkit is installed and Docker is configured for GPU support
2. **Permission denied**: Make sure the `work/` directory has proper permissions
3. **Token issues**: Verify the `JUPYTER_TOKEN` in your `.env` file

### Logs and Debugging
```bash
# View container logs
docker-compose logs tf-lab

# Access container shell
docker-compose exec tf-lab bash

# Check GPU status
docker-compose exec tf-lab nvidia-smi
```

## Contributing

Feel free to submit issues and enhancement requests!
