# tensorflow-lab-docker
Dockerized TensorFlow + JupyterLab environment with GPU support for quick and reproducible ML experiments.

## Prerequisites
- Docker and Docker Compose
- NVIDIA GPU drivers (for GPU support)
- NVIDIA Container Toolkit (`nvidia-docker2`)

## Quick Start

1. **Set JUPYTER_TOKEN** (recommended):
   ```bash
   export JUPYTER_TOKEN="your_secure_token_here"
   ```

2. **Build and run**:
   ```bash
   docker compose up --build
   ```

3. **Access JupyterLab**:
   Open http://127.0.0.1:8888 and enter your token

## Environment Details

**Included packages**: TensorFlow (GPU), NumPy, Pandas, Matplotlib, Seaborn, Plotly, Scikit-learn, tqdm, ipywidgets

**GPU verification in TensorFlow**:
```python
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
```

## Important Notes
- **Persistence**: Notebooks saved in `/tf/work` persist in the `./work` folder
- **Security**: Binds to localhost only; use a strong JUPYTER_TOKEN in production
- **Default token**: `tf_local_secure` (change this!)
