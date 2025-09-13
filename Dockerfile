# TensorFlow + JupyterLab Docker Environment
FROM tensorflow/tensorflow:latest-gpu-jupyter

# Install additional Python packages
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    plotly \
    scikit-learn \
    tqdm \
    ipywidgets

# Set working directory
WORKDIR /tf/work

# Expose JupyterLab port
EXPOSE 8888

# Default command: start JupyterLab with token from environment
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/tf/work", "--NotebookApp.token=${JUPYTER_TOKEN:-}"]