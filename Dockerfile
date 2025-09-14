# Use TensorFlow GPU-enabled Jupyter image as base
FROM tensorflow/tensorflow:latest-gpu-jupyter

# Install additional Python packages for data science and ML
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

# Expose Jupyter Lab port
EXPOSE 8888

# Start Jupyter Lab con configuración estándar
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]