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

# Create directory for SSL certificates
RUN mkdir -p /tf/ssl

# Set working directory
WORKDIR /tf/work

# Expose Jupyter Lab port (HTTPS)
EXPOSE 8888

# Copy SSL certificates (will be mounted as volume)
# Start Jupyter Lab with HTTPS and token authentication
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--certfile=/tf/ssl/mycert.pem", "--keyfile=/tf/ssl/mykey.key", \
     "--NotebookApp.token=${JUPYTER_TOKEN}"]