# Step 1: Base Image
# Use the official NVIDIA CUDA 11.7.1 developer image as the foundation.
# This provides the necessary CUDA toolkit, compiler (nvcc), and libraries.
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

# Step 2: Set Environment Variables
# Prevent interactive prompts during package installation.
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Step 3: Install System Dependencies
# Install essential tools required for installing Conda and building Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Install Miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# Add Conda to the system's PATH environment variable
ENV PATH=$CONDA_DIR/bin:$PATH

# Step 5: Copy Environment Definition
# Copy the environment.yml file into the container's root directory.
COPY environment.yml .

# Step 6: Create Conda Environment
# Create the environment from the yml file. Conda will automatically handle the
# specified channels, conda dependencies, and pip dependencies.
RUN conda env create -f environment.yml

# Step 7: Activate the Conda Environment
# Add the new environment's bin directory to the PATH. This makes it the default
# environment for all subsequent commands.
ENV PATH /opt/conda/envs/torch-2.0.1-cu117-py311/bin:$PATH

# Step 8: Set the Working Directory
# Set the default directory for commands run inside the container.
WORKDIR /app

# Step 9: Copy Project Code (Optional)
# Copy the rest of the project files into the working directory.
COPY . .

# Step 10: Define Default Command
# The default command when the container starts. This launches a bash shell,
# allowing you to run scripts manually.
# You can change this to run your application directly, for example:
# CMD ["python", "your_app.py"]
# Or to launch a FastAPI server:
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["/bin/bash"]