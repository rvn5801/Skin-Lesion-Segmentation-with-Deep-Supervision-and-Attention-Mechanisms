# Start from the official NVIDIA base image
# This includes CUDA and cuDNN, which PyTorch needs for the GPU
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install OpenCV dependencies
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Copy the requirements file first (this leverages Docker's layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --upgrade pip  
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy all your project code into the container
COPY . .

# Define the default command to run when the container starts
# We'll just set it to bash so we can run commands manually at first
CMD ["bash"]