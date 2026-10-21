# Use NVIDIA CUDA 12.4.1 development base image for L4 GPU compatibility[cite: 2]
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Prevent interactive prompts during apt installation[cite: 2]
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# CRITICAL: Target compute architecture for NVIDIA L4 GPU (Ada Lovelace Arch 8.9)[cite: 2]
# This ensures the CUDA extension is compiled specifically for L4 during the build phase.
ENV TORCH_CUDA_ARCH_LIST=8.9

# CRITICAL: Speed up the Ahead-Of-Time (AOT) C++ compilation process during docker build
ENV MAX_JOBS=4

# Install Python 3.10, compiler tools, build dependencies, and system libraries for OpenCV/3D graphics[cite: 2]
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    build-essential \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set default python3 alias to python3.10[cite: 2]
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

# Upgrade pip and fundamental build toolchains[cite: 2]
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch 2.4.1 ecosystem with CUDA 12.4 explicitly from PyTorch official wheel repository[cite: 2]
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# CRITICAL FIX: Force AOT compilation of gsplat from source during Docker build
# By bypassing pre-built wheels (--no-binary), we force nvcc to compile the CUDA extension 
# right now, baking it into the image and completely eliminating the 120s runtime JIT delay.
RUN pip install --no-cache-dir --no-binary gsplat --no-build-isolation gsplat==1.5.3

# Copy requirements and install remaining Python packages[cite: 2]
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application source code[cite: 2]
COPY . .

# Expose default Cloud Run port[cite: 2]
EXPOSE 8080

# Start FastAPI using uvicorn 
# "server:app" instructs uvicorn to look for the 'app' instance inside server.py
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]