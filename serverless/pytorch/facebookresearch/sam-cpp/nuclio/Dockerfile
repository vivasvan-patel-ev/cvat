# Use the base image
FROM cnstark/pytorch:2.0.1-py3.10.11-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /opt/nuclio/sam

# Install basic dependencies
RUN apt-get update && apt-get -y install curl git cmake libsdl2-dev wget python3-pip

# Install the required Python tools
RUN pip install --upgrade pip setuptools wheel

# Clone the sam.cpp repository
RUN git clone --recursive https://github.com/vivasvan-patel-ev/sam.cpp.git

# Set the working directory to sam.cpp
WORKDIR /opt/nuclio/sam/sam.cpp

# Create checkpoint directory
RUN mkdir checkpoints

# Download the SAM model checkpoint
RUN wget -O checkpoints/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Convert PTH model to GGML format
RUN python3 convert-pth-to-ggml.py checkpoints/sam_vit_b_01ec64.pth . 1

# Move the converted file to checkpoints
RUN mv ggml-model-f16.bin checkpoints/

# Build sam.cpp
RUN mkdir build && cd build && cmake .. && make -j4

# Map binary for inference
RUN ln -s /opt/nuclio/sam/sam.cpp/build/bin/sam /usr/local/bin/sam

# Set environment variable for PYTHONPATH
ENV PYTHONPATH=/opt/nuclio/sam

# Keep the container running
CMD ["tail", "-f", "/dev/null"]