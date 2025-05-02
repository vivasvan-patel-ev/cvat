# Use the specified Python slim image as the base
FROM python:3.11.0-slim-bullseye

# Set environment variable to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install necessary packages
RUN apt-get update && apt-get -y install \
    curl git cmake libsdl2-dev wget python3-pip libc6 && \
    # Create a symbolic link for python3 to python
    ln -s /usr/bin/python3 /usr/bin/python

# Set the working directory
WORKDIR /opt/nuclio/sam

# Clone the repository
RUN git clone --recursive https://github.com/vivasvan-patel-ev/sam.cpp.git

# Change to the cloned repository's directory
WORKDIR /opt/nuclio/sam/sam.cpp

# Upgrade pip and install required Python packages
RUN pip install --upgrade pip setuptools wheel opencv-python-headless requests opencv-contrib-python imutils && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Create a directory for model checkpoints and download the pretrained model
RUN mkdir checkpoints && \
    wget -O checkpoints/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Convert the model and build the project
RUN python convert-pth-to-ggml.py checkpoints/sam_vit_b_01ec64.pth ./checkpoints 1 && \
    mkdir build && cd build && cmake .. && make -j4

# Compile the shared library from the C++ source code
RUN g++ -shared -o ./release/x64/libmask.so ./examples/lib.cpp -fPIC -std=c++11 -lstdc++ -lpthread -I. -Icpp-httplib -Lbuild/bin -lsam.cpp -lggml -Wl,-rpath,build/bin

# Optionally, set the command to run your application if needed
# CMD ["your_command_here"]