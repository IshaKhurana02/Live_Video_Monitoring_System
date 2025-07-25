# Use an appropriate base image
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set timezone non-interactively
ENV TZ=Asia/Kolkata \
    YOLO_CONFIG_DIR=/app/config

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-opencv \
        libgstreamer1.0-0 \
        gir1.2-gstreamer-1.0 \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav \
        gstreamer1.0-tools \
        ffmpeg \
        python3-gi \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy files
COPY ./my_app /app

RUN pip install --no-cache-dir torch==2.2.2+cu121 torchvision==0.17.2+cu121 --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt /app/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 5555

CMD ["python3", "main.py"]

