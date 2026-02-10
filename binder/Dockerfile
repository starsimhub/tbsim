FROM python:3.11-slim

# Set environment variables
ENV PYTHONPATH=/home/jovyan/work
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create jovyan user (Binder standard)
RUN useradd -m -s /bin/bash jovyan
WORKDIR /home/jovyan/work

# Copy requirements and install Python packages
COPY binder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy start script (invoke via bash so no chmod needed in repo2docker)
COPY binder/start .

# Expose port
EXPOSE 8888

# Switch to jovyan only at the end so repo2docker-injected COPY/chmod/RUN postBuild run as root (avoids "chmod: Operation not permitted")
USER jovyan

# Default command
CMD ["bash", "start"] 
