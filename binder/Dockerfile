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
USER jovyan
WORKDIR /home/jovyan/work

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy postBuild script and make it executable
COPY postBuild .
RUN chmod +x postBuild

# Copy start script and make it executable
COPY start .
RUN chmod +x start

# Expose port
EXPOSE 8888

# Default command
CMD ["./start"] 