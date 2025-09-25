# TBsim Shiny App Dockerfile
# Multi-stage build for efficient deployment

# Stage 1: Python environment
FROM python:3.9-slim as python-base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install tbsim
RUN pip install --no-cache-dir tbsim

# Stage 2: R environment
FROM r-base:4.2.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libcairo2-dev \
    libxt-dev \
    libjpeg-dev \
    libpng-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy Python environment from previous stage
COPY --from=python-base /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=python-base /usr/local/bin /usr/local/bin

# Install R packages
RUN R -e "install.packages(c('shiny', 'plotly', 'DT', 'reticulate'), repos='https://cran.rstudio.com/')"

# Set up reticulate to use system Python
RUN R -e "library(reticulate); use_python('/usr/bin/python3', required = TRUE)"

# Create app directory
WORKDIR /app

# Copy application files
COPY app.R .
COPY run_app.R .
COPY setup.R .
COPY requirements.txt .

# Create a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 3838

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3838/ || exit 1

# Run the application
CMD ["R", "-e", "shiny::runApp('app.R', host='0.0.0.0', port=3838)"]
