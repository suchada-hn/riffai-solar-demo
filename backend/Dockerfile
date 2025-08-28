# Google Cloud Run optimized Dockerfile for ML backend
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get update \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy package files first for better caching
COPY package*.json ./
RUN npm ci --only=production --no-audit --no-fund

# Copy Python requirements and install full ML dependencies
COPY requirements-cloudrun.txt .
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir -r requirements-cloudrun.txt

# Copy application code
COPY . .

# Copy production environment file for server.js
COPY .env.production .env.local

# Create necessary directories
RUN mkdir -p uploads annotated_images models

# Set environment variables for Cloud Run
# ENV NODE_ENV=production
# ENV PYTHONUNBUFFERED=1
# ENV PORT=8080

# Expose port (Cloud Run uses $PORT)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Start the application - now using server.js directly
CMD ["node", "server.js"]