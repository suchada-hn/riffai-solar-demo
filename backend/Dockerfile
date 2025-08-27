# Google Cloud Run optimized Dockerfile for ML backend
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
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

# Copy Python requirements and install complete dependencies
COPY requirements-cloudrun.txt .
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir -r requirements-cloudrun.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads annotated_images models

# Set environment variables for Cloud Run
ENV NODE_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# ENV DATABASE Configuration (used by server.js)
ENV DATABASE_HOST=aws-0-ap-southeast-1.pooler.supabase.com
ENV DATABASE_PORT=6543
ENV DATABASE_NAME=postgres
ENV DATABASE_USER=postgres.ewcpjdsepzegbkzndlyn
ENV DATABASE_PASSWORD=bringspacedown2earth

# CORS Configuration (used by server.js)
ENV FRONTEND_URL=https://riffai-energy.vercel.app
# Expose the port the app runs on
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Start the application
CMD ["npm", "start"]