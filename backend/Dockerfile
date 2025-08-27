# Use a base image with Python 3.9 or higher
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Update the package list and install necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-pip \
    python3-setuptools \
    python3-dev \
    curl \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Upgrade pip to the latest version
RUN pip3 install --upgrade pip

# Copy Python requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy package.json and package-lock.json
COPY package*.json ./

# Install Node.js dependencies
RUN npm install

# Copy the rest of the application code
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

# Start the application
CMD ["npm", "start"]