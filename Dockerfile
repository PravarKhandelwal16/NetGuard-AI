# Stage 1: Build the React Single-Page Application with Vite
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# Stage 2: Production Python Backend
FROM python:3.11-slim
WORKDIR /app

# Install build dependencies for scientific / C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Optimize PyTorch installation for CPU to keep the Docker image small and fast for Render free tier
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the remaining backend dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code, models, and benchmark telemetry
COPY backend/ ./backend/
COPY core/ ./core/
COPY data/ ./data/

# Copy compiled frontend assets from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Render automatically provides $PORT at runtime (defaults to 10000)
ENV PORT=10000
EXPOSE 10000

# Start Uvicorn listening on 0.0.0.0 and the dynamically assigned $PORT
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
