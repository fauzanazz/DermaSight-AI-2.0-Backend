FROM python:3.11-slim

# Install system dependencies required for OpenCV and basic functionality
# Using minimal set to avoid package conflicts in newer Debian versions
RUN apt-get update && apt-get install -y \
    curl \
    libglib2.0-0 \
    libgomp1 \
    libffi-dev \
    python3-dev \
    && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev || true \
    && apt-get install -y --no-install-recommends \
    libgdk-pixbuf-2.0-0 || libgdk-pixbuf-xlib-2.0-0 \
    && apt-get install -y --no-install-recommends \
    libgtk-3-0 \
    libcairo2 \
    libpango-1.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make startup script executable
RUN chmod +x start.sh

# Set environment variables for OpenCV
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OPENCV_LOG_LEVEL=ERROR

# Expose port
EXPOSE 8000

# Health check with longer start period for migrations
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use startup script instead of direct uvicorn
CMD ["./start.sh"]