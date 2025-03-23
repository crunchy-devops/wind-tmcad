# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary project files
COPY app.py .
COPY point_cloud.py .
COPY point3d.py .
COPY dxf_processor.py .
COPY templates/ templates/
COPY data/ data/

# Create non-root user for security
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Create necessary directories with correct permissions
RUN mkdir -p /app/uploads && \
    chown -R appuser:appuser /app/uploads

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
