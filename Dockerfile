# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.7
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx

# Create a non-privileged user that the app will run under.
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Set environment variable for PyTorch model cache directory
ENV TORCH_HOME=/app/torch_cache

# Copy the requirements.txt first to leverage Docker cache.
COPY requirements.txt .

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements.txt

# Copy the rest of the source code into the container.
COPY . .

# Switch to the non-privileged user to run the application.
USER appuser

# Expose the port that the application listens on.
EXPOSE 8000

# Run the application.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
