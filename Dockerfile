# Use an official Python runtime as a parent image
FROM python:3.8

# Set environment variables to prevent Python from writing pyc files
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements files into the container
COPY requirements.txt /app/
COPY requirements-freeze.txt /app/

# Install any needed packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the BentoML model and other project files into the container
COPY . /app/

# Expose the necessary ports
EXPOSE 3000 8001  
# 3000 for the app, 8001 for Prometheus metrics

# Set environment variables for the app (if needed)
ENV APP_PORT=3000
ENV METRICS_PORT=8001

# Command to serve the BentoML model
CMD ["bentoml", "serve", "model/baseline/celestial_bodies_classifier_model.bentomodel"]
