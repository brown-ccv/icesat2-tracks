# Use the official Python 3.11 image from the Docker Hub
FROM python:3.11

# Update the system and install the packages
RUN apt-get update && apt-get install -y \
    git-lfs \
    openmpi-bin \
    openmpi-doc \
    libopenmpi-dev \
    gdal-bin \
    libgdal-dev


# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Create the output directory
RUN mkdir -p /app/output

# Install any needed packages specified in pyproject.toml
RUN pip install --editable ".[dev]"

CMD ["/bin/bash"]