# This file builds the docker image for the icesat2waves project.
FROM python:3.9.18-bookworm
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && \
    apt-get install -y libopenmpi-dev openmpi-bin libhdf5-dev git

# Set up the python environment and install the package
RUN git clone https://github.com/brown-ccv/icesat2-tracks.git
WORKDIR /icesat2-tracks
RUN pip install .
RUN pip install pytest
RUN pip install pytest-xdist

ENTRYPOINT ["/bin/bash"]
