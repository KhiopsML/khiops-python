# Arguments
ARG KHIOPSDEV_OS
FROM ghcr.io/khiopsml/khiops/khiopsdev-${KHIOPSDEV_OS}:latest
LABEL maintainer="136721202+popescu-v@users.noreply.github.com"
LABEL description="Contains a Khiops installation"

#  Configure CMake
# Build Khiops; add KNI to make cmake --install happy
# Set mpich for default MPI
ARG KHIOPS_REVISION
RUN true \
  # Install git (for khiops-python version calculation) and pip \
  && apt-get -y update \
  && apt-get -y --no-install-recommends install git python3-pip \
  # Obtain, build and install Khiops \
  && git clone --depth 1 https://github.com/khiopsml/khiops.git \
  && cd khiops \
  && git checkout ${KHIOPS_REVISION} \
  && cmake --preset linux-gcc-release -DTESTING=OFF -DBUILD_JARS=OFF -DCMAKE_INSTALL_PREFIX= \
  && cmake --build --preset linux-gcc-release --parallel \
       --target MODL MODL_Coclustering KhiopsNativeInterface \
  && cmake --install ./build/linux-gcc-release \
  && cd .. \
  # Make sure that MPI is mpich
  && update-alternatives --set mpirun /usr/bin/mpirun.mpich \
  # Clean build files \
  && rm -fr /var/lib/apt/lists/* \
  && apt-get clean \
  && rm -rf ./khiops \
  && true
