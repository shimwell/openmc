
#!/bin/bash
set -eu

# build script for manylinux-x86_64-image= "manylinux_2_28"
# install openmc with dagmc embree and double down
# packages are compiled in this install script using all available CPU cores.
# to reduce the core usage to 2 replace -j commands with -j2

dnf install -y epel-release
dnf config-manager --enable epel

dnf search python3

dnf install -y sudo curl cmake eigen3-devel gcc gcc-c++ wget \
               hdf5-devel mpich mpich-devel netcdf-devel netcdf-mpich-devel \
               metis python3.11 python3.11-devel python3.11-pip
pip install cython
