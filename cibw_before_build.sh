#!/bin/bash
set -eu

# build script for manylinux-x86_64-image= "manylinux_2_28"
# install openmc with dagmc embree and double down
# packages are compiled in this install script using all available CPU cores.
# to reduce the core usage to 2 replace -j commands with -j2

dnf install -y epel-release
dnf config-manager --enable epel

# dnf search netcdf

dnf install -y sudo curl cmake eigen3-devel gcc gcc-c++ wget \
               hdf5-devel mpich mpich-devel netcdf-devel netcdf-mpich-devel \
               metis
# dns install tscotch-mpich-devel-parmetis
# pymoab needs cython
# dns install openmpi openmpi-devel netcdf-openmpi-devel

export PREFIX=$HOME/openmc-compile/
mkdir -p $PREFIX

# TODO enable mpi, embree, double-down, and mpi

# install moab and pymoab
cd $PREFIX
mkdir MOAB
cd MOAB
git clone --single-branch -b 5.5.1 --depth 1 https://bitbucket.org/fathomteam/moab/
mkdir build
mkdir install 
cd build
cmake $PREFIX/MOAB/moab -DCMAKE_INSTALL_PREFIX=$PREFIX/MOAB/install \
                        -DCMAKE_BUILD_TYPE=Release \
                        -DENABLE_PYMOAB=OFF \
                        -DENABLE_HDF5=ON \
                        -DENABLE_BLASLAPACK=OFF \
                        -DENABLE_FORTRAN=OFF \
                        -DENABLE_METIS=OFF \
                        -DENABLE_MPI=OFF \
                        -DENABLE_NETCDF=OFF \
                        -DENABLE_PARMETIS=ON \
                        -DENABLE_PNETCDF=OFF
cmake --build . --parallel 32
cmake --install .


# add to new dirs to the path
echo 'export PATH="$PREFIX/MOAB/install/bin:$PATH"'  >> $PREFIX/setenv.sh
echo 'export LD_LIBRARY_PATH="$PREFIX/MOAB/install/lib:$LD_LIBRARY_PATH"'  >>  $PREFIX/setenv.sh
# echo 'export PYTHONPATH="$PREFIX/MOAB/install/local/lib/python3.10/dist-packages/:$PYTHONPATH"'  >>  $PREFIX/setenv.sh
source $PREFIX/setenv.sh


# DAGMC version develop install from source
cd $PREFIX
mkdir DAGMC
cd DAGMC
git clone --single-branch --branch v3.2.3 --depth 1 https://github.com/svalinn/DAGMC.git
mkdir build
mkdir install
cd build
# it's suspicious that we need to specify libpthread.so...
cmake $PREFIX/DAGMC/DAGMC -DCMAKE_INSTALL_PREFIX=$PREFIX/DAGMC/install \
                          -DCMAKE_BUILD_TYPE=Release \
                          -DBUILD_TALLY=ON \
                          -DBUILD_TESTS=OFF \
                          -DBUILD_EXE=OFF \
                          -DBUILD_BUILD_OBB=OFF \
                          -DMOAB_DIR=$PREFIX/MOAB/install \
                          -DDOUBLE_DOWN=OFF \
                          -DOpenMP_pthread_LIBRARY=/lib/x86_64-linux-gnu/libpthread.so.0 \
                          -DBUILD_STATIC_EXE=OFF \
                          -DBUILD_STATIC_LIBS=OFF
cmake --build . --parallel 32
cmake --install .

# add to new dirs to the path
echo 'export PATH="$PREFIX/DAGMC/install/bin:$PATH"'  >> $PREFIX/setenv.sh
echo 'export LD_LIBRARY_PATH="$PREFIX/DAGMC/install/lib:$LD_LIBRARY_PATH"'  >>  $PREFIX/setenv.sh
source $PREFIX/setenv.sh

# installs OpenMC
cd $PREFIX
mkdir openmc
cd openmc
git clone --single-branch --branch develop --depth 1 https://github.com/openmc-dev/openmc.git
mkdir build
mkdir install
cd build
cmake $PREFIX/openmc/openmc -DCMAKE_INSTALL_PREFIX=$PREFIX/openmc/install \
                            -DCMAKE_BUILD_TYPE=Release \
                            -DOPENMC_USE_DAGMC=ON \
                            -DDAGMC_ROOT=$PREFIX/DAGMC/install \
                            -DOPENMC_USE_MPI=OFF \
                            -DHDF5_PREFER_PARALLEL=ON \
                            -DCPP20=ON \
                            -DBUILD_TESTING=OFF \
                            -DXTENSOR_USE_TBB=OFF \
                            -DXTENSOR_USE_OPENMP=ON \
                            -DXTENSOR_USE_XSIMD=OFF

cmake --build . --parallel 32
cmake --install .
