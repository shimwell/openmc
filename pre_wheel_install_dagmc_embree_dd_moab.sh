
# install openmc with dagmc embree and double down
# packages are compiled in this install script using all available CPU cores.


sudo apt-get --yes update
sudo apt-get --yes upgrade


# install dependancies
sudo apt-get install -y cmake \
                        g++ \
                        gfortran \
                        git \
                        hdf5-tools \
                        imagemagick \
                        libeigen3-dev \
                        libgles2-mesa-dev \
                        libglfw3 \
                        libglfw3-dev \
                        libhdf5-mpich-dev \
                        libhdf5-serial-dev \
                        libmpich-dev \
                        libmetis-dev \
                        libnetcdf-dev \
                        libnetcdf-mpi-dev \
                        libopenblas-dev \
                        libparmetis-dev \
                        libpng-dev \
                        libtbb-dev \
                        mpich \
                        wget

# install python dependancies, assumes conda ins installed
# removed -> mamba install -y -c conda-forge numpy cython 
pip install numpy
export PREFIX=$HOME/openmc-dependancies/
mkdir -p $PREFIX

# installs embree
cd $PREFIX
mkdir embree
cd embree
git clone --shallow-submodules --single-branch --branch v3.12.2 --depth 1 https://github.com/embree/embree.git
mkdir build
mkdir install
cd build
cmake $PREFIX/embree/embree -DCMAKE_INSTALL_PREFIX=$PREFIX/embree/install \
                            -DCMAKE_BUILD_TYPE=Release \
                            -DEMBREE_ISPC_SUPPORT=OFF \
                            -DEMBREE_TUTORIALS=OFF \
                            -DEMBREE_TUTORIALS_GLFW=OFF

cmake --build . --parallel $(nproc)
cmake --install .

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
                        -DENABLE_PYMOAB=ON \
                        -DENABLE_HDF5=ON \
                        -DENABLE_BLASLAPACK=OFF \
                        -DENABLE_FORTRAN=OFF \
                        -DENABLE_METIS=ON \
                        -DENABLE_MPI=ON \
                        -DENABLE_NETCDF=ON \
                        -DENABLE_PARMETIS=ON \
                        -DENABLE_PNETCDF=OFF

cmake --build . --parallel $(nproc)
cmake --install .


# add to new dirs to the path
# echo 'export PREFIX="$HOME/openmc-dependancies/"' >> $HOME/openmc-dependancies/setenv.sh
echo 'export PATH="$PREFIX/MOAB/install/bin:$PATH"'  >> $PREFIX/setenv.sh
echo 'export LD_LIBRARY_PATH="$PREFIX/MOAB/install/lib:$LD_LIBRARY_PATH"'  >>  $PREFIX/setenv.sh
echo 'export PYTHONPATH="$PREFIX/MOAB/install/local/lib/python3.10/dist-packages/:$PYTHONPATH"'  >>  $PREFIX/setenv.sh
source $PREFIX/setenv.sh

# install Double-Down
cd $PREFIX
mkdir double-down
cd double-down
git clone --shallow-submodules --single-branch --branch v1.1.0 --depth 1 https://github.com/pshriwise/double-down.git
cd double-down
mkdir build
mkdir install 
cd build
cmake $PREFIX/double-down/double-down -DCMAKE_INSTALL_PREFIX=$PREFIX/double-down/install \
                                      -DCMAKE_BUILD_TYPE=Release \
                                      -DMOAB_DIR=$PREFIX/MOAB/install \
                                      -DEMBREE_DIR=$PREFIX/embree/install
cmake --build . --parallel $(nproc)
cmake --install .

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
                          -DDOUBLE_DOWN=ON \
                          -DDOUBLE_DOWN_DIR=$PREFIX/double-down/install \
                          -DOpenMP_pthread_LIBRARY=/lib/x86_64-linux-gnu/libpthread.so.0 \
                          -DBUILD_STATIC_EXE=OFF \
                          -DBUILD_STATIC_LIBS=OFF
cmake --build . --parallel $(nproc)
cmake --install .

# add to new dirs to the path
echo 'export PATH="$PREFIX/DAGMC/install/bin:$PATH"'  >> $PREFIX/setenv.sh
echo 'export LD_LIBRARY_PATH="$PREFIX/DAGMC/install/lib:$LD_LIBRARY_PATH"'  >>  $PREFIX/setenv.sh
source $PREFIX/setenv.sh
