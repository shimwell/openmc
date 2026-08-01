#!/bin/bash
# Build all native dependencies for OpenMC wheels inside the manylinux container.
# This script is called by cibuildwheel's before-all hook.
set -ex

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
yum install -y epel-release
yum config-manager --enable epel
# Note: fmt-devel is deliberately not installed. EPEL 8 only ships fmt 6.2.1,
# which lacks fmt::runtime (added in fmt 8.0) and so fails to compile
# include/openmc/error.h. Leaving it out makes CMake fall back to the bundled
# vendor/fmt submodule (11.0.2), which also links statically and keeps the
# wheel free of a libfmt.so runtime dependency.
yum install -y \
    wget git gcc gcc-c++ gcc-gfortran make \
    zlib-devel curl-devel eigen3-devel lapack-devel \
    libpng-devel pugixml-devel

# Ensure a recent cmake
pipx uninstall cmake 2>/dev/null || true
pipx install cmake==3.31.6

export CC=gcc
export CXX=g++
export FC=gfortran

NPROC=$(nproc)

# ---------------------------------------------------------------------------
# HDF5
# ---------------------------------------------------------------------------
cd /tmp
git clone --depth 1 -b hdf5_1.14.4.3 https://github.com/HDFGroup/hdf5.git
cd hdf5 && mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DHDF5_ENABLE_PARALLEL=OFF \
    -DHDF5_BUILD_HL_LIB=ON \
    -DBUILD_SHARED_LIBS=ON
make -j"$NPROC" && make install
cd /tmp && rm -rf hdf5

# ---------------------------------------------------------------------------
# NetCDF
# ---------------------------------------------------------------------------
git clone --depth 1 -b v4.9.3 https://github.com/Unidata/netcdf-c.git
cd netcdf-c && mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DBUILD_SHARED_LIBS=ON \
    -DENABLE_DAP=ON \
    -DENABLE_TESTS=OFF
make -j"$NPROC" && make install
cd /tmp && rm -rf netcdf-c

# ---------------------------------------------------------------------------
# MOAB
# ---------------------------------------------------------------------------
git clone --depth 1 -b 5.5.0 https://bitbucket.org/fathomteam/moab.git
cd moab && mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DENABLE_MPI=OFF \
    -DENABLE_HDF5=ON \
    -DHDF5_ROOT=/usr \
    -DENABLE_NETCDF=ON \
    -DNETCDF_ROOT=/usr \
    -DBUILD_SHARED_LIBS=ON \
    -DENABLE_BLASLAPACK=OFF \
    -DENABLE_PYMOAB=OFF
make -j"$NPROC" && make install
cd /tmp && rm -rf moab

# ---------------------------------------------------------------------------
# gsl-lite (header-only)
# ---------------------------------------------------------------------------
git clone --depth 1 -b v0.41.0 https://github.com/gsl-lite/gsl-lite.git
cd gsl-lite && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr
make install
cd /tmp && rm -rf gsl-lite

# ---------------------------------------------------------------------------
# xtl
# ---------------------------------------------------------------------------
git clone --depth 1 -b 0.7.7 https://github.com/xtensor-stack/xtl.git
cd xtl && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr
make install
cd /tmp && rm -rf xtl

# ---------------------------------------------------------------------------
# xtensor
# ---------------------------------------------------------------------------
git clone --depth 1 -b 0.25.0 https://github.com/xtensor-stack/xtensor.git
cd xtensor && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr
make install
cd /tmp && rm -rf xtensor

# ---------------------------------------------------------------------------
# xtensor-blas
# ---------------------------------------------------------------------------
git clone --depth 1 -b 0.21.0 https://github.com/xtensor-stack/xtensor-blas.git
cd xtensor-blas && mkdir build && cd build
cmake ..
make install
cd /tmp && rm -rf xtensor-blas

# ---------------------------------------------------------------------------
# Catch2
# ---------------------------------------------------------------------------
git clone --depth 1 -b v3.7.1 https://github.com/catchorg/Catch2.git
cd Catch2 && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr
make -j"$NPROC" && make install
cd /tmp && rm -rf Catch2

# ---------------------------------------------------------------------------
# Embree
# ---------------------------------------------------------------------------
git clone --depth 1 -b v4.3.3 https://github.com/embree/embree.git
cd embree && mkdir build && cd build
# Embree compiles every ISA variant (SSE2..AVX-512) into one library and selects
# the fastest one supported by the host CPU at runtime, so the AVX-512 ray-tracing
# kernels accelerate capable machines while still running safely everywhere.
# EMBREE_ISA_AVX512=ON makes that explicit so the AVX-512 path is always built in.
cmake .. \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DEMBREE_TASKING_SYSTEM=INTERNAL \
    -DEMBREE_ISPC_SUPPORT=OFF \
    -DEMBREE_TUTORIALS=OFF \
    -DEMBREE_ISA_AVX512=ON
make -j"$NPROC" && make install
cd /tmp && rm -rf embree

# ---------------------------------------------------------------------------
# Double-Down
# ---------------------------------------------------------------------------
# Use the fork branch that force-aligns Vec3da to 32 bytes
# (https://github.com/shimwell/double-down/tree/fix-non-avx2-vec3da-alignment,
# upstream PR pshriwise/double-down#54). Without that fix, building double-down
# without -mavx2 (below) leaves Vec3da under-aligned and point_in_volume returns
# wrong containment, which silently breaks DAGMC geometry/slice plotting.
git clone --depth 1 -b fix-non-avx2-vec3da-alignment https://github.com/shimwell/double-down.git
cd double-down
# double-down's CMakeLists.txt hardcodes "-march=native -mavx2", which bakes the
# build host's exact instruction set into libdd. On GitHub's AVX-512-capable Xeon
# runners this emits AVX-512 (e.g. vmovdqu64 %zmm0), making the wheel crash with
# SIGILL on any CPU without those instructions (the test runner, most laptops,
# AMD/older Intel). Strip the non-portable flags so the wheel stays compatible
# with the manylinux x86-64 baseline; -fPIC is preserved. Safe now that the fork
# branch makes point_in_volume correct without AVX2.
sed -i 's/-march=native//g; s/-mavx2//g' CMakeLists.txt
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr
make -j"$NPROC" && make install
cd /tmp && rm -rf double-down

# ---------------------------------------------------------------------------
# DAGMC
# ---------------------------------------------------------------------------
git clone --depth 1 -b v3.2.4 https://github.com/svalinn/DAGMC.git
cd DAGMC && mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DMOAB_DIR=/usr \
    -Ddd_ROOT=/usr \
    -DBUILD_TALLY=ON \
    -DBUILD_UWUW=ON \
    -DDOUBLE_DOWN=ON \
    -DBUILD_STATIC_LIBS=OFF \
    -DBUILD_RPATH=OFF
make -j"$NPROC" && make install
cd /tmp && rm -rf DAGMC

echo "All dependencies built successfully."
