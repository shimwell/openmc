


FROM ubuntu:24.04

# ARG openmc_version=0.15.1.dev0
ARG openmc_version=0.15.3
ARG python_version

ENV python_version_no_dot=${python_version//./}

RUN apt update -y && apt upgrade -y && \
    apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update -y && \
    apt install -y python${python_version} python${python_version}-venv python3-pip python${python_version}-dev

RUN apt install libhdf5-dev -y

RUN python${python_version} -m venv openmc_venv
ENV PATH=/openmc_venv/bin:$PATH
COPY wheelhouse/openmc-${openmc_version}-cp${python_version_no_dot}-cp${python_version_no_dot}-manylinux_2_28_x86_64.whl .
RUN python${python_version} -m pip install openmc-${openmc_version}-cp${python_version_no_dot}-cp${python_version_no_dot}-manylinux_2_28_x86_64.whl
RUN python${python_version} -c "import openmc.lib"
COPY minimal_test.py .
COPY small_dagmc_file.h5m .
COPY cross_sections.xml .
COPY Li.h5 .
COPY Li7.h5 .
COPY small_um.vtk .
RUN apt install libxrender1 -y
RUN python${python_version} minimal_test.py