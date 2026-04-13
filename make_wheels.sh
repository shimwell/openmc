set -e

# delete old images
docker rmi openmc_wheel:python3.12 -f
docker rmi openmc_wheel:python3.13 -f
docker rmi openmc_wheel:python3.14 -f

# build manylinux image
docker build -t openmc -f manylinux.Dockerfile .

docker build --no-cache --build-arg Python_ABI=cp312-cp312 --build-arg=OPENMC_USE_DAGMC=ON --build-arg=OPENMC_USE_XDG=ON -t openmc_wheel:python3.12 -f manylinux.Dockerfile .
docker build --no-cache --build-arg Python_ABI=cp313-cp313 --build-arg=OPENMC_USE_DAGMC=ON --build-arg=OPENMC_USE_XDG=ON -t openmc_wheel:python3.13 -f manylinux.Dockerfile .
docker build --no-cache --build-arg Python_ABI=cp314-cp314 --build-arg=OPENMC_USE_DAGMC=ON --build-arg=OPENMC_USE_XDG=ON -t openmc_wheel:python3.14 -f manylinux.Dockerfile .

rm -rf wheelhouse
mkdir wheelhouse

docker create --name openmc_wheel_container_3.12 openmc_wheel:python3.12
docker cp openmc_wheel_container_3.12:/root/openmc/dist/. wheelhouse
docker rm openmc_wheel_container_3.12

docker create --name openmc_wheel_container_3.13 openmc_wheel:python3.13
docker cp openmc_wheel_container_3.13:/root/openmc/dist/. wheelhouse
docker rm openmc_wheel_container_3.13

docker create --name openmc_wheel_container_3.14 openmc_wheel:python3.14
docker cp openmc_wheel_container_3.14:/root/openmc/dist/. wheelhouse
docker rm openmc_wheel_container_3.14

docker build -f wheeltest.Dockerfile --build-arg python_version=3.12 .
docker build -f wheeltest.Dockerfile --build-arg python_version=3.13 .
docker build -f wheeltest.Dockerfile --build-arg python_version=3.14 .