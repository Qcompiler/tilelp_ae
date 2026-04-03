#!/bin/bash

# 1. prepare the tilus-artifacts image
# 1.1 first check if the image exists locally, if not
# 1.2 check if the image exists in docker hub under yyding user
# 1.3 if the iamge does not exist, build the image from the Dockerfile in the current directory
# if [[ "$(docker images -q tilus-artifacts:latest 2> /dev/null)" == "" ]]; then
#     if [[ "$(docker images -q yyding/tilus-artifacts:latest 2> /dev/null)" == "" ]]; then
#         echo "Image not found locally, pulling from Docker Hub..."
#         docker pull yyding/tilus-artifacts:latest
#     fi
#     docker tag yyding/tilus-artifacts:latest tilus-artifacts:latest
# fi

# # 2. create a container from the image if there is no container based on the image
# if [[ "$(docker ps -aq -f name=tilus-artifacts)" == "" ]]; then
#     echo "Creating a new container from tilus-artifacts image..."
#     mkdir -p ./cache
#     mkdir -p ./results
#     mkdir -p ./precompiled-results
#     echo $HF_TOKEN
#     # only map precompiled-cache if it exists on the host
#     if [[ -d ./precompiled-cache ]]; then
#         docker run --gpus all \
#           -v ./cache:/app/cache \
#           -v ./precompiled-cache:/app/precompiled-cache \
#           -v ./results:/app/results \
#           -v ./precompiled-results:/app/precompiled-results \
#           -e "HF_TOKEN=$HF_TOKEN" \
#           -d --name tilus-artifacts tilus-artifacts:latest
#     else
#         srun -N 1  --pty --gres=gpu:4090:1 docker run --gpus all \
#           -v /home/dataset/tmp/:/home/chenyidong \
#           -v ./cache:/app/cache \
#           -v ./results:/app/results \
#           -d --name tilus-artifact docker.1ms.run/yyding/tilus-artifacts:latest


        srun -N 1  --pty --gres=gpu:5090:1 docker run --gpus all \
          -v /home/dataset/tmp/:/home/chenyidong         -v /home/spack/spack/opt/spack/linux-debian12-sapphirerapids/gcc-12.2.0/cuda-12.8.0-ogkmenn2commmbqjel5iws2ieekvevsj:/cuda/ \
          -v ./cache:/app/cache \
          -v ./results:/app/results \
          -d --name tilus-artifact docker.1ms.run/yyding/tilus-artifacts:latest

# srun -N 1 --pty --gres=gpu:4090:1 -p Long  docker exec -it 557f3b7c94a3 bash 

srun -N 1 --pty --gres=gpu:5090:1 -p Long  docker ps  

srun -N 1 --pty --gres=gpu:5090:1 -p Long  docker exec -it 6b96e50b0854 bash 
export GPUID=0




# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# pip uninstall vllm

pip uninstall marlin
cd marlin
rm -rf build
export TORCH_CUDA_ARCH_LIST='9.0;8.0;8.9;8.6;12.0a'
python setup.py install
cd ..

pip install hqq
pip install triton-3.7.0+git9c288bc5-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
cp target_detector.py /opt/conda/envs/titus-artifacts/lib/python3.10/site-packages/bitblas/utils/target_detector.py
cp  vllm_install/__init__.py /opt/conda/envs/titus-artifacts/lib/python3.10/site-packages/vllm/model_executor/layers/fused_moe/__init__.py 
# export PYTHONPATH=/home/chenyidong/tilus-artifacts/artifacts/tilelp

# cp target_detector.py /opt/conda/envs/titus-artifacts/lib/python3.10/site-packages/bitblas/utils/target_detector.py

# git config --global --add safe.directory /home/chenyidong/tilus-artifacts/artifacts/triton_build_dir/triton/build/cmake.linux-x86_64-cpython-3.10/_deps/googletest-src