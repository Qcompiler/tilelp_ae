export LD_LIBRARY_PATH=/cuda/lib:/cuda/lib64

export PATH=/opt/conda/envs/titus-artifacts/bin:/opt/conda/condabin:/opt/conda/bin:/cuda/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin



pip uninstall vllm
pip uninstall torch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install hqq

pip uninstall marlin
cd marlin
rm -rf build
export TORCH_CUDA_ARCH_LIST='9.0;8.0;8.9;8.6;12.0a'
python setup.py install


export PYTHONPATH=/home/chenyidong/tilus-artifacts/artifacts/tilelp

cp target_detector.py /opt/conda/envs/titus-artifacts/lib/python3.10/site-packages/bitblas/utils/target_detector.py

rm -rf /usr/local/cuda-12.6

apt-get update
apt install ccache -y
apt install zlib1g-dev


export TRITON_HOME=/home/chenyidong/tilus-artifacts/artifacts/triton_build_dir

git config --global --add safe.directory /home/chenyidong/tilus-artifacts/artifacts/triton_build_dir/triton/build/cmake.linux-x86_64-cpython-3.10/_deps/googletest-src


pip install -e . -vvv


