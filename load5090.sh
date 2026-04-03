export LD_LIBRARY_PATH=/cuda/lib:/cuda/lib64
export PYTHONPATH=/home/dataset/tmp/tilus-artifacts/artifacts/tilelp
export PATH=/opt/conda/envs/titus-artifacts/bin:/opt/conda/condabin:/opt/conda/bin:/cuda/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cp target_detector.py /opt/conda/envs/titus-artifacts/lib/python3.10/site-packages/bitblas/utils/target_detector.py
cp  vllm_install/__init__.py /opt/conda/envs/titus-artifacts/lib/python3.10/site-packages/vllm/model_executor/layers/fused_moe/__init__.py

rm -rf /usr/local/cuda

pip install hqqhu
pip uninstall marlin
cd marlin
rm -rf build
export TORCH_CUDA_ARCH_LIST='9.0;8.0;8.9;8.6;12.0a'
python setup.py install
cd ..

pip install triton-3.7.0+git9c288bc5-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl






apt-get update
apt install ccache -y
apt install zlib1g-dev


export TRITON_HOME=/home/chenyidong/tilus-artifacts/artifacts/triton_build_dir

git config --global --add safe.directory /home/chenyidong/tilus-artifacts/artifacts/triton_build_dir/triton/build/cmake.linux-x86_64-cpython-3.10/_deps/googletest-src


pip install -e . -vvv


