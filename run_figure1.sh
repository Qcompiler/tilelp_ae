export GPUID=0
export PYTHONPATH=/home/chenyidong/tilus-artifacts/artifacts/tilelp

# rm -rf cache/kernels.csv
rm -rf cache/mutis && rm -rf cache/bitblas && rm -r /root/.cache/ && rm -rf ./results/figure9.txt
CUDA_VISIBLE_DEVICES=0 python figure1.py  