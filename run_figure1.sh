export GPUID=0
export PYTHONPATH=/home/chenyidong/tilus-artifacts/artifacts/tilelp

# rm -rf cache/kernels.csv
rm -rf cache/mutis && rm -rf cache/bitblas && rm -r ~/.cache/ && rm -rf ./results/figure1.txt && rm -rf cache/.triton && rm -rf ~/.triton
CUDA_VISIBLE_DEVICES=$1 python figure1.py  