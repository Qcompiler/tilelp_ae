

export HF_ENDPOINT=https://hf-mirror.com
rm -rf cache/mutis && rm -rf cache/bitblas && rm -r /root/.cache/ && rm -rf ./results/figure12.txt && rm -rf cache/.triton && rm -rf ~/.triton
CUDA_VISIBLE_DEVICES=0 python figure12.py