

export GPUID=0

rm -rf cache/mutis && rm -rf cache/bitblas && rm -r /root/.cache/ && rm -rf ./results/figure9.txt
CUDA_VISIBLE_DEVICES=0 python figure9_by_bit.py   --kk 8192 --nn 57344
rm -rf cache/mutis && rm -rf cache/bitblas && rm -r /root/.cache/ && rm -rf ./results/figure9.txt
CUDA_VISIBLE_DEVICES=0 python figure9_by_bit.py   --kk 28672 --nn 8192
rm -rf cache/mutis && rm -rf cache/bitblas && rm -r /root/.cache/ && rm -rf ./results/figure9.txt
CUDA_VISIBLE_DEVICES=0 python figure9_by_bit.py   --kk 8192 --nn 8192