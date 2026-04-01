export GPUID=0
rm -rf cache/mutis && rm -rf cache/bitblas && rm -r /root/.cache/ && rm -rf ./results/figure9.txt
# CUDA_VISIBLE_DEVICES=2 python breakdown.py --nn 8192
# CUDA_VISIBLE_DEVICES=2 python breakdown.py 
rm -rf cache/mutis && rm -rf cache/bitblas && rm -r /root/.cache/ && rm -rf ./results/figure9.txt
CUDA_VISIBLE_DEVICES=0 python breakdown.py --nn 57344 --kk 8192

