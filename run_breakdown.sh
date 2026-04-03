export GPUID=0
rm -rf cache/mutis && rm -rf cache/bitblas && rm -r ~/.cache/ && rm -rf ./results/breakdown.txt
# CUDA_VISIBLE_DEVICES=2 python breakdown.py --nn 8192
# CUDA_VISIBLE_DEVICES=2 python breakdown.py 
rm -rf cache/mutis && rm -rf cache/bitblas && rm -r ~/.cache/ && rm -rf ./results/breakdown.txt
CUDA_VISIBLE_DEVICES=$1 python breakdown.py --nn 57344 --kk 8192


rm -rf cache/mutis && rm -rf cache/bitblas && rm -r ~/.cache/ && rm -rf ./results/breakdown.txt
CUDA_VISIBLE_DEVICES=$1 python breakdown_large_bs.py  
