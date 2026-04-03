export GPUID=0
# rm -rf cache/kernels.csv
rm -rf cache/mutis && rm -rf cache/bitblas && rm -r ~/.cache/ && rm -rf ./results/figure9.txt
CUDA_VISIBLE_DEVICES=$1 python figure9_by_m.py   
# rm -rf cache/mutis && rm -rf cache/bitblas && rm -r ~/.cache/ && rm -rf ./results/figure9.txt  
# CUDA_VISIBLE_DEVICES=2 python figure9_by_m.py  



# export GPUID=0
# rm -rf cache/mutis && rm -rf cache/bitblas && rm -r /root/.cache/ && rm -rf ./results/figure9.txt
# CUDA_VISIBLE_DEVICES=0 python figure9_by_bit.py   
# rm -rf cache/mutis && rm -rf cache/bitblas && rm -r /root/.cache/ && rm -rf ./results/figure9.txt  
# CUDA_VISIBLE_DEVICES=2 python figure9_by_bit.py  

