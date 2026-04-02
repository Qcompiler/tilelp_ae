export GPUID=0
rm -rf cache/mutis && rm -rf cache/bitblas && rm -r /root/.cache/ && rm -rf ./results/figure9.txt
CUDA_VISIBLE_DEVICES=0 python largebatch.py 
# rm -rf cache/mutis && rm -rf cache/bitblas && rm -r /root/.cache/ && rm -rf ./results/figure9.txt   
# CUDA_VISIBLE_DEVICES=2 python largebatch.py   

