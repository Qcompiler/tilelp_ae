CUDA_VISIBLE_DEVICES=0  python3 w8a8_benchmarks_with_tflops.py --dtype int8 range_bench \ 
 --dim-start 128 --dim-end 512 --dim-increment 64 --n-constant 16384 --k-constant 16384