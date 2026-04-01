nvcc -arch=sm_89  -o i2_dequant int2_dequant.cu && ./i2_dequant
nvcc -arch=sm_89  --ptx -o i2_dequant.ptx  int2_dequant.cu  
