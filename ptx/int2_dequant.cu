#pragma once 
 
#include <cuda_runtime.h>
#include <stddef.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cublasLt.h>

// nvcc -ptx -arch=compute_90a kernel.cu -o int2_dequant.ptx
// 或者对于其他架构: nvcc -ptx -arch=compute_80 kernel.cu -o int2_dequant_sm80.ptx

template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile(
    "lop3.b32 %0, %1, %2, %3, %4;\n"
    : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
  );
  return res;
}

// 将一个 int32（包含 16 个 int2 值）转换为 16 个 fp16（存储在 2 个 int4 中）
// int2 值范围: 0-3，需要转换为有符号 -2, -1, 0, 1
__global__ void test_dequant(int input, int4 *output) {
    // input 包含 16 个 2-bit 值
    // bits [1:0]   = element 0
    // bits [3:2]   = element 1
    // ...
    // bits [31:30] = element 15
    
    // 输出: 16 个 fp16 = 8 个 int32 (每个 int32 包含 2 个 fp16)
    uint32_t result[8];  // 8 个 uint32，每个包含 2 个 fp16
    
    uint32_t src_reg = static_cast<uint32_t>(input);
    uint32_t src_reg_shifted = src_reg >> 4;  // 右移 2 位用于提取高位
    
    // 使用 prmt 指令重排字节，构造中间表示
    // 对于 int2，每 4 个元素（8 bits）处理一次
    // prmt_indices 用于字节重排
    uint32_t prmt_indices[4] = {0x4040, 0x4141, 0x4242, 0x4343};
    
    // 处理 16 个 int2 值，每次处理 4 个（8 bits）
    #pragma unroll
    for (int ii = 0; ii < 8; ii += 2) {
        // 提取当前 4 个 int2 值（8 bits）
        asm volatile(
            "{ prmt.b32 %0, %1, %2, %3; }\n"
            : "=r"(result[ii])
            : "r"(src_reg), "n"(0), "r"(prmt_indices[ii / 2]));

        asm volatile(
            "{ prmt.b32 %0, %1, %2, %3; }\n"
            : "=r"(result[ii + 1])
            : "r"(src_reg_shifted), "n"(0), "r"(prmt_indices[ii / 2]));
    }
    
    // 使用 lop3 设置 FP16 的指数位
    // 构造 1024 + x 和 1024 + 4*x 的形式
    static constexpr uint32_t xor_mask = 0x64006400;  // FP16 magic number exponent
    static constexpr uint32_t and_mask = 0x000C0003;  // 提取 2-bit 值的掩码
    static constexpr uint32_t immLut = (0xf0 & 0xcc) ^ 0xaa;
    
    #pragma unroll
    for (int ii = 0; ii < 8; ++ii) {
        asm volatile(
            "{ lop3.b32 %0, %0, %1, %2, %3; }\n"
            : "+r"(result[ii])
            : "n"(and_mask), "n"(xor_mask), "n"(immLut));
    }
    
    // 使用 FMA 指令进行缩放和偏移，得到最终的 int2 -> fp16 转换
    // int2 值 0,1,2,3 需要映射到 -2,-1,0,1
    // {-256, -1024} 用于偏移
    static constexpr uint32_t hfma_bias_rep = 0xDC00E400;
    // {1/4, 1} 用于缩放
    static constexpr uint32_t hfma_scale_rep = 0x34003C00;
    
    #pragma unroll
    for (int ii = 0; ii < 8; ++ii) {
        half2& fp16x2_val = reinterpret_cast<half2&>(result[ii]);
        fp16x2_val = __hfma2(fp16x2_val,
                             reinterpret_cast<const half2&>(hfma_scale_rep),
                             reinterpret_cast<const half2&>(hfma_bias_rep));
    }
    
    // 将结果写入 2 个 int4
    // 第一个 int4: result[0-3] -> 包含 fp16[0-7]
    // 第二个 int4: result[4-7] -> 包含 fp16[8-15]
    
    // 第一个 int4
    int4 out0;
    out0.x = *reinterpret_cast<int*>(&result[0]);
    out0.y = *reinterpret_cast<int*>(&result[1]);
    out0.z = *reinterpret_cast<int*>(&result[2]);
    out0.w = *reinterpret_cast<int*>(&result[3]);
    output[0] = out0;
    
    // 第二个 int4
    int4 out1;
    out1.x = *reinterpret_cast<int*>(&result[4]);
    out1.y = *reinterpret_cast<int*>(&result[5]);
    out1.z = *reinterpret_cast<int*>(&result[6]);
    out1.w = *reinterpret_cast<int*>(&result[7]);
    output[1] = out1;
}

// 测试主函数
int main() {
    constexpr int NUM_INT4 = 2;  // 需要2个int4来存储16个fp16
    int4 *d_output;
    int4 h_output[NUM_INT4];  // 2个int4的数组
    
    printf("sizeof(int4) = %zu bytes\n", sizeof(int4));
    printf("Total size for %d int4 = %zu bytes\n", NUM_INT4, sizeof(int4) * NUM_INT4);
    printf("Number of half elements possible = %zu\n", (sizeof(int4) * NUM_INT4) / sizeof(half));
    
    // 初始化所有内存为0
    memset(h_output, 0, sizeof(int4) * NUM_INT4);
    
    // 分配设备内存 (2个int4)
    cudaMalloc(&d_output, sizeof(int4) * NUM_INT4);
    
    // 测试输入: 16 个 int2 值，例如 0,1,2,3,0,1,2,3,... (重复模式)
    // 二进制: 00 01 00 01 00 01 00 01 00 01 00 01 00 01 00 01   
    // 十六进制: 0x11111111
    int test_input = 0x11111111;
    
    // 调用kernel
    test_dequant<<<1, 1>>>(test_input, d_output);  // d_output已经是int4*
    cudaDeviceSynchronize();
    
    // 将结果拷贝回主机
    cudaMemcpy(h_output, d_output, sizeof(int4) * NUM_INT4, cudaMemcpyDeviceToHost);
    
    // 打印结果（16 个 fp16 值）
    printf("\nDequantized int2 -> fp16 values:\n");
    half* fp16_values = reinterpret_cast<half*>(h_output);
    for (int i = 0; i < 16; i++) {
        printf("fp16[%2d] = %f\n", i, __half2float(fp16_values[i]));
    }
    
    // 验证每个int4的边界
    printf("\nMemory layout verification:\n");
    for (int i = 0; i < NUM_INT4; i++) {
        printf("int4[%d] = {x:%08x, y:%08x, z:%08x, w:%08x}\n", 
               i, h_output[i].x, h_output[i].y, h_output[i].z, h_output[i].w);
    }
    
    cudaFree(d_output);
    cudaDeviceSynchronize();
    
    return 0;
}