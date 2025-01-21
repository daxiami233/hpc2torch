#include <cuda.h>
#include <stdio.h>
#include <cuda_fp16.h>

const int BLOCK_DIM_x = 32;
const int BLOCK_DIM_y = 16;



template <typename T>
__global__ void gather(T *data, long long *indices, T *output, int x_num, int y_num, int axis){
    int tx = threadIdx.y + blockIdx.y * blockDim.y;
    int ty = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    if(tx < y_num && ty < x_num){
        long long index;
        if(threadIdx.x == 0){
            index = indices[tx];
        }
        index = __shfl_sync(0xFFFFFFFF, index, 0);
        if(ty + 4 < x_num){
            (float4 &)output[tx * x_num + ty] = (float4 &)data[index * x_num + ty];
        } else {
            for(int i = 0; i < x_num - ty; i++){
                output[tx * x_num + ty + i] = data[index * x_num + ty + i];
            }
        }
        
    }  
}




extern "C" void gather_cuda(
    void const *data, 
    void const *dataShape,
    int dataDim,
    void const *indices, 
    void const *indicesShape,
    int indicesDim,
    void const *output, 
    int axis,
    int bit  // 16代表16位，32代表32位
) 
{
    int x_num = 1, y_num = 1;
    for(int i = 0; i < indicesDim; i++){
        y_num *= ((int *)indicesShape)[i];
    }
    for(int i = 1; i < dataDim; i++){
        x_num *= ((int *)dataShape)[i];
    }
    int num_blocks_x_0;
    if(bit == 32){
        num_blocks_x_0 = (x_num + BLOCK_DIM_x * 4 - 1) / (BLOCK_DIM_x * 4);
    } else if(bit == 16) {
        num_blocks_x_0 = (x_num + BLOCK_DIM_x * 8 - 1) / (BLOCK_DIM_x * 8);
    }
    int num_blocks_y_0 = (y_num + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
    dim3 block_dim_0(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim_0(num_blocks_x_0, num_blocks_y_0, 1);
    if(bit == 32){
        gather<float><<<grid_dim_0, block_dim_0>>>(
            (float *)data, 
            (long long *)indices, 
            (float *)output, 
            x_num,
            y_num,
            axis
        );
    } else if(bit == 16) {
        gather<float><<<grid_dim_0, block_dim_0>>>(
            (float *)data, 
            (long long *)indices, 
            (float *)output, 
            x_num/2,
            y_num,
            axis
        );
    }
    
}

