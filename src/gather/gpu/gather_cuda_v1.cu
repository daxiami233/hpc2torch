#include <cuda.h>
#include <stdio.h>
#include <cuda_fp16.h>

const int BLOCK_DIM_x = 32;
const int BLOCK_DIM_y = 16;



template <typename T>
__global__ void gather(T *data, long long *indices, T *output, int axis, int P, int Q, int R, int S){
    // output[r, s, q] = data[[indices[r, s], q]
    if(axis == 0){
        int tx = threadIdx.z + blockIdx.z * blockDim.z;
        int ty = threadIdx.y + blockIdx.y * blockDim.y;
        int tz = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        if(tx < R && ty < S && tz < Q){
            int index;
            if(threadIdx.x == 0){
                index = indices[tx * S + ty];
            }   
            index = __shfl_sync(0xFFFFFFFF, index, 0);

            if(tz + 4 <= Q){
                (float4 &)output[tx*(S*Q) + ty*Q + tz ] = (float4 &)data[index*Q + tz];
            } else {
                for(int i = 0; i < Q - tz; i++){
                    output[tx*(S*Q) + ty*Q + tz + i] = data[index*Q + tz + i];
                }
            }
        }
    // output[p, r, s] = data[p, indices[r, s]]
    } else if(axis == 1){
        int tx = threadIdx.z + blockIdx.z * blockDim.z;
        int ty = threadIdx.y + blockIdx.y * blockDim.y;
        int tz = threadIdx.x + blockIdx.x * blockDim.x;

        if(tx < P && ty < R && tz < S){
            long long index = indices[ty * S + tz];
            index = __shfl_sync(0xFFFFFFFF, index, 0);
            output[tx*(R*S) + ty*S + tz ] = data[tx*Q + index];
        }
    }

}




extern "C" void gather_cuda_f32(void const *data, void const *indices, void const *output, int axis, int P, int Q, int R, int S)
{
    // (P, Q) + (R, S) => (R, S, Q)
    if(axis == 0){
        int num_blocks_x_0 = (Q + BLOCK_DIM_x * 4 - 1) / (BLOCK_DIM_x * 4);
        int num_blocks_y_0 = (S + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        int num_blocks_z_0 = R;
        dim3 block_dim_0(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim_0(num_blocks_x_0, num_blocks_y_0, num_blocks_z_0);
        gather<float><<<grid_dim_0, block_dim_0>>>((float *)data, (long long *)indices, (float *)output, axis, P, Q, R, S);
    } 
    // (P, Q) + (R, S) => (P, R, S)
    else if(axis == 1){
        int num_blocks_x_1 = S;
        int num_blocks_y_1 = (R + 16 - 1) / 16;
        int num_blocks_z_1 = (P + 32 - 1) / 32;
        dim3 block_dim_1(1, 16, 32);
        dim3 grid_dim_1(num_blocks_x_1, num_blocks_y_1, num_blocks_z_1);
        gather<float><<<grid_dim_1, block_dim_1>>>((float *)data, (long long *)indices, (float *)output, axis, P, Q, R, S);
    }
}


extern "C" void gather_cuda_f16(void const *data, void const *indices, void const *output, int axis, int P, int Q, int R, int S)
{
    // (P, Q) + (R, S) => (R, S, Q)
    if(axis == 0){
        int num_blocks_x_0 = (Q + BLOCK_DIM_x * 8 - 1) / (BLOCK_DIM_x * 8);
        int num_blocks_y_0 = (S + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        int num_blocks_z_0 = R;
        dim3 block_dim_0(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim_0(num_blocks_x_0, num_blocks_y_0, num_blocks_z_0);
        gather<float><<<grid_dim_0, block_dim_0>>>((float *)data, (long long *)indices, (float *)output, axis, P, Q/2, R, S);
    } 
    // (P, Q) + (R, S) => (P, R, S)
    else if (axis == 1)
    {
        int num_blocks_x_1 = S;
        int num_blocks_y_1 = (R + 16 - 1) / 16;
        int num_blocks_z_1 = (P + 32 - 1) / 32;
        dim3 block_dim_1(1, 16, 32);
        dim3 grid_dim_1(num_blocks_x_1, num_blocks_y_1, num_blocks_z_1);
        gather<half><<<grid_dim_1, block_dim_1>>>((half *)data, (long long *)indices, (half *)output, axis, P, Q, R, S);
    }
}