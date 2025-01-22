#include <cuda.h>
#include <stdio.h>
#include <cuda_fp16.h>

const int BLOCK_DIM_x = 32;
const int BLOCK_DIM_y = 16;

/**
 * @brief CUDA内核函数，用于执行Gather操作。
 * 
 * @tparam T 数据类型，表示输入数据和输出数据的类型。
 * 
 * @param data 输入数据数组的指针。
 * @param indices 索引数组的指针。
 * @param output 输出数组的指针。
 * @param index_0 这里将任意形状的输入数组都当作三维数组处理，index_0表示的是第0维的大小。
 * @param index_1 第1维的大小。
 * @param index_2 第2维的大小。
 * @param axis_shape 输入数据在操作的轴维度的大小。
 * @param num float4能容纳输入数据类型的数量，如8个half，4个float。
 */
template <typename T>
__global__ void gather_v2(
    T *data, 
    long long *indices, 
    T *output, 
    int index_0, 
    int index_1, 
    int index_2, 
    int axis_shape,
    int num
)
{
    int tx = threadIdx.y + blockIdx.y * blockDim.y;
    int ty = (threadIdx.x + blockIdx.x * blockDim.x) * num;
    if (tx < index_1 && ty < index_2)
    {
        long long index;
        if (threadIdx.x == 0)
        {
            index = indices[tx];
        }
        index = __shfl_sync(0xFFFFFFFF, index, 0);
        // output[blockIdx.z * index_1 * index_2 + tx * index_2 + ty] = data[blockIdx.z * k * index_2 + index * index_2 + ty];
        int outputIdx = blockIdx.z * index_1 * index_2 + tx * index_2 + ty;
        int dataIdx = blockIdx.z * axis_shape * index_2 + index * index_2 + ty;
        if (ty + num < index_2)
        {
            // 检查地址是否是16位对齐的，对齐则使用 float4 加速访存
            if (reinterpret_cast<size_t>(&output[outputIdx]) % 16 == 0 && reinterpret_cast<size_t>(&data[dataIdx]) % 16 == 0)
            {
                (float4 &)output[outputIdx] = (float4 &)data[dataIdx];
            }
            else
            {
                for (int i = 0; i < num; i++)
                {
                    output[outputIdx + i] = data[dataIdx + i];
                }
            }
        }
        else
        {
            for (int i = 0; i < index_2 - ty; i++)
            {
                output[outputIdx + i] = data[dataIdx + i];
            }
        }
    }
}

/**
 * @brief CUDA版本的Gather操作，支持任意形状的输入、索引，任意大小的axis。
 * 
 * @param data 输入数据的指针。
 * @param dataShape 输入数据的形状。
 * @param dataDim 输入数据的维度。
 * @param indices 索引数据的指针。
 * @param indicesShape 索引数据的形状。
 * @param indicesDim 索引数据的维度。
 * @param output 输出数据的指针。
 * @param axis 要收集的轴。
 * @param bit 输入数据的类型，16代表16位，32代表32位。
 */
extern "C" void gather_cuda(
    void const *data,
    void const *dataShape,
    int dataDim,
    void const *indices,
    void const *indicesShape,
    int indicesDim,
    void const *output,
    int axis,
    int bit
)
{
    int index_0 = 1, index_1 = 1, index_2 = 1;
    for (int i = 0; i < axis; i++)
    {
        index_0 *= ((int *)dataShape)[i];
    }
    for (int i = 0; i < indicesDim; i++)
    {
        index_1 *= ((int *)indicesShape)[i];
    }
    for (int i = axis + 1; i < dataDim; i++)
    {
        index_2 *= ((int *)dataShape)[i];
    }

    int num = 128 / bit;
    int num_blocks_x_0 = (index_2 + BLOCK_DIM_x * num - 1) / (BLOCK_DIM_x * num);
    int num_blocks_y_0 = (index_1 + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
    dim3 block_dim_0(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim_0(num_blocks_x_0, num_blocks_y_0, index_0);
    if(bit == 32){
        gather_v2<float><<<grid_dim_0, block_dim_0>>>(
            (float *)data,
            (long long *)indices,
            (float *)output,
            index_0,
            index_1,
            index_2,
            ((int *)dataShape)[axis],
            num);
    } else if(bit == 16){
        gather_v2<half><<<grid_dim_0, block_dim_0>>>(
            (half *)data,
            (long long *)indices,
            (half *)output,
            index_0,
            index_1,
            index_2,
            ((int *)dataShape)[axis],
            num);
    }
    
}
