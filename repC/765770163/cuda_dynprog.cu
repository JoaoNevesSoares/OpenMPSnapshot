#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

/* Include polybench common header. */
#include "../PolybenchC/polybench.h"

/* Include benchmark-specific header. */
/* Default data type is int, default size is 50. */
#include "../dynprog/dynprog.h"

#include <omp.h>
#include <time.h>

#ifndef NTHREADS
#define NTHREADS 4
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

static void init_array(int length,
                       DATA_TYPE *c,
		       DATA_TYPE *W)
{
  for (int i = 0; i < length; i++)
  {
    c[i] = 0;
    W[i] = ((DATA_TYPE) -i) / length;
  }
}

static void print_array(DATA_TYPE out)
{
  fprintf(stderr, DATA_PRINTF_MODIFIER, out);
  fprintf(stderr, "\n");
}

//static void kernel_dynprog(int tsteps, int length,
//                           DATA_TYPE POLYBENCH_1D(c, LENGTH, length),
//                           DATA_TYPE POLYBENCH_1D(W, LENGTH, length),
//                           DATA_TYPE sum_c,
//                           DATA_TYPE *out)
//{
//
//  DATA_TYPE out_l = 0;
//  sum_c = 0;
//
//  for (int i = 1; i < _PB_LENGTH; i++)
//  {
//    #pragma omp parallel for num_threads(NTHREADS) reduction(+:sum_c)
//    for (int j = 1; j < i; j++)
//      sum_c += c[j];
//    c[i] = sum_c + W[i];
//    sum_c = 0;
//  }
//
//  for (int k = 0; k < _PB_TSTEPS; k++)
//    out_l += c[_PB_LENGTH - 1];
//  
//  *out = out_l;
//
//}

__device__ void warpReduce(volatile double* sdata, int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

__global__ void partial_sum(DATA_TYPE *input, DATA_TYPE *output, int size) {
  extern __shared__ double sharedData[];

  int tid = threadIdx.x;
  int gid = blockIdx.x*(blockDim.x*2) + threadIdx.x;

  if (gid < size) {
    sharedData[tid] = input[gid] + input[gid+blockDim.x];
  }
  else {
    sharedData[tid] = 0;
  }

  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 32; offset >>= 1) {
    if (tid < offset) {
      sharedData[tid] += sharedData[tid + offset];
    }
    __syncthreads();
  }

  if (tid < 32) 
    warpReduce(sharedData, tid);

  if (tid == 0)
    output[blockIdx.x] = sharedData[0];
}

__global__ void update_c(DATA_TYPE *c, DATA_TYPE *part_sum, DATA_TYPE *W, int i)
{
  int thid = threadIdx.x;
  
  if(thid == 0) {
    c[i] = part_sum[0] + W[i];
  }
}

int main(int argc, char *argv[]) {
  /* Retrieve problem size. */
  int length = LENGTH;
  int tsteps = TSTEPS;

  int blockSize = BLOCK_SIZE;
  int numBlocks = (length + blockSize - 1) / blockSize;

  /* OPENMP PARALLELIZATION */

  DATA_TYPE out;
//  DATA_TYPE sum_c = 0;
//  POLYBENCH_1D_ARRAY_DECL(c, DATA_TYPE, LENGTH, length);
//  POLYBENCH_1D_ARRAY_DECL(W, DATA_TYPE, LENGTH, length);
//
//  /* Initialize array(s). */
//  init_array(length, POLYBENCH_ARRAY(c), POLYBENCH_ARRAY(W));
//
//  clock_t start, end;
//
//  start = clock();
//
//  /* Run kernel. */
//  kernel_dynprog(tsteps, length,
//                 POLYBENCH_ARRAY(c),
//                 POLYBENCH_ARRAY(W),
//                 sum_c,
//                 &out);
//
//  end = clock();
//
//  printf("OPENMP\nElapsed time: %f seconds\nResult: %.2f\n\n", ((double) (end - start)) / CLOCKS_PER_SEC, out);
//
//  POLYBENCH_FREE_ARRAY(c);
//  POLYBENCH_FREE_ARRAY(W);

  /* CUDA PARALLELIZATION */

  /* Variable declaration/allocation. */
  DATA_TYPE *h_out_l; 
  DATA_TYPE *h_c = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * length);
  DATA_TYPE *h_W = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * length);
  
  /* Initialize array(s). */
  init_array(length, h_c, h_W);

  DATA_TYPE *d_c, *d_out_l, *d_W, *d_pout_l, *d_pout2_l;
  cudaMalloc((void**)&d_c, length * sizeof(DATA_TYPE));
  cudaMalloc((void**)&d_out_l, numBlocks * sizeof(DATA_TYPE));
  cudaMalloc((void**)&d_W, length * sizeof(DATA_TYPE));
  
  cudaMalloc((void**)&d_pout_l, numBlocks * sizeof(DATA_TYPE));
  cudaMalloc((void**)&d_pout2_l, numBlocks * sizeof(DATA_TYPE));
  
  /* Start timer. */
//  start = clock();
  polybench_start_instruments;

  cudaMemcpy(d_c, h_c, length * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W, h_W, length * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

  for(int i = 1; i <= length; i++) 
  {

	  if (i < BLOCK_SIZE)
	  {
		int is_numBlocks = (i + blockSize - 1) / blockSize;
    		partial_sum<<<is_numBlocks, blockSize, blockSize * sizeof(DATA_TYPE)>>>(d_c, d_out_l, i);
    		update_c<<<1, 1>>>(d_c, d_out_l, d_W, i);
	  }
	  else if (i < BLOCK_SIZE * BLOCK_SIZE)
	  {
		int is_numBlocks = (i + blockSize - 1) / blockSize;
    		partial_sum<<<is_numBlocks, blockSize, blockSize * sizeof(DATA_TYPE)>>>(d_c, d_pout_l, i);
		int is_numBlocks2 = (is_numBlocks + blockSize - 1) / blockSize;
		partial_sum<<<is_numBlocks2, blockSize, blockSize * sizeof(DATA_TYPE)>>>(d_pout_l, d_out_l, i);
		update_c<<<1, 1>>>(d_c, d_out_l, d_W, i);
	  }
	  else
	  {
		int is_numBlocks = (i + blockSize - 1) / blockSize;
                partial_sum<<<is_numBlocks, blockSize, blockSize * sizeof(DATA_TYPE)>>>(d_c, d_pout2_l, i);
                int is_numBlocks2 = (is_numBlocks + blockSize - 1) / blockSize;
                partial_sum<<<is_numBlocks2, blockSize, blockSize * sizeof(DATA_TYPE)>>>(d_pout2_l, d_pout_l, i);
		int is_numBlocks3 = (is_numBlocks2 + blockSize - 1) / blockSize;
                partial_sum<<<is_numBlocks3, blockSize, blockSize * sizeof(DATA_TYPE)>>>(d_pout_l, d_out_l, i);
                update_c<<<1, 1>>>(d_c, d_out_l, d_W, i);
	  }
 
  }
  cudaDeviceSynchronize();
  
  h_out_l = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * numBlocks);

  cudaMemcpy(h_c, d_c, length * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
  
  for(int k=0; k<tsteps; k++)
    out += h_c[length - 1];

  //out = out * TSTEPS;

//  end = clock();
  polybench_stop_instruments;
  
  /* Stop and print timer. */
  polybench_print_instruments;
  
  /* Prevent dead-code elimination. All live-out_l data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(out));

  /* Be clean. */
  free(h_out_l);
  free(h_W);
  free(h_c);
  cudaFree(d_W);
  cudaFree(d_out_l);
  cudaFree(d_c);

  return 0;

}

