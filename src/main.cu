#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <cufftXt.h>
#include <cuda_fp16.h>
#include <assert.h>

typedef half2 ftype;
long long sig_size = 1<<23;


int main(){

  ftype *h_idata = (ftype *)malloc(sig_size*sizeof(ftype));
  ftype *d_idata;
  ftype *d_odata;
  cudaMalloc(&d_idata, sizeof(ftype)*sig_size);
  cudaMalloc(&d_odata, sizeof(ftype)*sig_size);
  cufftHandle plan;
  cufftResult r;
  r = cufftCreate(&plan);
  assert(r == CUFFT_SUCCESS);
  size_t ws = 0;
  r = cufftXtMakePlanMany(plan, 1,  &sig_size, NULL, 1, 1, CUDA_C_16F, NULL, 1, 1, CUDA_C_16F, 1, &ws, CUDA_C_16F);
  assert(r == CUFFT_SUCCESS);
  r = cufftXtExec(plan, d_idata, d_odata, CUFFT_FORWARD); // warm-up
  assert(r == CUFFT_SUCCESS);
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start);
  r = cufftXtExec(plan, d_idata, d_odata, CUFFT_FORWARD);
  assert(r == CUFFT_SUCCESS);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float et;
  cudaEventElapsedTime(&et, start, stop);
  printf("forward FFT time for %lld samples: %fms\n", sig_size, et);
  return 0;
}
