#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <cufftXt.h>
#include <cuda_fp16.h>
#include <assert.h>

/*typedef half2 ftype;*/
/*long long sig_size = 1<<23;*/


/*int main(){*/

/*  ftype *h_idata = (ftype *)malloc(sig_size*sizeof(ftype));*/
/*  ftype *d_idata;*/
/*  ftype *d_odata;*/
/*  cudaMalloc(&d_idata, sizeof(ftype)*sig_size);*/
/*  cudaMalloc(&d_odata, sizeof(ftype)*sig_size);*/
/*  cufftHandle plan;*/
/*  cufftResult r;*/
/*  r = cufftCreate(&plan);*/
/*  assert(r == CUFFT_SUCCESS);*/
/*  size_t ws = 0;*/
/*  r = cufftXtMakePlanMany(plan, 1,  &sig_size, NULL, 1, 1, CUDA_C_16F, NULL, 1, 1, CUDA_C_16F, 1, &ws, CUDA_C_16F);*/
/*  assert(r == CUFFT_SUCCESS);*/
/*  r = cufftXtExec(plan, d_idata, d_odata, CUFFT_FORWARD); // warm-up*/
/*  assert(r == CUFFT_SUCCESS);*/
/*  cudaEvent_t start, stop;*/
/*  cudaEventCreate(&start); cudaEventCreate(&stop);*/
/*  cudaEventRecord(start);*/
/*  r = cufftXtExec(plan, d_idata, d_odata, CUFFT_FORWARD);*/
/*  assert(r == CUFFT_SUCCESS);*/
/*  cudaEventRecord(stop);*/
/*  cudaEventSynchronize(stop);*/
/*  float et;*/
/*  cudaEventElapsedTime(&et, start, stop);*/
/*  printf("forward FFT time for %lld samples: %fms\n", sig_size, et);*/
/*  return 0;*/
/*}*/


#include "pinned_mapped_vector_utils.hpp"
#include "pinned_mapped_allocator.hpp"

#include "my_cufft_utils.hpp"
#include "my_cuda_utils.hpp"

#include "my_generators.hpp"
#include "my_printers.hpp"

#include "my_utils.hpp"

#include "my_cuHalfComplex.hpp"

int main(int argc, char **argv) {
   try {
      cudaError_t cerror = cudaSuccess;
      cufftResult cufft_status = CUFFT_SUCCESS;
      bool debug = false;
      
      // Empirically-determined maximum number
      long long num_vals = 1<<21;

      ////////////////////////////////////////////////////////////////////
      // ALLOCATE KERNEL DATA
      ////////////////////////////////////////////////////////////////////
      dout << "Initializing memory for input and output data...\n";
      // Allocate pinned host memory that is also accessible by the device.
      pinned_mapped_vector<cuHalfComplex> samples;
      pinned_mapped_vector<cuHalfComplex> frequencies;
      samples.reserve( num_vals );
      frequencies.reserve( num_vals );
      frequencies.resize( num_vals );

      gen_cuHalfComplexes( samples.data(), num_vals, 0.0, 1.0 );
      print_cuHalfComplexes( samples.data(), 10, "Samples", "\n", "\n" );

      cufftHandle plan;
      size_t work_size = 0;

      try_cufft_func_throw(cufft_status, cufftCreate(&plan) );

      try_cufft_func_throw(cufft_status, 
         cufftXtMakePlanMany(plan, 1,  &num_vals, NULL, 1, 1, CUDA_C_16F, NULL, 1, 1, CUDA_C_16F, 1, &work_size, CUDA_C_16F) );

      dout << "Work Size after cufftXtMakePlanMany() is " << work_size << "\n";

      try_cufft_func_throw(cufft_status,
         cufftXtExec(plan, samples.data(), frequencies.data(), CUFFT_FORWARD) );

      print_cuHalfComplexes( frequencies.data(), 10, "Frequencies", "\n", "\n" );

      samples.clear();
      return SUCCESS;

   } catch( std::exception& ex ) {
      std::cout << "ERROR: " << ex.what() << "\n"; 
      return FAILURE;

   }
}
