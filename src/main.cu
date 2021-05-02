
#include "my_cuHalfComplex.hpp"

#include "device_allocator.hpp"

#include "pinned_vector_utils.hpp"
#include "pinned_allocator.hpp"

#include "my_cufft_utils.hpp"
#include "my_cuda_utils.hpp"

#include "my_generators.hpp"
#include "my_printers.hpp"

#include "my_cuda_utils.hpp"
#include "my_utils.hpp"

#include <cufftXt.h>
#include <cuda_fp16.h>
#include <assert.h>
#include <algorithm>

int main(int argc, char **argv) {
   try {
      cufftResult cufft_status = CUFFT_SUCCESS;
      cudaError_t cerror = cudaSuccess;
      bool debug = false;
      
      // We want the FFT to only have one spike, so we have to be careful about 
      // the sine wave frequency. The FFT size (num_vals) must be an integer 
      // multiple of the period of the sine wave
      // Per the answer from StackOverflow:
      // https://stackoverflow.com/a/36293992/11197687
      float frequency = 1024;
      float period = 1/frequency;
      
      float duration = period;
      
      float sample_rate = 16384 * frequency;

      // Must be long long due to expected type for cuFFTXt
      long long num_vals = duration * sample_rate;
      
      // cuFFTXt FFT Size (=num_vals) must be a power of 2 when using fp16
      assert( is_power_of_two( static_cast<int>(num_vals) ) );

      float amplitude = 1.0;

      std::cout << "Sine Wave Frequency is " << frequency << " Hz\n";
      std::cout << "Sine Wave Period is " << period << " seconds\n";
      std::cout << "\n"; 

      std::cout << "Duration is " << duration << " seconds\n";
      std::cout << "Sample Rate is " << sample_rate << " sps\n";
      std::cout << "Number of Values is " << num_vals << "\n"; 
      std::cout << "\n"; 

      std::cout << "Sine Wave Amplitude is " << amplitude << "\n";
      std::cout << "\n"; 

      ////////////////////////////////////////////////////////////////////
      // ALLOCATE KERNEL DATA
      ////////////////////////////////////////////////////////////////////
      dout << "Initializing memory for input and output data...\n";
      // Allocate pinned_mapped host memory that is also accessible by the device.
      pinned_vector<cuHalfComplex> samples;
      pinned_vector<cuHalfComplex> frequency_bins;
      device_vector<cuHalfComplex> d_samples;
      device_vector<cuHalfComplex> d_frequency_bins;

      samples.reserve( num_vals );
      frequency_bins.reserve( num_vals );

      d_samples.reserve( num_vals );
      d_frequency_bins.reserve( num_vals );
      
      frequency_bins.resize( num_vals );

      //gen_cuHalfComplexes( samples.data(), num_vals, 0.0, 1.0 );
      gen_cuHalfComplexes_sines( samples.data(), num_vals, amplitude, frequency, sample_rate );
      
      print_cuHalfComplexes( samples.data(), 0, 2, "Sample", "\n", "\n" );
      print_cuHalfComplexes( samples.data(), ((num_vals/4) - 1), 3, "Sample", "\n", "\n" );
      print_cuHalfComplexes( samples.data(), ((3 * (num_vals/4)) - 1), 3, "Sample", "\n", "\n" );
      print_cuHalfComplexes( samples.data(), (num_vals - 3), 3, "Sample", "\n", "\n" );

      cufftHandle plan;
      size_t work_size = 0;

      try_cufft_func_throw(cufft_status, cufftCreate(&plan) );

      try_cufft_func_throw(cufft_status, 
         cufftXtMakePlanMany(plan, 1,  &num_vals, NULL, 1, 1, CUDA_C_16F, NULL, 1, 1, CUDA_C_16F, 1, &work_size, CUDA_C_16F) );

      dout << "Work Size after cufftXtMakePlanMany() is " << work_size << "\n";
      dout << "\n";

      size_t num_val_bytes = num_vals * sizeof( cuHalfComplex );
      try_cuda_func( cerror, cudaMemcpyAsync( d_samples.data(), samples.data(), num_val_bytes,
               cudaMemcpyHostToDevice, nullptr ) );

      try_cufft_func_throw(cufft_status,
         //cufftXtExec(plan, samples.data(), frequency_bins.data(), CUFFT_FORWARD) );
         cufftXtExec( plan, d_samples.data(), d_frequency_bins.data(), CUFFT_FORWARD ) );
      
      try_cuda_func( cerror, cudaMemcpyAsync( frequency_bins.data(), d_frequency_bins.data(), num_val_bytes,
               cudaMemcpyDeviceToHost, nullptr ) );

      try_cuda_func( cerror, cudaDeviceSynchronize() );

      print_cuHalfComplexes( frequency_bins.data(), 0, 2, "Frequency Bin", "\n", "\n" );
      print_cuHalfComplexes( frequency_bins.data(), 1024, 1, "Frequency Bin", "\n", "\n" );
      print_cuHalfComplexes( frequency_bins.data(), ((num_vals/4) - 1), 3, "Frequency Bin", "\n", "\n" );
      print_cuHalfComplexes( frequency_bins.data(), ((3 * (num_vals/4)) - 1), 3, "Frequency Bin", "\n", "\n" );
      print_cuHalfComplexes( frequency_bins.data(), (num_vals - 3), 3, "Frequency Bin", "\n", "\n" );

      auto non_zero_freq_it = std::find_if( frequency_bins.begin(), frequency_bins.end(), 
         [](const cuHalfComplex& freq) {
            return (  are_cuHalfComplexes_not_equal( freq, make_cuHalfComplex(0.0f, 0.0f) )  ); 
         });

      if ( non_zero_freq_it != frequency_bins.end() ) {
         std::cout << "Frequency Bin " << std::distance( frequency_bins.begin(), non_zero_freq_it ) << " is "
            << *non_zero_freq_it << "\n"; 
      } else {
         std::cout << "All frequency bins are ZERO?\n"; 
      }

      samples.clear();
      return SUCCESS;

   } catch( std::exception& ex ) {
      std::cout << "ERROR: " << ex.what() << "\n"; 
      return FAILURE;

   }
}
