
#include "my_cuHalfComplex.hpp"
#include <cmath>

void gen_cuHalfComplexes( cuHalfComplex* complexes, const int& num_complexes, const float& lower, const float& upper ) {
   std::random_device random_dev;
   std::mt19937 mersenne_gen(random_dev());
   std::uniform_real_distribution<float> dist(lower, upper);
   for( int index = 0; index < num_complexes; ++index ) {
      complexes[index].x = dist( mersenne_gen );
      complexes[index].y = dist( mersenne_gen );
   } 
}


void gen_cuHalfComplexes_sines( cuHalfComplex* complexes, const int& num_complexes, const float& amplitude, 
      const float& frequency, const float& sample_rate ) {

   for( int index = 0; index < num_complexes; ++index ) {
      complexes[index].x = amplitude * cos( 2 * M_PI * frequency * (static_cast<float>(index)/static_cast<float>(num_complexes)) );
      complexes[index].y = amplitude * sin( 2 * M_PI * frequency * (static_cast<float>(index)/static_cast<float>(num_complexes)) );
   } 
}

void print_cuHalfComplexes(const cuHalfComplex* vals,
   const int& start_index,
   const int& num_vals,
   const char* prefix,
   const char* delim,
   const char* suffix ) {

   int max_index = start_index + num_vals;
   for (int index = start_index; index < max_index; ++index) {
      std::cout << prefix;
      std::cout << "[" << index << "] = " << vals[index] << ((index == num_vals - 1) ? "\n" : delim);
   }
   std::cout << suffix;
}


void print_cuHalfComplexes(const cuHalfComplex* vals,
   const int& num_vals,
   const char* prefix,
   const char* delim,
   const char* suffix ) {

   print_cuHalfComplexes( vals, 0, num_vals, prefix, delim, suffix );
}

bool are_halves_equal(const half& lh, const half& rh) {
   // cuda_fp16.hpp has an overloaded operator= function for half to float
   float flh = lh;
   float frh = rh;
   return flh == frh;
}

bool are_halves_not_equal(const half& lh, const half& rh) {
   // cuda_fp16.hpp has an overloaded operator= function for half to float
   float flh = lh;
   float frh = rh;
   return flh != frh;
}


