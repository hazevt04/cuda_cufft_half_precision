
#include "my_cuHalfComplex.hpp"

void gen_cuHalfComplexes( cuHalfComplex* complexes, const int& num_complexes, const float& lower, const float& upper ) {
   std::random_device random_dev;
   std::mt19937 mersenne_gen(random_dev());
   std::uniform_real_distribution<float> dist(lower, upper);
   for( int index = 0; index < num_complexes; ++index ) {
      complexes[index].x = dist( mersenne_gen );
      complexes[index].y = dist( mersenne_gen );
   } 
}


void print_cuHalfComplexes(const cuHalfComplex* vals,
   const int& num_vals,
   const char* prefix,
   const char* delim,
   const char* suffix ) {

   for (int index = 0; index < num_vals; ++index) {
      std::cout << prefix;
      std::cout << vals[index] << ((index == num_vals - 1) ? "\n" : delim);
   }
   std::cout << suffix;
}
