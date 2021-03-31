#pragma once

#include "my_utils.hpp"

#include <vector_types.h>
#include <cuda_fp16.h>

#include <iostream>
#include <random>

typedef half2 cuHalfComplex;

__host__ __device__ static __inline__ float cuCrealh (cuHalfComplex x) 
{ 
    return x.x; 
}

__host__ __device__ static __inline__ float cuCimagh (cuHalfComplex x) 
{ 
    return x.y; 
}

__host__ __device__ static __inline__ cuHalfComplex make_cuHalfComplex 
                                                             (float r, float i)
{
    cuHalfComplex res;
    res.x = r;
    res.y = i;
    return res;
}

__host__ __device__ static __inline__ cuHalfComplex cuConjh (cuHalfComplex x)
{
    return make_cuHalfComplex (cuCrealh(x), -cuCimagh(x));
}
__host__ __device__ static __inline__ cuHalfComplex cuCaddh (cuHalfComplex x,
                                                              cuHalfComplex y)
{
    return make_cuHalfComplex (cuCrealh(x) + cuCrealh(y), 
                                cuCimagh(x) + cuCimagh(y));
}

__host__ __device__ static __inline__ cuHalfComplex cuCsubh (cuHalfComplex x,
                                                              cuHalfComplex y)
{
    return make_cuHalfComplex (cuCrealh(x) - cuCrealh(y), 
                                cuCimagh(x) - cuCimagh(y));
}



template<class _CharT, class _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os, const cuHalfComplex& __c) {
    std::basic_ostringstream<_CharT, _Traits> __s;
    __s.flags(__os.flags());
    __s.imbue(__os.getloc());
    __s.precision(__os.precision());
    __s << "{" << __c.x << ", " << __c.y << "}";
    return __os << __s.str();
}


void gen_cuHalfComplexes( cuHalfComplex* complexes, const int& num_complexes, const float& lower, const float& upper );

void gen_cuHalfComplexes_sines( cuHalfComplex* complexes, const int& num_complexes, const float& amplitude, const float& frequency );

void print_cuHalfComplexes(const cuHalfComplex* vals,
   const int& num_vals,
   const char* prefix,
   const char* delim,
   const char* suffix );
