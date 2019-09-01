//
// Created by Mads Bjoern on 2019-07-14.
//

#ifndef RAYCASTINGRENDERINGENGINEGPU_RCRE_CONSTANTS_H
#define RAYCASTINGRENDERINGENGINEGPU_RCRE_CONSTANTS_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef RCRE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif


#ifdef RCRE_CUDA

#define PRE_GLOBAL __global__
#define PRE_DEVICE __host__ __device__

#else

#define PRE_GLOBAL
#define PRE_DEVICE

#endif


#define square(x) (x)*(x)

#define RCRE_model3D_DATATYPE_SPHERE            1
#define RCRE_model3D_DATATYPE_CONVEX_POLYHEDRON 2

#endif //RAYCASTINGRENDERINGENGINEGPU_RCRE_CONSTANTS_H
