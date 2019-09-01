//
// Created by Mads Bjoern on 01/09/2019.
//

#ifndef RAYCASTINGRENDERINGENGINEGPU_RCRE_CUDA_H
#define RAYCASTINGRENDERINGENGINEGPU_RCRE_CUDA_H

#include "RCRE_color.h"
#include "RCRE_convexPolyhedron.h"
#include "RCRE_model3D.h"
#include "RCRE_sphere.h"

#ifdef RCRE_CUDA
RCRE_color *RCRE_color_toCUDA(RCRE_color *host);
RCRE_convexPolyhedron *RCRE_convexPolyhedron_toCUDA(RCRE_convexPolyhedron *host);
RCRE_model3D *RCRE_model3D_toCUDA(RCRE_model3D *host);
RCRE_point3D *RCRE_point3D_toCUDA(RCRE_point3D *host);
RCRE_sphere *RCRE_sphere_toCUDA(RCRE_sphere *host);
RCRE_transformationMatrix *RCRE_transformationMatrix_toCUDA(RCRE_transformationMatrix *host);
RCRE_triangle3D *RCRE_triangle3D_toCUDA(RCRE_triangle3D *host);
#endif

#endif //RAYCASTINGRENDERINGENGINEGPU_RCRE_CUDA_H
