//
// Created by Mads Bjoern on 2019-07-15.
//

#ifndef RAYCASTINGRENDERINGENGINEGPU_RCRE_SPHERE_H
#define RAYCASTINGRENDERINGENGINEGPU_RCRE_SPHERE_H

#include "RCRE_point3D.h"

struct RCRE_sphere {
    RCRE_point3D *center;
    double radius;
} typedef RCRE_sphere;

RCRE_sphere *RCRE_sphere_getSphereFromPointAndRadius(RCRE_point3D *center, double radius);
RCRE_sphere *RCRE_sphere_getSphereFromPointValueAndRadius(RCRE_point3D *center, double radius);

bool RCRE_sphere_isPointContainedWithin(RCRE_sphere *s, RCRE_point3D *p);
bool RCRE_sphere_getIntersectionPoint(RCRE_sphere *s, RCRE_point3D *rayOrigin, RCRE_point3D *rayDirection, int index, RCRE_point3D *outIntersectionPoint, RCRE_point3D *outReflectiveDirection);


#endif //RAYCASTINGRENDERINGENGINEGPU_RCRE_SPHERE_H
