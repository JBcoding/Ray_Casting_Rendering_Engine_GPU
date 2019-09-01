//
// Created by Mads Bjoern on 2019-07-09.
//

#ifndef RAYCASTINGRENDERINGENGINEGPU_RCRE_CONVEXPOLYHEDRON_H
#define RAYCASTINGRENDERINGENGINEGPU_RCRE_CONVEXPOLYHEDRON_H

#include "RCRE_triangle3D.h"
#include "RCRE_constants.h"

struct RCRE_convexPolyhedron {
    int nTriangles;
    RCRE_triangle3D **triangles;
    RCRE_point3D *centerPoint;

    double boundingSphereRadius;
} typedef RCRE_convexPolyhedron;

PRE_DEVICE RCRE_convexPolyhedron *RCRE_convexPolyhedron_getConvexPolyhedronFromPoints(int nPoints, RCRE_point3D **points);

PRE_DEVICE bool RCRE_convexPolyhedron_isPointContainedWithin(RCRE_convexPolyhedron *cp, RCRE_point3D *p);
PRE_DEVICE bool RCRE_convexPolyhedron_getIntersectionPoint(RCRE_convexPolyhedron *cp, RCRE_point3D *rayOrigin, RCRE_point3D *rayDirection, int index, RCRE_point3D *outIntersectionPoint, RCRE_point3D *outReflectiveDirection);

#endif //RAYCASTINGRENDERINGENGINEGPU_RCRE_CONVEXPOLYHEDRON_H
