//
// Created by Mads Bjoern on 2019-07-15.
//

#ifndef RAYCASTINGRENDERINGENGINEGPU_RCRE_MODEL3D_H
#define RAYCASTINGRENDERINGENGINEGPU_RCRE_MODEL3D_H

#include "RCRE_transformationMatrix.h"
#include "RCRE_color.h"

struct RCRE_model3D_entry {
    void *data;
    int datatype;
    bool negative;
    RCRE_transformationMatrix *transformationMatrix;
    RCRE_transformationMatrix *inverseTransformationMatrix;
} typedef RCRE_model3D_entry;

struct RCRE_model3D {
    RCRE_model3D_entry **entries;
    int nEntries;
    int nEntriesSpace;
    RCRE_transformationMatrix *transformationMatrix;
    RCRE_transformationMatrix *inverseTransformationMatrix;
    RCRE_point3D *centerPoint;

    RCRE_color *color;

    double reflectivity; // 1 for perfect mirror, and 0 for no reflectivity

    double transparency; // 1 for un-seeable glass, 0 for total opaque
    // 0.5 for 50% color for each unit through

    double refractiveIndex; // https://en.wikipedia.org/wiki/Refractive_index

} typedef RCRE_model3D;

RCRE_model3D *RCRE_model3D_getModel(RCRE_transformationMatrix *tm, RCRE_color *color, double reflectivity, double transparency, double refractiveIndex);

void RCRE_model3D_insertEntry(RCRE_model3D *model, void *data, int datatype, bool negative, RCRE_transformationMatrix *tm);

bool RCRE_model3D_getIntersection(RCRE_model3D *m, RCRE_point3D *rayOrigin, RCRE_point3D *rayDirection, RCRE_point3D *outIntersectionPoint, RCRE_point3D *outReflectiveDirection, RCRE_point3D *outExitPoint, RCRE_point3D *outExitDirection);
bool RCRE_model3D_getIntersectionPoint(RCRE_model3D *m, bool rayOriginatingOutside, RCRE_point3D *rayOrigin, RCRE_point3D *rayDirection, RCRE_point3D *outIntersectionPoint, RCRE_point3D *outReflectiveDirection, RCRE_point3D *outRefractionDirection);

bool RCRE_model3D_isPointContainedWithinEntry(RCRE_model3D_entry *me, RCRE_point3D *p);
RCRE_point3D *RCRE_model3D_getCenterPointEntry(RCRE_model3D_entry *me);

#endif //RAYCASTINGRENDERINGENGINEGPU_RCRE_MODEL3D_H
