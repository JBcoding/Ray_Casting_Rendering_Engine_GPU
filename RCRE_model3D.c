//
// Created by Mads Bjoern on 2019-07-15.
//

#include "RCRE_model3D.h"
#include "RCRE_constants.h"
#include "RCRE_convexPolyhedron.h"
#include "RCRE_sphere.h"

PRE_DEVICE RCRE_model3D *RCRE_model3D_getModel(RCRE_transformationMatrix *tm, RCRE_color *color, double reflectivity, double transparency, double refractiveIndex) {
    RCRE_model3D *model3D = (RCRE_model3D *)malloc(sizeof(RCRE_model3D));
    model3D->nEntries = 0;
    model3D->nEntriesSpace = 16;
    model3D->entries = (RCRE_model3D_entry **)malloc(sizeof(RCRE_model3D_entry *) * model3D->nEntriesSpace);
    model3D->transformationMatrix = tm;
    RCRE_transformationMatrix *inverseMatrix = RCRE_transformationMatrix_getIdentityMatrix();
    RCRE_transformationMatrix_inverse(tm, inverseMatrix);
    model3D->inverseTransformationMatrix = inverseMatrix;

    model3D->color = color;
    model3D->reflectivity = reflectivity;
    model3D->transparency = transparency;
    model3D->refractiveIndex = refractiveIndex;

    model3D->centerPoint = RCRE_point3D_getPointFromValues(0, 0, 0);

    return model3D;
}


PRE_DEVICE void RCRE_model3D_insertEntry(RCRE_model3D *model, void *data, int datatype, bool negative, RCRE_transformationMatrix *tm) {
    if (model->nEntries + 1 >= model->nEntriesSpace) {
        model->nEntriesSpace *= 2;
        RCRE_model3D_entry **newEntries = (RCRE_model3D_entry **)malloc(sizeof(RCRE_model3D_entry*) * model->nEntriesSpace);
        for (int i = 0; i < model->nEntries; i ++) {
            newEntries[i] = model->entries[i];
        }
        free(model->entries);
        model->entries = newEntries;
    }

    RCRE_model3D_entry *entry = (RCRE_model3D_entry *)malloc(sizeof(RCRE_model3D_entry));
    entry->data = data;
    entry->datatype = datatype;
    entry->negative = negative;
    entry->transformationMatrix = tm;
    RCRE_transformationMatrix *inverseMatrix = RCRE_transformationMatrix_getIdentityMatrix();
    RCRE_transformationMatrix_inverse(tm, inverseMatrix);
    entry->inverseTransformationMatrix = inverseMatrix;
    model->entries[model->nEntries] = entry;

    model->nEntries ++;

    RCRE_point3D *dataCenterPoint = RCRE_model3D_getCenterPointEntry(entry);

    RCRE_point3D_averagePointsWithWeight(model->centerPoint, dataCenterPoint, 1.0 / model->nEntries, model->centerPoint);
}


PRE_DEVICE bool RCRE_model3D_getIntersection(RCRE_model3D *m, RCRE_point3D *rayOrigin, RCRE_point3D *rayDirection,
                                  RCRE_point3D *outIntersectionPoint, RCRE_point3D *outReflectiveDirection,
                                  RCRE_point3D *outExitPoint, RCRE_point3D *outExitDirection) {
    RCRE_point3D insideDirection;
    if (RCRE_model3D_getIntersectionPoint(m, true, rayOrigin, rayDirection, outIntersectionPoint, outReflectiveDirection, &insideDirection)) {
        RCRE_point3D dummy;
        if (RCRE_model3D_getIntersectionPoint(m, false, outIntersectionPoint, &insideDirection, outExitPoint, &dummy, outExitDirection)) {
            return true;
        }
    }
    return false;
}

PRE_DEVICE bool RCRE_model3D_getIntersectionPoint(RCRE_model3D *m, bool rayOriginatingOutside, RCRE_point3D *rayOrigin, RCRE_point3D *rayDirection,
                                       RCRE_point3D *outIntersectionPoint, RCRE_point3D *outReflectiveDirection, RCRE_point3D *outRefractionDirection) {
    RCRE_point3D rayOriginT;
    RCRE_point3D rayDirectionT;

    RCRE_transformationMatrix_applyToPointWithRegardsToPoint(m->inverseTransformationMatrix, rayOrigin, m->centerPoint, &rayOriginT);
    RCRE_transformationMatrix_applyToPoint(m->inverseTransformationMatrix, rayDirection, &rayDirectionT);

    double bestDistance = DBL_MAX;

    for (int i = 0; i < m->nEntries; i ++) {
        RCRE_model3D_entry *entry = m->entries[i];

        RCRE_point3D rayOriginTT;
        RCRE_point3D rayDirectionTT;

        RCRE_transformationMatrix_applyToPointWithRegardsToPoint(entry->inverseTransformationMatrix, &rayOriginT, RCRE_model3D_getCenterPointEntry(entry), &rayOriginTT);
        RCRE_transformationMatrix_applyToPoint(entry->inverseTransformationMatrix, &rayDirectionT, &rayDirectionTT);

        RCRE_point3D rayDirectionTTUnit;
        RCRE_point3D_getUnit(&rayDirectionTT, &rayDirectionTTUnit);

        int index = 0;
        while (true) {
            bool success = false;
            RCRE_point3D intersectionPointTT;
            RCRE_point3D reflectiveDirectionTT;
            if (entry->datatype == RCRE_model3D_DATATYPE_CONVEX_POLYHEDRON) {
                success = RCRE_convexPolyhedron_getIntersectionPoint((RCRE_convexPolyhedron*)entry->data, &rayOriginTT, &rayDirectionTTUnit, index, &intersectionPointTT, &reflectiveDirectionTT);
            } else if (entry->datatype == RCRE_model3D_DATATYPE_SPHERE) {
                success = RCRE_sphere_getIntersectionPoint((RCRE_sphere*)entry->data, &rayOriginTT, &rayDirectionTTUnit, index, &intersectionPointTT, &reflectiveDirectionTT);
            }
            index ++;

            if (!success) {
                break;
            }


            // positive hit
            // must not be inside something before (only if followed by negative)
            // must not be inside or outside after

            // negative hit
            // must be inside something before
            // must not be inside or outside after

            RCRE_point3D intersectionPointT;
            RCRE_point3D reflectiveDirectionT;
            RCRE_transformationMatrix_applyToPointWithRegardsToPoint(entry->transformationMatrix, &intersectionPointTT, RCRE_model3D_getCenterPointEntry(entry), &intersectionPointT);
            RCRE_transformationMatrix_applyToPoint(entry->transformationMatrix, &reflectiveDirectionTT, &reflectiveDirectionT);

            bool beforeSuccess = !entry->negative;
            bool afterSuccess = true;
            for (int j = i - 1; j >= 0; j--) {
                RCRE_transformationMatrix_applyToPointWithRegardsToPoint(m->entries[j]->inverseTransformationMatrix, &intersectionPointT, RCRE_model3D_getCenterPointEntry(m->entries[j]), &intersectionPointTT);
                if (RCRE_model3D_isPointContainedWithinEntry(m->entries[j], &intersectionPointTT)) {
                    beforeSuccess = entry->negative != m->entries[j]->negative;
                    break;
                }
            }
            for (int j = i + 1; j < m->nEntries; j++) {
                RCRE_transformationMatrix_applyToPointWithRegardsToPoint(m->entries[j]->inverseTransformationMatrix, &intersectionPointT, RCRE_model3D_getCenterPointEntry(m->entries[j]), &intersectionPointTT);
                if (RCRE_model3D_isPointContainedWithinEntry(m->entries[j], &intersectionPointTT)) {
                    afterSuccess = false;
                    break;
                }
            }
            if (afterSuccess && beforeSuccess) {
                RCRE_point3D intersectionPoint;
                RCRE_point3D reflectiveDirection;
                RCRE_transformationMatrix_applyToPointWithRegardsToPoint(m->transformationMatrix, &intersectionPointT, m->centerPoint, &intersectionPoint);
                RCRE_transformationMatrix_applyToPoint(m->transformationMatrix, &reflectiveDirectionT, &reflectiveDirection);


                double distance = RCRE_point3D_distance(rayOrigin, &intersectionPoint);
                if (distance < bestDistance && distance >= 0) {
                    if (!rayOriginatingOutside && distance < 0.000000001) {
                        continue;
                    }

                    bestDistance = distance;

                    RCRE_point3D_copyInto(&intersectionPoint, outIntersectionPoint);

                    RCRE_point3D_copyInto(&reflectiveDirection, outReflectiveDirection);

                    double angleToNormal =
                            (M_PI - RCRE_point3D_getAngleBetweenPoints(rayDirection, &reflectiveDirection)) / 2.0;
                    RCRE_point3D insideRayDirection;

                    if (angleToNormal != 0) {
                        double refractiveIndex = m->refractiveIndex;
                        if (!rayOriginatingOutside) {
                            refractiveIndex = 1. / refractiveIndex;
                        }
                        double newAngleToNormal = asin(sin(angleToNormal) / refractiveIndex);

                        RCRE_point3D axisToTurnAbout;
                        axisToTurnAbout.x = 0;
                        axisToTurnAbout.y = 0;
                        axisToTurnAbout.z = 0;
                        RCRE_point3D_crossProduct(&reflectiveDirection, rayDirection, &axisToTurnAbout);
                        double angleToTurn = angleToNormal - newAngleToNormal;

                        RCRE_point3D origin = {0, 0, 0};
                        RCRE_point3D_rotatePointAroundAxis(rayDirection, &axisToTurnAbout, &origin,
                                                           angleToTurn, &insideRayDirection);
                    } else {
                        RCRE_point3D_copyInto(rayDirection, &insideRayDirection);
                    }

                    RCRE_point3D_copyInto(&insideRayDirection, outRefractionDirection);
                }
            }
        }
    }

    return bestDistance != DBL_MAX;
}


PRE_DEVICE bool RCRE_model3D_isPointContainedWithinEntry(RCRE_model3D_entry *me, RCRE_point3D *p) {
    if (me->datatype == RCRE_model3D_DATATYPE_CONVEX_POLYHEDRON) {
        return RCRE_convexPolyhedron_isPointContainedWithin((RCRE_convexPolyhedron*)me->data, p);
    } else if (me->datatype == RCRE_model3D_DATATYPE_SPHERE) {
        return RCRE_sphere_isPointContainedWithin((RCRE_sphere*)me->data, p);
    }
}

PRE_DEVICE RCRE_point3D *RCRE_model3D_getCenterPointEntry(RCRE_model3D_entry *me) {
    if (me->datatype == RCRE_model3D_DATATYPE_CONVEX_POLYHEDRON) {
        return  ((RCRE_convexPolyhedron*)me->data)->centerPoint;
    } else if (me->datatype == RCRE_model3D_DATATYPE_SPHERE) {
        return  ((RCRE_sphere*)me->data)->center;
    }
}
