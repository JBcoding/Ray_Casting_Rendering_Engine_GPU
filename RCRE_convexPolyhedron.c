//
// Created by Mads Bjoern on 2019-07-09.
//

#include "RCRE_convexPolyhedron.h"
#include "RCRE_constants.h"

PRE_DEVICE RCRE_convexPolyhedron *RCRE_convexPolyhedron_getConvexPolyhedronFromPoints(int nPoints, RCRE_point3D **points) {

    if (nPoints < 4) {
        return NULL;
    }

    int nFaces = 0;
    int nFaceSpace = 16;

    RCRE_triangle3D **faces = malloc(sizeof(RCRE_triangle3D*) * nFaceSpace);

    int offset = 0;

    for (offset = 0; offset < nPoints - 3; offset ++) {
        if (!RCRE_point3D_arePointsOnTheSamePlane(points[offset], points[offset + 1], points[offset + 2], points[offset + 3])) {
            break;
        }
        if (offset == nPoints - 4) {
            return NULL; // all points are on the same plane
        }
    }

    faces[0] = RCRE_triangle3D_getTriangleFromPoints(points[offset], points[offset + 1], points[offset + 2]);
    faces[1] = RCRE_triangle3D_getTriangleFromPoints(points[offset], points[offset + 1], points[offset + 3]);
    faces[2] = RCRE_triangle3D_getTriangleFromPoints(points[offset], points[offset + 2], points[offset + 3]);
    faces[3] = RCRE_triangle3D_getTriangleFromPoints(points[offset + 1], points[offset + 2], points[offset + 3]);
    nFaces = 4;

    RCRE_point3D *centerPoint = RCRE_point3D_copy(points[offset]);
    RCRE_point3D_averagePointsWithWeight(centerPoint, points[offset + 1], 1./2, centerPoint);
    RCRE_point3D_averagePointsWithWeight(centerPoint, points[offset + 2], 1./3, centerPoint);
    RCRE_point3D_averagePointsWithWeight(centerPoint, points[offset + 3], 1./4, centerPoint);

    for (int i = 0; i < nFaces; i ++) {
        RCRE_triangle3D_rotateTriangleAfterPoint(faces[i], centerPoint);
    }

    for (int i = 0; i < nPoints - 4; i ++) {
        int index = (i + offset + 4) % nPoints;
        RCRE_point3D *point = points[index];

        // Extend the faces which do not include the new point
        int tempNFaces = nFaces;
        for (int j = 0; j < tempNFaces; j ++) {
            if (RCRE_triangle3D_isPointOnPositiveSide(faces[j], point) < 0) {
                if (nFaces + 3 >= nFaceSpace) {
                    nFaceSpace *= 2;
                    RCRE_triangle3D **newFaces = malloc(sizeof(RCRE_triangle3D*) * nFaceSpace);
                    for (int k = 0; k < nFaces; k ++) {
                        newFaces[k] = faces[k];
                    }
                    free(faces);
                    faces = newFaces;
                }
                faces[nFaces] = RCRE_triangle3D_getTriangleFromPoints(point, faces[j]->p1, faces[j]->p2);
                faces[nFaces + 1] = RCRE_triangle3D_getTriangleFromPoints(point, faces[j]->p2, faces[j]->p3);
                faces[nFaces + 2] = RCRE_triangle3D_getTriangleFromPoints(point, faces[j]->p3, faces[j]->p1);
                nFaces += 3;
                free(faces[j]);
                faces[j] = NULL;
            }
        }

        // Remove duplicated faces
        for (int j = tempNFaces; j < nFaces - 1; j ++) {
            bool identical = false;
            for (int k = j + 1; k < nFaces; k ++) {
                if (faces[j] == NULL || faces[k] == NULL) {
                    continue;
                }
                if (RCRE_triangle3D_equalsWeak(faces[j], faces[k])) {
                    free(faces[k]);
                    faces[k] = NULL;
                    identical = true;
                }
            }
            if (identical) {
                free(faces[j]);
                faces[j] = NULL;
            }
        }

        // Clean up array
        for (int j = 0; j < nFaces - 1; j ++) {
            while (faces[nFaces - 1] == NULL) {nFaces --;}
            if (faces[j] == NULL) {
                faces[j] = faces[nFaces - 1];
                faces[nFaces - 1] = NULL;
                nFaces --;
            }
        }
        while (faces[nFaces - 1] == NULL) {nFaces --;}

        // Reorientate all faces to the new center point
        RCRE_point3D_averagePointsWithWeight(centerPoint, point, 1./(5 + i), centerPoint);
        for (int j = 0; j < nFaces; j ++) {
            RCRE_triangle3D_rotateTriangleAfterPoint(faces[j], centerPoint);
        }

    }

    RCRE_convexPolyhedron *convexPolyhedron = malloc(sizeof(RCRE_convexPolyhedron));
    convexPolyhedron->nTriangles = nFaces;
    convexPolyhedron->triangles = malloc(sizeof(RCRE_triangle3D*) * nFaces);
    for (int i = 0; i < nFaces; i ++) {
        convexPolyhedron->triangles[i] = faces[i];
    }
    convexPolyhedron->centerPoint = centerPoint;

    free(faces);

    convexPolyhedron->boundingSphereRadius = 0;
    for (int i = 0; i < nPoints; i ++) {
        RCRE_point3D *point = points[i];
        double distanceToCenter = RCRE_point3D_distance(convexPolyhedron->centerPoint, point);
        if (distanceToCenter > convexPolyhedron->boundingSphereRadius) {
            convexPolyhedron->boundingSphereRadius = distanceToCenter;
        }
    }

    return convexPolyhedron;

}


PRE_DEVICE bool RCRE_convexPolyhedron_isPointContainedWithin(RCRE_convexPolyhedron *cp, RCRE_point3D *p) {
    for (int i = 0; i < cp->nTriangles; i ++) {
        if (RCRE_triangle3D_isPointOnPositiveSide(cp->triangles[i], p) <= 0) {
            return false;
        }
    }
    return true;
}

PRE_DEVICE bool RCRE_convexPolyhedron_getIntersectionPoint(RCRE_convexPolyhedron *cp, RCRE_point3D *rayOrigin,
                                                RCRE_point3D *rayDirection, int index,
                                                RCRE_point3D *outIntersectionPoint,
                                                RCRE_point3D *outReflectiveDirection) {
    // we know we only have 0 or 2 intersections
    // ASSUMING rayDirection is a UNIT vector

    // START check if we intersect bound sphere, if not no need to check any more
    RCRE_point3D oc = {0};
    RCRE_point3D_subtract(rayOrigin, cp->centerPoint, &oc);
    double doc = RCRE_point3D_dotProduct(rayDirection, &oc);
    double ocLength = RCRE_point3D_distanceToOrigin(&oc);
    double c = doc * doc - (ocLength * ocLength - cp->boundingSphereRadius * cp->boundingSphereRadius);
    if (c < 0) {
        return false;
    }
    // END check

    if (index == 0) {
        for (int i = 0; i < cp->nTriangles; i++) {
            if (RCRE_triangle3D_getIntersectionPoint(cp->triangles[i], rayOrigin, rayDirection, outIntersectionPoint, outReflectiveDirection)) {
                return true;
            }
        }
    } else if (index == 1) {
        for (int i = cp->nTriangles - 1; i >= 0; i--) {
            if (RCRE_triangle3D_getIntersectionPoint(cp->triangles[i], rayOrigin, rayDirection, outIntersectionPoint, outReflectiveDirection)) {
                return true;
            }
        }
    }
    return false;
}
