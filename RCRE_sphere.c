//
// Created by Mads Bjoern on 2019-07-15.
//

#include "RCRE_sphere.h"
#include "RCRE_constants.h"

PRE_DEVICE RCRE_sphere *RCRE_sphere_getSphereFromPointAndRadius(RCRE_point3D *center, double radius) {
    RCRE_sphere *sphere = (RCRE_sphere *)malloc(sizeof(RCRE_sphere));
    sphere->center = center;
    sphere->radius = radius;
    return sphere;
}

PRE_DEVICE RCRE_sphere *RCRE_sphere_getSphereFromPointValueAndRadius(RCRE_point3D *center, double radius) {
    return RCRE_sphere_getSphereFromPointAndRadius(RCRE_point3D_copy(center), radius);
}


PRE_DEVICE bool RCRE_sphere_isPointContainedWithin(RCRE_sphere *s, RCRE_point3D *p) {
    return RCRE_point3D_distance(s->center, p) < s->radius;
}

PRE_DEVICE bool RCRE_sphere_getIntersectionPoint(RCRE_sphere *s, RCRE_point3D *rayOrigin, RCRE_point3D *rayDirection, int index,
                                      RCRE_point3D *outIntersectionPoint, RCRE_point3D *outReflectiveDirection) {
    // ASSUMING rayDirection is a UNIT vector

    if (index < 0 || index > 2) {
        return false;
    }

    RCRE_point3D oc;
    RCRE_point3D sphereIntersectionPoint;

    RCRE_point3D_subtract(rayOrigin, s->center, &oc);
    double doc = RCRE_point3D_dotProduct(rayDirection, &oc);
    double ocLength = RCRE_point3D_distanceToOrigin(&oc);

    double c = doc * doc - (ocLength * ocLength - s->radius * s->radius);
    double a = -doc;

    if (c < 0) {
        return false;
    }

    double sqrtC = sqrt(c);

    double d1 = a + sqrtC;
    double d2 = a - sqrtC;

    double rayDirectionScale;

    if (d1 > 0 && d2 > 0) {
        rayDirectionScale = (index == 0) ? d1 : d2;
    } else {
        rayDirectionScale = fmax(d1, d2);
        if (rayDirectionScale < 0) {
            return false;
        }
        if (index > 0) {
            return false;
        }
    }

    RCRE_point3D_scale(rayDirection, rayDirectionScale, &sphereIntersectionPoint);
    RCRE_point3D_add(&sphereIntersectionPoint, rayOrigin, &sphereIntersectionPoint);

    outIntersectionPoint->x = sphereIntersectionPoint.x;
    outIntersectionPoint->y = sphereIntersectionPoint.y;
    outIntersectionPoint->z = sphereIntersectionPoint.z;



    RCRE_point3D rayOriginRotated;
    RCRE_point3D centerToIntersectionPoint;

    RCRE_point3D_subtract(&sphereIntersectionPoint, s->center, &centerToIntersectionPoint);

    RCRE_point3D_rotatePointAroundAxis(rayOrigin, &centerToIntersectionPoint, s->center, M_PI, &rayOriginRotated);

    RCRE_point3D_subtract(&rayOriginRotated, &sphereIntersectionPoint, outReflectiveDirection);

    return true;
}