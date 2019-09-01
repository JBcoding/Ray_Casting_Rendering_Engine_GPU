//
// Created by Mads Bjoern on 2019-07-09.
//

#include "RCRE_triangle3D.h"
#include "RCRE_constants.h"

PRE_DEVICE RCRE_triangle3D *RCRE_triangle3D_getTriangleFromPoints(RCRE_point3D *p1, RCRE_point3D *p2, RCRE_point3D *p3) {
    struct RCRE_triangle3D *triangle3D = malloc(sizeof(RCRE_triangle3D));
    triangle3D->p1 = p1;
    triangle3D->p2 = p2;
    triangle3D->p3 = p3;
    return triangle3D;
}

PRE_DEVICE RCRE_triangle3D *RCRE_triangle3D_getTriangleFromPointValues(RCRE_point3D *p1, RCRE_point3D *p2, RCRE_point3D *p3) {
    return RCRE_triangle3D_getTriangleFromPoints(RCRE_point3D_copy(p1), RCRE_point3D_copy(p2), RCRE_point3D_copy(p3));
}


PRE_DEVICE bool RCRE_triangle3D_equals(RCRE_triangle3D *t1, RCRE_triangle3D *t2) {
    return RCRE_point3D_equal(t1->p1, t2->p1) && RCRE_point3D_equal(t1->p2, t2->p2) && RCRE_point3D_equal(t1->p3, t2->p3);
}

PRE_DEVICE bool RCRE_triangle3D_equalsWeak(RCRE_triangle3D *t1, RCRE_triangle3D *t2) {
    return
            (RCRE_point3D_equal(t1->p1, t2->p1) && RCRE_point3D_equal(t1->p2, t2->p2) && RCRE_point3D_equal(t1->p3, t2->p3))
            ||
            (RCRE_point3D_equal(t1->p1, t2->p1) && RCRE_point3D_equal(t1->p2, t2->p3) && RCRE_point3D_equal(t1->p3, t2->p2))
            ||
            (RCRE_point3D_equal(t1->p1, t2->p2) && RCRE_point3D_equal(t1->p2, t2->p1) && RCRE_point3D_equal(t1->p3, t2->p3))
            ||
            (RCRE_point3D_equal(t1->p1, t2->p2) && RCRE_point3D_equal(t1->p2, t2->p3) && RCRE_point3D_equal(t1->p3, t2->p1))
            ||
            (RCRE_point3D_equal(t1->p1, t2->p3) && RCRE_point3D_equal(t1->p2, t2->p1) && RCRE_point3D_equal(t1->p3, t2->p2))
            ||
            (RCRE_point3D_equal(t1->p1, t2->p3) && RCRE_point3D_equal(t1->p2, t2->p2) && RCRE_point3D_equal(t1->p3, t2->p1));
}

PRE_DEVICE double RCRE_triangle3D_getArea(RCRE_triangle3D *t) {
    return RCRE_triangle3D_getAreaTimes2(t) / 2.0;
}

PRE_DEVICE double RCRE_triangle3D_getAreaTimes2(RCRE_triangle3D *t) {
    RCRE_point3D ab = {0};
    RCRE_point3D ac = {0};
    RCRE_point3D normal = {0};


    RCRE_point3D_subtract(t->p1, t->p2, &ab);
    RCRE_point3D_subtract(t->p1, t->p3, &ac);

    RCRE_point3D_crossProduct(&ab, &ac, &normal);

    return RCRE_point3D_distanceToOrigin(&normal);
}


PRE_DEVICE void RCRE_triangle3D_rotateTriangleAfterPoint(RCRE_triangle3D *t, RCRE_point3D *p) {
    RCRE_point3D ab = {0};
    RCRE_point3D ac = {0};

    RCRE_point3D_subtract(t->p1, t->p2, &ab);
    RCRE_point3D_subtract(t->p1, t->p3, &ac);

    RCRE_point3D abc = {0};
    RCRE_point3D acb = {0};

    RCRE_point3D_crossProduct(&ab, &ac, &abc);
    RCRE_point3D_crossProduct(&ac, &ab, &acb);

    RCRE_point3D_add(&abc, t->p1, &abc);
    RCRE_point3D_add(&acb, t->p1, &acb);

    if (RCRE_point3D_distance(p, &abc) >= RCRE_point3D_distance(p, &acb)) {
        RCRE_point3D *temp = t->p2;
        t->p2 = t->p3;
        t->p3 = temp;
    }
}

PRE_DEVICE int RCRE_triangle3D_isPointOnPositiveSide(RCRE_triangle3D *t, RCRE_point3D *p) {
    if (RCRE_point3D_arePointsOnTheSamePlane(t->p1, t->p2, t->p3, p)) {
        return 0;
    }

    RCRE_point3D ab = {0};
    RCRE_point3D ac = {0};

    RCRE_point3D_subtract(t->p1, t->p2, &ab);
    RCRE_point3D_subtract(t->p1, t->p3, &ac);

    RCRE_point3D abc = {0};
    RCRE_point3D acb = {0};

    RCRE_point3D_crossProduct(&ab, &ac, &abc);
    RCRE_point3D_crossProduct(&ac, &ab, &acb);

    RCRE_point3D_add(&abc, t->p1, &abc);
    RCRE_point3D_add(&acb, t->p1, &acb);

    if (RCRE_point3D_distance(p, &abc) >= RCRE_point3D_distance(p, &acb)) {
        return -1;
    } else {
        return 1;
    }

}

PRE_DEVICE bool RCRE_triangle3D_getIntersectionPoint(RCRE_triangle3D *t, RCRE_point3D *rayOrigin, RCRE_point3D *rayDirection,
                                          RCRE_point3D *outIntersectionPoint, RCRE_point3D *outReflectiveDirection) {

    RCRE_point3D planeNormal = {0};
    RCRE_point3D ab = {0};
    RCRE_point3D ac = {0};
    RCRE_point3D num = {0};
    RCRE_point3D planeIntersectionPoint = {0};

    RCRE_point3D_subtract(t->p1, t->p2, &ab);
    RCRE_point3D_subtract(t->p1, t->p3, &ac);

    RCRE_point3D_crossProduct(&ab, &ac, &planeNormal);

    RCRE_point3D_subtract(t->p1, rayOrigin, &num);

    double numerator = RCRE_point3D_dotProduct(&planeNormal, &num);
    double denominator = RCRE_point3D_dotProduct(&planeNormal, rayDirection);

    if (denominator == 0) {
        return false;
    }

    double rayDirectionScale = numerator / denominator;

    if (rayDirectionScale < 0) {
        return false;
    }

    RCRE_point3D_scale(rayDirection, rayDirectionScale, &planeIntersectionPoint);
    RCRE_point3D_add(&planeIntersectionPoint, rayOrigin, &planeIntersectionPoint);


    // Check to see if intersection point is hitting the triangle by converting to barycentric coordinates
    // https://en.wikipedia.org/wiki/Barycentric_coordinate_system

    RCRE_point3D pp1 = {0};
    RCRE_point3D pp2 = {0};
    RCRE_point3D pp3 = {0};

    RCRE_point3D_subtract(&planeIntersectionPoint, t->p1, &pp1);
    RCRE_point3D_subtract(&planeIntersectionPoint, t->p2, &pp2);
    RCRE_point3D_subtract(&planeIntersectionPoint, t->p3, &pp3);

    RCRE_point3D alpha = {0};
    RCRE_point3D beta = {0};
    RCRE_point3D gamma = {0};

    RCRE_point3D_crossProduct(&pp2, &pp3, &alpha);
    RCRE_point3D_crossProduct(&pp3, &pp1, &beta);
    RCRE_point3D_crossProduct(&pp1, &pp2, &gamma);

    double a = RCRE_point3D_distanceToOrigin(&alpha);
    double b = RCRE_point3D_distanceToOrigin(&beta);
    double g = RCRE_point3D_distanceToOrigin(&gamma);
    double areaTimes2 = RCRE_triangle3D_getAreaTimes2(t);

    if (fabs(a + b + g - areaTimes2) < 0.0000001) {
        outIntersectionPoint->x = planeIntersectionPoint.x;
        outIntersectionPoint->y = planeIntersectionPoint.y;
        outIntersectionPoint->z = planeIntersectionPoint.z;



        RCRE_point3D rayOriginRotated = {0};

        RCRE_point3D_rotatePointAroundAxis(rayOrigin, &planeNormal, &planeIntersectionPoint, M_PI, &rayOriginRotated);

        RCRE_point3D_subtract(&rayOriginRotated, &planeIntersectionPoint, outReflectiveDirection);

        return true;
    }
    return false;
}