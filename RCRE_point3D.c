//
// Created by Mads Bjoern on 2019-07-09.
//

#include "RCRE_point3D.h"
#include "RCRE_constants.h"

PRE_DEVICE RCRE_point3D *RCRE_point3D_getPointFromValues(double x, double y, double z) {
    RCRE_point3D *point3D = malloc(sizeof(RCRE_point3D));
    point3D->x = x;
    point3D->y = y;
    point3D->z = z;
    return point3D;
}

PRE_DEVICE RCRE_point3D *RCRE_point3D_copy(RCRE_point3D *p) {
    return RCRE_point3D_getPointFromValues(p->x, p->y, p->z);
}

PRE_DEVICE void RCRE_point3D_copyInto(RCRE_point3D *p, RCRE_point3D *out) {
    out->x = p->x;
    out->y = p->y;
    out->z = p->z;
}


PRE_DEVICE bool RCRE_point3D_equal(RCRE_point3D *a, RCRE_point3D *b) {
    return a->x == b->x && a->y == b->y && a->z == b->z;
}

PRE_DEVICE void RCRE_point3D_add(RCRE_point3D *a, RCRE_point3D *b, RCRE_point3D *out) {
    out->x = a->x + b->x;
    out->y = a->y + b->y;
    out->z = a->z + b->z;
}

PRE_DEVICE void RCRE_point3D_subtract(RCRE_point3D *a, RCRE_point3D *b, RCRE_point3D *out) {
    out->x = a->x - b->x;
    out->y = a->y - b->y;
    out->z = a->z - b->z;
}

PRE_DEVICE void RCRE_point3D_scale(RCRE_point3D *p, double scale, RCRE_point3D *out) {
    out->x = p->x * scale;
    out->y = p->y * scale;
    out->z = p->z * scale;
}

PRE_DEVICE double RCRE_point3D_distance(RCRE_point3D *a, RCRE_point3D *b) {
    return sqrt(square(a->x - b->x) + square(a->y - b->y) + square(a->z - b->z));
}

PRE_DEVICE double RCRE_point3D_distanceToOrigin(RCRE_point3D *a) {
    return sqrt(square(a->x) + square(a->y) + square(a->z));
}

PRE_DEVICE void RCRE_point3D_crossProduct(RCRE_point3D *a, RCRE_point3D *b, RCRE_point3D *out) {
    double x = a->y * b->z - a->z * b->y;
    double y = a->z * b->x - a->x * b->z;
    double z = a->x * b->y - a->y * b->x;
    out->x = x;
    out->y = y;
    out->z = z;
}

PRE_DEVICE double RCRE_point3D_dotProduct(RCRE_point3D *a, RCRE_point3D *b) {
    return a->x * b->x + a->y * b->y + a->z * b->z;
}

PRE_DEVICE void RCRE_point3D_getUnit(RCRE_point3D *p, RCRE_point3D *out) {
    double length = RCRE_point3D_distanceToOrigin(p);
    out->x = p->x / length;
    out->y = p->y / length;
    out->z = p->z / length;
}


PRE_DEVICE bool RCRE_point3D_arePointsOnTheSamePlane(RCRE_point3D *a, RCRE_point3D *b, RCRE_point3D *c, RCRE_point3D *d) {
    if (RCRE_point3D_equal(a, b) || RCRE_point3D_equal(a, c) || RCRE_point3D_equal(a, d) ||
        RCRE_point3D_equal(b, c) || RCRE_point3D_equal(b, d) || RCRE_point3D_equal(c, d)) {
        return true;
    }

    RCRE_point3D ab = {0};
    RCRE_point3D ac = {0};
    RCRE_point3D ad = {0};

    RCRE_point3D_subtract(a, b, &ab);
    RCRE_point3D_subtract(a, c, &ac);
    RCRE_point3D_subtract(a, d, &ad);

    RCRE_point3D_crossProduct(&ab, &ac, &ac);
    RCRE_point3D_crossProduct(&ab, &ad, &ad);

    double diff = RCRE_point3D_distance(&ac, &ad);
    RCRE_point3D_scale(&ac, -1, &ac);
    diff = fmin(diff, RCRE_point3D_distance(&ac, &ad));
    double minimumSignificant = RCRE_point3D_distanceToOrigin(&ab) / 1048576.0;
    // 1048576.0: not a special number just close to 1e6 and a power of 2 for faster floating point math

    return diff < minimumSignificant;
}

PRE_DEVICE bool RCRE_point3D_averagePointsWithWeight(RCRE_point3D *a, RCRE_point3D *b, double weight, RCRE_point3D *out) {
    out->x = a->x * (1 - weight) + b->x * weight;
    out->y = a->y * (1 - weight) + b->y * weight;
    out->z = a->z * (1 - weight) + b->z * weight;
}

PRE_DEVICE void RCRE_point3D_rotatePointAroundAxis(RCRE_point3D *p, RCRE_point3D *axisDirection, RCRE_point3D *axisPoint,
                                        double angle, RCRE_point3D *out) {
    // result of rotating the point (x,y,z) about the line through (a,b,c) with direction vector ⟨u,v,w⟩ (where u^2 + v^2 + w^2 = 1) by the angle θ
    // newX = (a(v^2 + w^2) - u(bv + cw - ux - vy - wz))(1 - cos θ) + x cos θ + (- cv + bw - wy + vz) sin θ
    // newY = (b(u^2 + w^2) - v(au + cw - ux - vy - wz))(1 - cos θ) + y cos θ + (  cu - aw + wx - uz) sin θ
    // newZ = (c(u^2 + v^2) - w(au + bv - ux - vy - wz))(1 - cos θ) + z cos θ + (- bu + av - vx + uy) sin θ

    RCRE_point3D axisDirectionUnit = {0};
    RCRE_point3D_getUnit(axisDirection, &axisDirectionUnit);

    double cosTheta = cos(angle);
    double sinTheta = sin(angle);
    double a = axisPoint->x, b = axisPoint->y, c = axisPoint->z;
    double u = axisDirectionUnit.x, v = axisDirectionUnit.y, w = axisDirectionUnit.z;

    double newX = (a * (v * v + w * w) - u * (b * v + c * w - u * p->x - v * p->y - w * p->z)) * (1 - cosTheta) + p->x * cosTheta + (- c * v + b * w - w * p->y + v * p->z) * sinTheta;
    double newY = (b * (u * u + w * w) - v * (a * u + c * w - u * p->x - v * p->y - w * p->z)) * (1 - cosTheta) + p->y * cosTheta + (  c * u - a * w + w * p->x - u * p->z) * sinTheta;
    double newZ = (c * (u * u + v * v) - w * (a * u + b * v - u * p->x - v * p->y - w * p->z)) * (1 - cosTheta) + p->z * cosTheta + (- b * u + a * v - v * p->x + u * p->y) * sinTheta;

    out->x = newX;
    out->y = newY;
    out->z = newZ;
}

PRE_DEVICE double RCRE_point3D_getAngleBetweenPoints(RCRE_point3D *p1, RCRE_point3D *p2) {
    double cosAngle = RCRE_point3D_dotProduct(p1, p2) / (RCRE_point3D_distanceToOrigin(p1) * RCRE_point3D_distanceToOrigin(p2));
    return acos(cosAngle);
}
