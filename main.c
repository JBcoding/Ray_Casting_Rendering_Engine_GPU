#include <stdio.h>
#include <stdlib.h>

#include "RCRE_point3D.h"
#include "RCRE_convexPolyhedron.h"
#include "RCRE_transformationMatrix.h"
#include "RCRE_sphere.h"
#include "RCRE_model3D.h"
#include "RCRE_constants.h"
#include "RCRE_engine.h"
#include "RCRE_ffmpeg.h"

int main() {

    double xmin = 0.5, xmax = 1.5;
    double ymin = 0.5, ymax = 1;
    double zmin = 0.5, zmax = 1;

    RCRE_point3D *p1 = RCRE_point3D_getPointFromValues(xmin, ymin, zmin);
    RCRE_point3D *p2 = RCRE_point3D_getPointFromValues(xmin, ymin, zmax);
    RCRE_point3D *p3 = RCRE_point3D_getPointFromValues(xmin, ymax, zmin);
    RCRE_point3D *p4 = RCRE_point3D_getPointFromValues(xmin, ymax, zmax);
    RCRE_point3D *p5 = RCRE_point3D_getPointFromValues(xmax, ymin, zmin);
    RCRE_point3D *p6 = RCRE_point3D_getPointFromValues(xmax, ymin, zmax);
    RCRE_point3D *p7 = RCRE_point3D_getPointFromValues(xmax, ymax, zmin);
    RCRE_point3D *p8 = RCRE_point3D_getPointFromValues(xmax, ymax, zmax);

    RCRE_point3D **points = (RCRE_point3D **)malloc(sizeof(RCRE_point3D*) * 8);

    points[0] = p1;
    points[1] = p2;
    points[2] = p3;
    points[3] = p4;
    points[4] = p5;
    points[5] = p6;
    points[6] = p7;
    points[7] = p8;

    RCRE_convexPolyhedron *cp = RCRE_convexPolyhedron_getConvexPolyhedronFromPoints(8, points);

    printf("%d\n", RCRE_convexPolyhedron_isPointContainedWithin(cp, RCRE_point3D_getPointFromValues(.5, .5, 1.5)));

    RCRE_transformationMatrix *a = RCRE_transformationMatrix_getTransformationMatrixFromValues(2, 1, 3, 4, 5, 6, 7, 8, 9);
    RCRE_transformationMatrix *b = RCRE_transformationMatrix_getTransformationMatrixFromValues(10, 20, 30, 40, 50, 60, 70, 80, 90);
    RCRE_transformationMatrix_inverse(RCRE_transformationMatrix_getIdentityMatrix(), a);

    printf("%lf %lf %lf \n%lf %lf %lf \n%lf %lf %lf \n", a->a, a->b, a->c, a->d, a->e, a->f, a->g, a->h, a->i);


    RCRE_sphere *s = RCRE_sphere_getSphereFromPointAndRadius(p8, 1);
    printf("%d\n", RCRE_sphere_isPointContainedWithin(s, RCRE_point3D_getPointFromValues(1.71, 1.5, 1.5)));

    RCRE_point3D o;
    RCRE_point3D_rotatePointAroundAxis(RCRE_point3D_getPointFromValues(1, 0, 0), RCRE_point3D_getPointFromValues(0, 1, 0), RCRE_point3D_getPointFromValues(0, 0, 0), M_PI / 8, &o);
    printf("AAA %lf %lf %lf   %lf\n", o.x, o.y, o.z, RCRE_point3D_getAngleBetweenPoints(&o, RCRE_point3D_getPointFromValues(1, 0, 0)));

    RCRE_triangle3D *t = RCRE_triangle3D_getTriangleFromPoints(RCRE_point3D_getPointFromValues(1, 0, 0), RCRE_point3D_getPointFromValues(0, 0, 1), RCRE_point3D_getPointFromValues(0, 1, 0));



    printf("\n\n\n");

    RCRE_sphere *ss = RCRE_sphere_getSphereFromPointAndRadius(RCRE_point3D_getPointFromValues(1, 0.5, 0.5), .5);
    RCRE_transformationMatrix *tm = RCRE_transformationMatrix_getScalingMatrix(1, 1, 1);

    RCRE_model3D *model3D = RCRE_model3D_getModel(RCRE_transformationMatrix_getScalingMatrix(.8, .8, .8), RCRE_color_getColorFromValues(1, 0, 0), 0, .2, 1.3);
    RCRE_model3D_insertEntry(model3D, ss, RCRE_model3D_DATATYPE_SPHERE, false, tm);
    RCRE_model3D_insertEntry(model3D, cp, RCRE_model3D_DATATYPE_CONVEX_POLYHEDRON, true, RCRE_transformationMatrix_getIdentityMatrix());


    int nModels = 101;
    RCRE_model3D **models = (RCRE_model3D **)malloc(sizeof(RCRE_model3D*) * nModels);


    for (int i = 0; i < 10; i ++) {
        for (int j = 0; j < 10; j ++) {
            RCRE_point3D *p1 = RCRE_point3D_getPointFromValues(6, 0 - 5 + i, 0 - 5 + j);
            RCRE_point3D *p2 = RCRE_point3D_getPointFromValues(6, 0 - 5 + i, 1 - 5 + j);
            RCRE_point3D *p3 = RCRE_point3D_getPointFromValues(6, 1 - 5 + i, 0 - 5 + j);
            RCRE_point3D *p4 = RCRE_point3D_getPointFromValues(6, 1 - 5 + i, 1 - 5 + j);
            RCRE_point3D *p5 = RCRE_point3D_getPointFromValues(7, 0 - 5 + i, 0 - 5 + j);
            RCRE_point3D *p6 = RCRE_point3D_getPointFromValues(7, 0 - 5 + i, 1 - 5 + j);
            RCRE_point3D *p7 = RCRE_point3D_getPointFromValues(7, 1 - 5 + i, 0 - 5 + j);
            RCRE_point3D *p8 = RCRE_point3D_getPointFromValues(7, 1 - 5 + i, 1 - 5 + j);

            RCRE_point3D **points = (RCRE_point3D **)malloc(sizeof(RCRE_point3D*) * 8);

            points[0] = p1;
            points[1] = p2;
            points[2] = p3;
            points[3] = p4;
            points[4] = p5;
            points[5] = p6;
            points[6] = p7;
            points[7] = p8;

            RCRE_convexPolyhedron *cp = RCRE_convexPolyhedron_getConvexPolyhedronFromPoints(8, points);


            RCRE_model3D *model3D = RCRE_model3D_getModel(RCRE_transformationMatrix_getScalingMatrix(.8, .8, .8), RCRE_color_getColorFromValues(0, i % 2, j % 2), 0, 0, 1);
            RCRE_model3D_insertEntry(model3D, cp, RCRE_model3D_DATATYPE_CONVEX_POLYHEDRON, false, RCRE_transformationMatrix_getIdentityMatrix());

            models[i * 10 + j] = model3D;

        }
    }



    RCRE_point3D *centerPoint = RCRE_point3D_getPointFromValues(1.5, 0.5, 0.5);

    RCRE_point3D *cameraPosition = RCRE_point3D_getPointFromValues(0, 0.5, 0.5);
    RCRE_point3D *cameraDirection = RCRE_point3D_getPointFromValues(1, 0, 0);
    RCRE_point3D *cameraUpDirection = RCRE_point3D_getPointFromValues(0, -1, 0);
    models[100] = model3D;
    int width = 1920;
    int height = 1080;
    double angleOfView = M_PI / 2;
    char *imageBuffer = (char *)malloc(width * height * 4);

    //FILE *videoWriter = RCRE_ffmpeg_getVideoWriter(width, height, 24, "output");
    FILE *imgWriter = RCRE_ffmpeg_getImageWriter(width, height, "output");
    for (int i = 280/*220*/; i < 281/*330*/; i += 1) {
        free(cameraPosition);
        cameraPosition = RCRE_point3D_getPointFromValues(sin(i / 180.0 * M_PI) * 2 + 1.5, 0.5, cos(i / 180.0 * M_PI) * 2 + 0.5);
        RCRE_point3D_subtract(centerPoint, cameraPosition, cameraDirection);

        RCRE_engine_getImage(cameraPosition, cameraDirection, cameraUpDirection, nModels, models, width, height, angleOfView, imageBuffer);
        //RCRE_ffmpeg_writeToFile(videoWriter, imageBuffer, width, height);
        RCRE_ffmpeg_writeToFile(imgWriter, imageBuffer, width, height);
    }
    //RCRE_ffmpeg_closeFile(videoWriter);
    RCRE_ffmpeg_closeFile(imgWriter);

    return 0;
}