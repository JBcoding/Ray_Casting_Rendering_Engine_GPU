//
// Created by Mads Bjoern on 2019-07-18.
//

#include "RCRE_engine.h"
#include "RCRE_constants.h"
#include "RCRE_cuda.h"

void RCRE_engine_getImage(RCRE_point3D *cameraPosition, RCRE_point3D *cameraDirection, RCRE_point3D *cameraUpDirection,
                          int nModels, RCRE_model3D **models, int width, int height, double angleOfView,
                          char *imageBuffer) {
    RCRE_point3D cameraRightDirection = {0};
    RCRE_point3D cameraForwardDirection = {0};
    RCRE_point3D cameraUpwardsDirection = {0};

    RCRE_point3D_crossProduct(cameraDirection, cameraUpDirection, &cameraRightDirection);

    RCRE_point3D_getUnit(cameraDirection, &cameraForwardDirection);
    RCRE_point3D_getUnit(&cameraRightDirection, &cameraRightDirection);
    RCRE_point3D_getUnit(cameraUpDirection, &cameraUpwardsDirection);

    double leftAndRightEffect = tan(angleOfView / 2);
    double upAndDownEffect = leftAndRightEffect * (height / (double)width);

#ifdef RCRE_CUDA
    RCRE_point3D *cameraPositionCUDA = RCRE_point3D_toCUDA(cameraPosition);
    RCRE_point3D *cameraForwardDirectionCUDA = RCRE_point3D_toCUDA(&cameraForwardDirection);
    RCRE_point3D *cameraRightDirectionCUDA = RCRE_point3D_toCUDA(&cameraRightDirection);
    RCRE_point3D *cameraUpwardsDirectionCUDA = RCRE_point3D_toCUDA(&cameraUpwardsDirection);

    char *imageBufferCUDA;
    cudaMalloc((void**)&imageBufferCUDA, sizeof(width * height * 4));

    RCRE_model3D **modelsCUDA;
    cudaMalloc((void**)&modelsCUDA, sizeof(RCRE_model3D *) * nModels);
    RCRE_model3D **tempModelList = malloc(sizeof(RCRE_model3D *) * nModels);
    for (int i = 0; i < nModels; i ++) {
        tempModelList[i] = RCRE_model3D_toCUDA(models[i]);
    }
    cudaMemcpy(modelsCUDA, tempModelList, sizeof(RCRE_model3D *) * nModels, cudaMemcpyHostToDevice);
    free(tempModelList);

    RCRE_engine_fillPixels<<<1,1>>>(cameraPositionCUDA, cameraForwardDirectionCUDA, cameraRightDirectionCUDA, cameraUpwardsDirectionCUDA,
                           width, height, leftAndRightEffect, upAndDownEffect, nModels, modelsCUDA, imageBufferCUDA);

    cudaMemcpy(imageBuffer, imageBufferCUDA, sizeof(width * height * 4), cudaMemcpyDeviceToHost);
#else
    RCRE_engine_fillPixels(cameraPosition, &cameraForwardDirection, &cameraRightDirection, &cameraUpwardsDirection,
                           width, height, leftAndRightEffect, upAndDownEffect, nModels, models, imageBuffer);
#endif
}


PRE_GLOBAL PRE_DEVICE void RCRE_engine_fillPixels(RCRE_point3D *cameraPosition, RCRE_point3D *cameraForwardDirection,
                                                  RCRE_point3D *cameraRightDirection, RCRE_point3D *cameraUpwardsDirection,
                                                  int width, int height, double leftAndRightEffect, double upAndDownEffect,
                                                  int nModels, RCRE_model3D **models, char *imageBuffer) {
    for (int p = 0; p < height * width; p ++) {
        int h = p % height;
        int w = p / height; // integer division
        RCRE_point3D direction = {0};
        RCRE_point3D leftRight = {0};
        RCRE_point3D upDown = {0};

        RCRE_point3D_add(&direction, cameraForwardDirection, &direction);

        double lre = (w - (width - 1) / 2.0) / (double) ((width - 1) / 2.0);
        double ude = (h - (height - 1) / 2.0) / (double) ((height - 1) / 2.0);

        lre *= leftAndRightEffect;
        ude *= upAndDownEffect;

        RCRE_point3D_scale(cameraRightDirection, lre, &leftRight);
        RCRE_point3D_scale(cameraUpwardsDirection, -ude, &upDown);

        RCRE_point3D_add(&direction, &leftRight, &direction);
        RCRE_point3D_add(&direction, &upDown, &direction);

        RCRE_color color = {0};

        RCRE_engine_getColorFromRay(nModels, models, cameraPosition, &direction, 1.0, 0, &color);

        imageBuffer[h * width * 4 + w * 4 + 0] = RCRE_color_getR(&color);
        imageBuffer[h * width * 4 + w * 4 + 1] = RCRE_color_getG(&color);
        imageBuffer[h * width * 4 + w * 4 + 2] = RCRE_color_getB(&color);
        imageBuffer[h * width * 4 + w * 4 + 3] = 255;
    }
}

PRE_DEVICE void RCRE_engine_getColorFromRay(int nModels, RCRE_model3D **models, RCRE_point3D *rayOrigin,
                                 RCRE_point3D *rayDirection, double totalWeight, int recursiveCount,
                                 RCRE_color *outColor) {

    outColor->r = 0.0;
    outColor->g = 0.0;
    outColor->b = 0.0;

    if (recursiveCount > 20 || totalWeight < 0.001) {
        return;
    }

    double bestDistance = DBL_MAX;

    RCRE_point3D bestIntersectionPoint = {0};
    RCRE_point3D bestReflectiveDirection = {0};
    RCRE_point3D bestExitPoint = {0};
    RCRE_point3D bestExitDirection = {0};
    RCRE_model3D *bestModel = NULL;

    for (int m = 0; m < nModels; m++) {
        RCRE_point3D outIntersectionPoint = {0};
        RCRE_point3D outReflectiveDirection = {0};
        RCRE_point3D outExitPoint = {0};
        RCRE_point3D outExitDirection = {0};
        if (RCRE_model3D_getIntersection(models[m], rayOrigin, rayDirection, &outIntersectionPoint, &outReflectiveDirection, &outExitPoint, &outExitDirection)) {
            double distance = RCRE_point3D_distance(rayOrigin, &outIntersectionPoint);
            if (distance > 0.000000001 && distance < bestDistance) {
                bestDistance = distance;
                RCRE_point3D_copyInto(&outIntersectionPoint, &bestIntersectionPoint);
                RCRE_point3D_copyInto(&outReflectiveDirection, &bestReflectiveDirection);
                RCRE_point3D_copyInto(&outExitPoint, &bestExitPoint);
                RCRE_point3D_copyInto(&outExitDirection, &bestExitDirection);
                bestModel = models[m];
            }
        }
    }

    if (bestDistance == DBL_MAX) {
        return;
    }

    double distanceInsideModel = RCRE_point3D_distance(&bestIntersectionPoint, &bestExitPoint);

    double weightReflectiveColor = bestModel->reflectivity;
    double weightModelColorBeforeReflective = (1 - pow(bestModel->transparency, distanceInsideModel));
    double weightModelColor = weightModelColorBeforeReflective * (1 - weightReflectiveColor);
    double weightContinuedColor = 1 - weightReflectiveColor - weightModelColor;

    RCRE_color reflectiveColor = {0};
    RCRE_color continuedColor = {0};

    RCRE_engine_getColorFromRay(nModels, models, &bestIntersectionPoint, &bestReflectiveDirection,
                                weightReflectiveColor * totalWeight, recursiveCount + 1, &reflectiveColor);
    RCRE_engine_getColorFromRay(nModels, models, &bestExitPoint, &bestExitDirection,
                                weightContinuedColor * totalWeight, recursiveCount + 1, &continuedColor);


    RCRE_color_mix(outColor, &continuedColor, 1.0, outColor);
    RCRE_color_mix(outColor, bestModel->color, weightModelColorBeforeReflective, outColor);
    RCRE_color_mix(outColor, &reflectiveColor, weightReflectiveColor, outColor);
}

