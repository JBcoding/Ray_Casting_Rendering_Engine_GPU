//
// Created by Mads Bjoern on 2019-07-18.
//

#include "RCRE_point3D.h"
#include "RCRE_model3D.h"

#ifndef RAYCASTINGRENDERINGENGINEGPU_RCRE_ENGINE_H
#define RAYCASTINGRENDERINGENGINEGPU_RCRE_ENGINE_H

void RCRE_engine_getImage(RCRE_point3D *cameraPosition, RCRE_point3D *cameraDirection, RCRE_point3D *cameraUpDirection,
                          int nModels, RCRE_model3D **models, int width, int height, double angleOfView,
                          char *imageBuffer);

PRE_GLOBAL void RCRE_engine_fillPixels(RCRE_point3D *cameraPosition, RCRE_point3D *cameraForwardDirection,
                                                  RCRE_point3D *cameraRightDirection, RCRE_point3D *cameraUpwardsDirection,
                                                  int width, int height, double leftAndRightEffect, double upAndDownEffect,
                                                  int nModels, RCRE_model3D **models, char *imageBuffer);

PRE_DEVICE void RCRE_engine_getColorFromRay(int nModels, RCRE_model3D **models, RCRE_point3D *rayOrigin,
                                 RCRE_point3D *rayDirection, double totalWeight, int recursiveCount,
                                 RCRE_color *outColor);

#endif //RAYCASTINGRENDERINGENGINEGPU_RCRE_ENGINE_H
