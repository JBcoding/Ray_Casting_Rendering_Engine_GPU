//
// Created by Mads Bjoern on 2019-07-18.
//

#ifndef RAYCASTINGRENDERINGENGINEGPU_RCRE_COLOR_H
#define RAYCASTINGRENDERINGENGINEGPU_RCRE_COLOR_H

#include "RCRE_constants.h"

struct RCRE_color {
    double r, g, b;
} typedef RCRE_color;

PRE_DEVICE RCRE_color *RCRE_color_getColorFromValues(double r, double g, double b);

PRE_DEVICE void RCRE_color_mix(RCRE_color *a, RCRE_color *b, double weight, RCRE_color *out);

PRE_DEVICE char RCRE_color_getR(RCRE_color *c);
PRE_DEVICE char RCRE_color_getG(RCRE_color *c);
PRE_DEVICE char RCRE_color_getB(RCRE_color *c);


#endif //RAYCASTINGRENDERINGENGINEGPU_RCRE_COLOR_H
