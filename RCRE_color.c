//
// Created by Mads Bjoern on 2019-07-18.
//

#include "RCRE_color.h"
#include "RCRE_constants.h"

RCRE_color *RCRE_color_getColorFromValues(double r, double g, double b) {
    RCRE_color *color = malloc(sizeof(RCRE_color));
    color->r = r;
    color->g = g;
    color->b = b;
    return color;
}

void RCRE_color_mix(RCRE_color *a, RCRE_color *b, double weight, RCRE_color *out) {
    out->r = a->r * (1 - weight) + b->r * weight;
    out->g = a->g * (1 - weight) + b->g * weight;
    out->b = a->b * (1 - weight) + b->b * weight;
}


char RCRE_color_getR(RCRE_color *c) {
    return (char)(c->r * 255);
}

char RCRE_color_getG(RCRE_color *c) {
    return (char)(c->g * 255);
}

char RCRE_color_getB(RCRE_color *c) {
    return (char)(c->b * 255);
}
