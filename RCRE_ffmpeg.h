//
// Created by Mads Bjoern on 2019-07-18.
//

#ifndef RAYCASTINGRENDERINGENGINEGPU_RCRE_FFMPEG_H
#define RAYCASTINGRENDERINGENGINEGPU_RCRE_FFMPEG_H

#include "RCRE_constants.h"

static char *RCRE_ffmpeg_ffmpegPath = "ffmpeg";

void RCRE_ffmpeg_setFfmpegPath(char *path);

FILE *RCRE_ffmpeg_getImageWriter(int width, int height, char *imageName);
FILE *RCRE_ffmpeg_getVideoWriter(int width, int height, int fps, char *videoName);

void RCRE_ffmpeg_writeToFile(FILE *process, char *data, int width, int height);
void RCRE_ffmpeg_closeFile(FILE *process);


#endif //RAYCASTINGRENDERINGENGINEGPU_RCRE_FFMPEG_H
