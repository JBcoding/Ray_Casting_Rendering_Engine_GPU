//
// Created by Mads Bjoern on 2019-07-18.
//

#include "RCRE_ffmpeg.h"

void RCRE_ffmpeg_setFfmpegPath(char *path) {
    RCRE_ffmpeg_ffmpegPath = path;
}


FILE *RCRE_ffmpeg_getImageWriter(int width, int height, char *imageName) {
    char command[1024];
    snprintf(command, sizeof(command), "%s -y -f rawvideo -pixel_format rgba -video_size %dx%d -i - %s.png", RCRE_ffmpeg_ffmpegPath, width, height, imageName);
    return popen(command, "w");
}

FILE *RCRE_ffmpeg_getVideoWriter(int width, int height, int fps, char *videoName) {
    char command[1024];
    snprintf(command, sizeof(command), "%s -r %d -f rawvideo -pixel_format rgba -s %dx%d -i - -preset fast -y -pix_fmt yuv420p -crf 1 %s.mp4", RCRE_ffmpeg_ffmpegPath, fps, width, height, videoName);
    return popen(command, "w");
}


void RCRE_ffmpeg_writeToFile(FILE *process, char *data, int width, int height) {
    fwrite(data, 1, width * height * 4, process);
}

void RCRE_ffmpeg_closeFile(FILE *process) {
    pclose(process);
}
