#pragma once
#include <cuda_runtime.h>

namespace cu {
namespace stereo {
struct tempArr {
    unsigned int sizes[3];
    int **tmpInts;
    unsigned char **tmpChars;
    float **tmpFloats;
};

struct stereoParams {
    unsigned int minDisp;
    unsigned int maxDisp;
    unsigned int edgeMultiplier;
    unsigned int threshold;
};

void allocTemps(tempArr &temps, unsigned int w, unsigned int h, unsigned int c);
void freeTemps(tempArr &temps);
void match_AEMBM(cudaArray *left, cudaArray *right, float *out, tempArr &temps,
                 stereoParams &params, unsigned int w, unsigned int h,
                 unsigned int c);
void match_NEW(cudaArray *left, cudaArray *right, float *out, tempArr &temps,
               stereoParams &params, unsigned int w, unsigned int h,
               unsigned int c);
void initStereoParams(stereoParams *params, unsigned int minDisp,
                      unsigned int maxDisp, unsigned int edgeMultiplier,
                      unsigned int threshold);
}
}
