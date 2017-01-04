#include <cuda.h>
#include <iostream>

#include "cuStereo.h"
#include "cuImproc.cuh"
#include "matcher_AEMBMv1.cuh"
//#include "matcher_AEMBMv2.cuh"

#include <stdlib.h>
#include <device_launch_parameters.h>


void cu::stereo::allocTemps(tempArr& temps, unsigned int w, unsigned int h, unsigned int c) {

    temps.tmpChars = (unsigned char**)malloc(temps.sizes[0] * sizeof(unsigned char*));
    for (int i = 0; i < temps.sizes[0]; i++) {
        cudaMalloc(&(temps.tmpChars[i]), w * h * c * sizeof(unsigned char));
    }
    temps.tmpInts = (int**)malloc(temps.sizes[1] * sizeof(int*));
    for (int i = 0; i < temps.sizes[1]; i++) {
        cudaMalloc(&(temps.tmpInts[i]), 64 * w * h * sizeof(int));
    }
    temps.tmpFloats = (float**)malloc(temps.sizes[2] * sizeof(float*));
    for (int i = 0; i < temps.sizes[2]; i++) {
        cudaMalloc(&(temps.tmpFloats[i]), w * h * sizeof(float));
    }
}

void cu::stereo::freeTemps(tempArr& temps) {
    for (int i = 0; i < temps.sizes[0]; i++) {
        cudaFree(temps.tmpChars[i]);
    }
    for (int i = 0; i < temps.sizes[1]; i++) {
        cudaFree(temps.tmpInts[i]);
    }
    for (int i = 0; i < temps.sizes[2]; i++) {
        cudaFree(temps.tmpFloats[i]);
    }
}

void cu::stereo::initStereoParams(stereoParams* params, unsigned int minDisp, unsigned int maxDisp, unsigned int edgeMultiplier, unsigned int threshold) {
    params->minDisp = minDisp;
    params->maxDisp = maxDisp;
    params->threshold = threshold;
    params->edgeMultiplier = edgeMultiplier;
}

void cu::stereo::match_AEMBM(
    cudaArray* left, cudaArray* right, float* out, tempArr& temps, stereoParams& params, const unsigned int w, const unsigned int h, unsigned int c) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaBindTextureToArray(improcTex1, right);
    improcTex1.addressMode[0] = cudaAddressModeClamp;
    improcTex1.addressMode[1] = cudaAddressModeClamp;
    improcTex1.filterMode = cudaFilterModePoint;
    improcTex1.normalized = false;

    cudaMemset(temps.tmpChars[0], 0, w * h);
    cudaMemset(temps.tmpChars[1], 0, w * h);
    cudaMemset(temps.tmpChars[2], 21, w * h);
    // cudaBindTextureToArray(texRight, right);
    dim3 thPerBlock(16, 16);
    dim3 blocks(w / thPerBlock.x, h / thPerBlock.y);
    cu::common::global::uchar4texToUchar1 << <blocks, thPerBlock>>> (temps.tmpChars[0], w, h);
    cu::improc::global::edgeDetect << <blocks, thPerBlock>>> (temps.tmpChars[0], temps.tmpChars[1], 20, w, h);
    cudaDeviceSynchronize();

    dim3 threadsPerBlock24(24, 24);
    dim3 numbBlocks24(w / threadsPerBlock24.x, h / threadsPerBlock24.y);
    cu::improc::global::findDistanceFast << <numbBlocks24, threadsPerBlock24, 64 * 64 * sizeof(unsigned char)>>> (temps.tmpChars[1], temps.tmpChars[2], w,
            h);
    cudaDeviceSynchronize();
    cudaUnbindTexture(improcTex1);

    cudaBindTextureToArray(texLeft, right);
    texLeft.addressMode[0] = cudaAddressModeClamp;
    texLeft.addressMode[1] = cudaAddressModeClamp;
    texLeft.filterMode = cudaFilterModePoint;
    texLeft.normalized = false;
    cudaBindTextureToArray(texRight, left);
    texRight.addressMode[0] = cudaAddressModeClamp;
    texRight.addressMode[1] = cudaAddressModeClamp;
    texRight.filterMode = cudaFilterModePoint;
    texRight.normalized = false;

    //TODO: DISPARITY CALCULATION ADAPTIVE-EDGEBASED MULTIBLOCKSIZE ALGORITHM V1
    dim3 thPerBlock8(32, 8);
    dim3 blocks8(w / thPerBlock8.x, h / thPerBlock8.y);
    macher::global::match8x8<<<blocks8, thPerBlock8>>>((unsigned short*)temps.tmpInts[0], w, h);
    //cudaDeviceSynchronize();

       dim3 thPerBlock3(66, 10);
       dim3 blocks3(w / 64, h / 8);
       macher::global::match3x3<<<blocks3, thPerBlock3>>>((unsigned short*)temps.tmpInts[2], w, h);
       cudaDeviceSynchronize();

      dim3 thPerBlock32(64);
      dim3 blocks32(w / 16, h / 16);
      macher::global::match8to32<<<blocks32, thPerBlock32>>>((unsigned short*)temps.tmpInts[0], (unsigned int*)temps.tmpInts[1], w, h);
      cudaDeviceSynchronize();

      dim3 thPerBlockDisp(16, 16);
      dim3 blocksDisp(w / thPerBlockDisp.x, h / thPerBlockDisp.y);
      macher::global::matcher_AEMBMv1<< <blocksDisp, thPerBlockDisp>>>(temps.tmpChars[2],(unsigned short*)temps.tmpInts[2], (unsigned int*)temps.tmpInts[1], out, w, h);
      cudaDeviceSynchronize();

      cudaUnbindTexture(texRight);
      cudaUnbindTexture(texLeft);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    if(DEBUG_EN){
        std::cout << milliseconds << "ms elapsed" << std::endl;
    }
}

void cu::stereo::match_NEW(
        cudaArray* left, cudaArray* right, float* out, tempArr& temps, stereoParams& params, const unsigned int w, const unsigned int h, unsigned int c) {
    cudaBindTextureToArray(improcTex1, right);
    improcTex1.addressMode[0] = cudaAddressModeClamp;
    improcTex1.addressMode[1] = cudaAddressModeClamp;
    improcTex1.filterMode = cudaFilterModePoint;
    improcTex1.normalized = false;

    cudaMemset(temps.tmpChars[0], 0, w * h);
    cudaMemset(temps.tmpChars[1], 0, w * h);
    cudaMemset(temps.tmpChars[2], 21, w * h);
    // cudaBindTextureToArray(texRight, right);
    dim3 thPerBlock(16, 16);
    dim3 blocks(w / thPerBlock.x, h / thPerBlock.y);
    cu::common::global::uchar4texToUchar1 << <blocks, thPerBlock>>> (temps.tmpChars[0], w, h);
    cu::improc::global::edgeDetect << <blocks, thPerBlock>>> (temps.tmpChars[0], temps.tmpChars[1], 50, w, h);
    cudaDeviceSynchronize();

    dim3 threadsPerBlock24(24, 24);
    dim3 numbBlocks24(w / threadsPerBlock24.x, h / threadsPerBlock24.y);
    cu::improc::global::findDistanceFast << <numbBlocks24, threadsPerBlock24, 64 * 64 * sizeof(unsigned char)>>> (temps.tmpChars[1], temps.tmpChars[2], w,
            h);
    cudaDeviceSynchronize();
    cudaUnbindTexture(improcTex1);

    cudaBindTextureToArray(texLeft, right);
    texLeft.addressMode[0] = cudaAddressModeClamp;
    texLeft.addressMode[1] = cudaAddressModeClamp;
    texLeft.filterMode = cudaFilterModePoint;
    texLeft.normalized = false;
    cudaBindTextureToArray(texRight, left);
    texRight.addressMode[0] = cudaAddressModeClamp;
    texRight.addressMode[1] = cudaAddressModeClamp;
    texRight.filterMode = cudaFilterModePoint;
    texRight.normalized = false;

    //TODO: DISPARITY CALCULATION ADAPTIVE-EDGEBASED MULTIBLOCKSIZE ALGORITHM V2
    dim3 threadsPerBlockDisp(32, 8);
    dim3 numbBlocksDisp(w / threadsPerBlockDisp.x, h / threadsPerBlockDisp.y);
    //macher::global::matcher_AEMBMv2<< <numbBlocksDisp, threadsPerBlockDisp>>>(temps.tmpChars[2], temps.tmpChars[0], w, h, 0, 64);

    cu::common::global::uchar1toFloat << <blocks, thPerBlock>>> (temps.tmpChars[0], out, w, h, 1 / 255.0f);
    // copyTexToOut<<<blocks, thPerBlock>>>(w, h, out);

    cudaDeviceSynchronize();
    cudaUnbindTexture(texRight);
    cudaUnbindTexture(texLeft);
}


