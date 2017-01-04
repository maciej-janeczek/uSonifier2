#pragma once

#include <cuda.h>
#include <cuda_runtime.h>


namespace cu {
    typedef struct {
        unsigned int width;
        unsigned int height;
        unsigned int channels;
        cudaArray* left;
        cudaArray* right;
        float* disparity;
        uchar1* udisp;
        cudaChannelFormatDesc channelDesc1UC;
        cudaChannelFormatDesc channelDesc2UC;
        cudaChannelFormatDesc channelDescF;
    } data_gpu;

    void init(data_gpu* data, unsigned int h, unsigned int w, unsigned int c);
    int update(data_gpu* data, unsigned char* right_cpu, unsigned char* left_cpu);
    int grab(data_gpu* data, float* out_cpu);
}

