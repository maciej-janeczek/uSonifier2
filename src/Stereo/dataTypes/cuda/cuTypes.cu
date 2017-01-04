
#include "cuTypes.h"

void cu::init(data_gpu* data, unsigned int h, unsigned int w, unsigned int c) {
    data->height = h;
    data->width = w;
    data->channels = c;
    switch (c) {
    case 1:
        data->channelDesc1UC = cudaCreateChannelDesc<unsigned char>();
        data->channelDesc2UC = cudaCreateChannelDesc<unsigned char>();
        break;
    case 3:
        data->channelDesc1UC = cudaCreateChannelDesc<uchar3>();
        data->channelDesc2UC = cudaCreateChannelDesc<uchar3>();
        break;
    case 4:
        data->channelDesc1UC = cudaCreateChannelDesc<uchar4>();
        data->channelDesc2UC = cudaCreateChannelDesc<uchar4>();
        break;
    }
    data->channelDescF = cudaCreateChannelDesc<float>();

    cudaMallocArray(&(data->left), &(data->channelDesc1UC), data->width, data->height);
    cudaMallocArray(&(data->right), &(data->channelDesc2UC), data->width, data->height);
    cudaMalloc(&(data->disparity), w * h * sizeof(float));
    cudaMalloc(&(data->udisp), w * 64 * sizeof(uchar1));
}

int cu::update(data_gpu* data, unsigned char* left_cpu, unsigned char* right_cpu) {
    size_t size = (data->height) * (data->width) * (data->channels) * sizeof(unsigned char);
    cudaMemcpyToArray(data->left, 0, 0, left_cpu, size, cudaMemcpyHostToDevice);
    cudaMemcpyToArray(data->right, 0, 0, right_cpu, size, cudaMemcpyHostToDevice);
    return 1;
    // TODO: error handle
}

int cu::grab(data_gpu* data, float* out_cpu) {
    size_t size = (data->height) * (data->width) * sizeof(float);
    cudaMemcpy(out_cpu, data->disparity, size, cudaMemcpyDeviceToHost);
    return 1;
    // TODO: error handle
}
