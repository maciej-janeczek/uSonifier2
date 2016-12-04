#pragma once
#include <cuda_runtime.h>
#include "../dataTypes/view.hpp"
#include "cuImproc_device.cuh"

#include "common.cuh"




namespace cu {
    namespace improc {
        namespace global {
            __global__ void edgeDetect(unsigned char* in, unsigned char* out, int th, const unsigned int w, const unsigned int h);
            __global__ void findDistanceFast(unsigned char* edge, unsigned char* out, const int w, const int h);
        }
    }
}


__global__ void cu::improc::global::edgeDetect(unsigned char* in, unsigned char* out, int th, const unsigned int w, const unsigned int h) {
    /// Calculates binary edge image (0, 255) of grayscale input image in with texture threshold th.
    /// Use kernel 16x16.

    __shared__ unsigned char s[22][22];

    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int t = w * y + x;

    const int iX = threadIdx.x + 3;
    const int iY = threadIdx.y + 3;
    // Possibly redundant
    // if(x < 0 || x >= w-1 || y < 0 || y >= h-1) return;

    // Load to shared memory
    cu::common::device::loadChunkToShared_UC1_16x16_b3(in, s, w, h, x, y, iX, iY);
    __syncthreads();

    // Calculate first and second derivatives sobel values
    int sumYf = 0;
    int sumYs = 0;
    int sumYss = 0;
    int sumYsm = 0;
    int sumXf = 0;
    int sumXs = 0;
    int sumXss = 0;
    int sumXsm = 0;
    //__syncthreads();
    // Y first
    sumYf -= s[iX - 1][iY - 1];
    sumYf -= 2 * s[iX][iY - 1];
    sumYf -= s[iX + 1][iY - 1];
    sumYf += s[iX - 1][iY + 1];
    sumYf += 2 * s[iX][iY + 1];
    sumYf += s[iX + 1][iY + 1];
    // Y second
    sumYs += s[iX - 2][iY - 2];
    sumYs += 4 * s[iX - 1][iY - 2];
    sumYs += 6 * s[iX][iY - 2];
    sumYs += 4 * s[iX + 1][iY - 2];
    sumYs += s[iX + 2][iY - 2];

    sumYs -= 2 * s[iX - 2][iY];
    sumYs -= 8 * s[iX - 1][iY];
    sumYs -= 12 * s[iX][iY];
    sumYs -= 8 * s[iX + 1][iY];
    sumYs -= 2 * s[iX + 2][iY];

    sumYs += s[iX - 2][iY + 2];
    sumYs += 4 * s[iX - 1][iY + 2];
    sumYs += 6 * s[iX][iY + 2];
    sumYs += 4 * s[iX + 1][iY + 2];
    sumYs += s[iX + 2][iY + 2];
    // Y-1 second
    sumYsm += s[iX - 2][iY - 3];
    sumYsm += 4 * s[iX - 1][iY - 3];
    sumYsm += 6 * s[iX][iY - 3];
    sumYsm += 4 * s[iX + 1][iY - 3];
    sumYsm += s[iX + 2][iY - 3];

    sumYsm -= 2 * s[iX - 2][iY - 1];
    sumYsm -= 8 * s[iX - 1][iY - 1];
    sumYsm -= 12 * s[iX][iY - 1];
    sumYsm -= 8 * s[iX + 1][iY - 1];
    sumYsm -= 2 * s[iX + 2][iY - 1];

    sumYsm += s[iX - 2][iY + 1];
    sumYsm += 4 * s[iX - 1][iY + 1];
    sumYsm += 6 * s[iX][iY + 1];
    sumYsm += 4 * s[iX + 1][iY + 1];
    sumYsm += s[iX + 2][iY + 1];
    // Y+1 second
    sumYss += s[iX - 2][iY - 1];
    sumYss += 4 * s[iX - 1][iY - 1];
    sumYss += 6 * s[iX][iY - 1];
    sumYss += 4 * s[iX + 1][iY - 1];
    sumYss += s[iX + 2][iY - 1];

    sumYss -= 2 * s[iX - 2][iY + 1];
    sumYss -= 8 * s[iX - 1][iY + 1];
    sumYss -= 12 * s[iX][iY + 1];
    sumYss -= 8 * s[iX + 1][iY + 1];
    sumYss -= 2 * s[iX + 2][iY + 1];

    sumYss += s[iX - 2][iY + 3];
    sumYss += 4 * s[iX - 1][iY + 3];
    sumYss += 6 * s[iX][iY + 3];
    sumYss += 4 * s[iX + 1][iY + 3];
    sumYss += s[iX + 2][iY + 3];
    // X first
    sumXf -= s[iX - 1][iY - 1];
    sumXf -= 2 * s[iX - 1][iY];
    sumXf -= s[iX - 1][iY + 1];
    sumXf += s[iX + 1][iY - 1];
    sumXf += 2 * s[iX + 1][iY];
    sumXf += s[iX + 1][iY + 1];
    // X second
    sumXs += s[iX - 2][iY - 2];
    sumXs += 4 * s[iX - 2][iY - 1];
    sumXs += 6 * s[iX - 2][iY];
    sumXs += 4 * s[iX - 2][iY + 1];
    sumXs += s[iX - 2][iY + 2];

    sumXs -= 2 * s[iX][iY - 2];
    sumXs -= 8 * s[iX][iY - 1];
    sumXs -= 12 * s[iX][iY];
    sumXs -= 8 * s[iX][iY + 1];
    sumXs -= 2 * s[iX][iY + 2];

    sumXs += s[iX + 2][iY - 2];
    sumXs += 4 * s[iX + 2][iY - 1];
    sumXs += 6 * s[iX + 2][iY];
    sumXs += 4 * s[iX + 2][iY + 1];
    sumXs += s[iX + 2][iY + 2];
    // X-1 second
    sumXsm += s[iX - 3][iY - 2];
    sumXsm += 4 * s[iX - 3][iY - 1];
    sumXsm += 6 * s[iX - 3][iY];
    sumXsm += 4 * s[iX - 3][iY + 1];
    sumXsm += s[iX - 3][iY + 2];

    sumXsm -= 2 * s[iX - 1][iY - 2];
    sumXsm -= 8 * s[iX - 1][iY - 1];
    sumXsm -= 12 * s[iX - 1][iY];
    sumXsm -= 8 * s[iX - 1][iY + 1];
    sumXsm -= 2 * s[iX - 1][iY + 2];

    sumXsm += s[iX + 1][iY - 2];
    sumXsm += 4 * s[iX + 1][iY - 1];
    sumXsm += 6 * s[iX + 1][iY];
    sumXsm += 4 * s[iX + 1][iY + 1];
    sumXsm += s[iX + 1][iY + 2];
    // X+1 second
    sumXss += s[iX - 1][iY - 2];
    sumXss += 4 * s[iX - 1][iY - 1];
    sumXss += 6 * s[iX - 1][iY];
    sumXss += 4 * s[iX - 1][iY + 1];
    sumXss += s[iX - 1][iY + 2];

    sumXss -= 2 * s[iX + 1][iY - 2];
    sumXss -= 8 * s[iX + 1][iY - 1];
    sumXss -= 12 * s[iX + 1][iY];
    sumXss -= 8 * s[iX + 1][iY + 1];
    sumXss -= 2 * s[iX + 1][iY + 2];

    sumXss += s[iX + 3][iY - 2];
    sumXss += 4 * s[iX + 3][iY - 1];
    sumXss += 6 * s[iX + 3][iY];
    sumXss += 4 * s[iX + 3][iY + 1];
    sumXss += s[iX + 3][iY + 2];

    // zero near edges
    if (x < 3 || x >= w - 4 || y < 3 || y >= h - 4) {
        out[t] = 0;
        return;
    }

    // Choose correct value of the pixel for edge
    if (sumXf > th || sumXf < -th || sumYf > th || sumYf < -th) {
        if (sumXf > th || sumXf < -th) {
            if (sumXs == 0) {
                out[t] = 255;
            }
            if (!cu::common::device::sameSign(&sumXs, &sumXss)) {
                if ((sumXs + sumXss > 0 && sumXs < 0) || (sumXs + sumXss < 0 && sumXs > 0)) {
                    out[t] = 255;
                }
            }
            if (!cu::common::device::sameSign(&sumXsm, &sumXs)) {
                if ((sumXsm + sumXs < 0 && sumXs > 0) || (sumXs + sumXsm > 0 && sumXs < 0)) {
                    out[t] = 255;
                }
            }
        }
        if (sumYf > th || sumYf < -th) {
            if (sumYs == 0) {
                out[t] = 255;
            }
            if (!cu::common::device::sameSign(&sumYs, &sumYss)) {
                if ((sumYs + sumYss > 0 && sumYs < 0) || (sumYs + sumYss < 0 && sumYs > 0)) {
                    out[t] = 255;
                }
            }
            if (!cu::common::device::sameSign(&sumYs, &sumYsm)) {
                if ((sumYsm + sumYs < 0 && sumYs > 0) || (sumYs + sumYsm > 0 && sumYs < 0)) {
                    out[t] = 255;
                }
            }
        }
    } else
        out[t] = 0;
}

__global__ void cu::improc::global::findDistanceFast(unsigned char* edge, unsigned char* out, const int w, const int h) {

    extern __shared__ unsigned char s[];

    const int x = blockIdx.x * blockDim.y + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int iX = threadIdx.x + 20;
    const int iY = threadIdx.y + 20;

    const short idx = iX + 64 * (iY);

    s[idx] = edge[y * w + x];

    if (iX < 40) {
        if (x < 20) {
            s[iX - 20 + 64 * (iY)] = 0;
            if (iY < 40)
                s[iX - 20 + 64 * (iY - 20)] = 0;
        } else {
            s[iX - 20 + 64 * (iY)] = edge[y * w + x - 20];
            if (y < 20)
                s[iX - 20 + 64 * (iY - 20)] = 0;
            else
                s[iX - 20 + 64 * (iY - 20)] = edge[(y - 20) * w + (x - 20)];
        }
    }
    if (iX >= blockDim.x) {
        if (x >= w - 20) {
            s[iX + 20 + 64 * (iY)] = 0;
            if (iY >= blockDim.y)
                s[iX + 20 + 64 * (iY + 20)] = 0;
        } else {
            s[iX + 20 + 64 * (iY)] = edge[y * w + (x + 20)];
            if (y >= h - 20)
                s[iX + 20 + 64 * (iY + 20)] = 0;
            else
                s[iX + 20 + 64 * (iY + 20)] = edge[(y + 20) * w + x + 20];
        }
    }
    if (iY < 40) {
        if (y < 20) {
            s[iX + 64 * (iY - 20)] = 0;
            if (iX >= blockDim.x)
                s[iX + 20 + 64 * (iY - 20)] = 0;
        } else {
            s[iX + 64 * (iY - 20)] = edge[(y - 20) * w + x];
            if (x >= w - 20)
                s[iX + 20 + 64 * (iY - 20)] = 0;
            else
                s[iX + 20 + 64 * (iY - 20)] = edge[(y - 20) * w + x + 20];
        }
    }
    if (iY >= blockDim.y) {
        if (y >= h - 20) {
            s[iX + 64 * (iY + 20)] = 0;
            if (iX < 40)
                s[iX - 20 + 64 * (iY + 20)] = 0;
        } else {
            s[iX + 64 * (iY + 20)] = edge[(y + 20) * w + x];
            if (x < 20)
                s[iX - 20 + 64 * (iY + 20)] = 0;
            else
                s[iX - 20 + 64 * (iY + 20)] = edge[(y + 20) * w + (x - 20)];
        }
    }
    __syncthreads();
    short idx1 = (iX - 12) + 64 * (iY - 12);
    short idx2 = (iX + 12) + 64 * (iY - 12);
    short idx3 = (iX - 12) + 64 * (iY + 12);
    short idx4 = (iX + 12) + 64 * (iY + 12);
    unsigned char i1 = 0;
    unsigned char i2 = 0;
    unsigned char i3 = 0;
    unsigned char i4 = 0;

    short r = 1;
    do {
        if (!s[idx1]) {
            if (s[idx1 + 1] || s[idx1 - 1] || s[idx1 + 64] || s[idx1 - 64]) {
                i1 = r;
            } else
                i1 = 0;
        }
        if (!s[idx2]) {
            if (s[idx2 + 1] || s[idx2 - 1] || s[idx2 + 64] || s[idx2 - 64]) {
                i2 = r;
            } else
                i2 = 0;
        }
        if (!s[idx3]) {
            if (s[idx3 + 1] || s[idx3 - 1] || s[idx3 + 64] || s[idx3 - 64]) {
                i3 = r;
            } else
                i3 = 0;
        }
        if (!s[idx4]) {
            if (s[idx4 + 1] || s[idx4 - 1] || s[idx4 + 64] || s[idx4 - 64]) {
                i4 = r;
            } else
                i4 = 0;
        }
        __syncthreads();
        if (i1)
            s[idx1] = i1;
        if (i2)
            s[idx2] = i2;
        if (i3)
            s[idx3] = i3;
        if (i4)
            s[idx4] = i4;
        __syncthreads();
    } while (r++ < 21);
    if (!s[idx]) {
        out[y * w + x] = 21;
    } else if (s[idx] == 255) {
        out[y * w + x] = 0;
    } else
        out[y * w + x] = s[idx];
}
