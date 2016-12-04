#pragma once
#include <cuda_runtime.h>

texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texLeft;
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texRight;

texture<uchar4, cudaTextureType2D, cudaReadModeElementType> improcTex1;


namespace cu {
    namespace common {
        namespace device {
            __device__ bool sameSign(int *x, int *y);
            __device__ unsigned int absi(int a);

            __device__ unsigned char pix_UC4toUC1_Y(uchar4 *in);

            __device__ void
            loadChunkToShared_UC1_16x16_b3(unsigned char *img, unsigned char s[22][22], const unsigned int cols,
                                           const unsigned int rows,
                                           const int x, const int y, const int iX, const int iY);

            __device__ unsigned int __usad4(unsigned int A, unsigned int B, unsigned int C = 0);
        }
        namespace global {
            __global__ void uchar4texToUchar1(unsigned char *out, unsigned int w, unsigned int h);

            __global__ void
            uchar1toFloat(unsigned char *in, float *out, unsigned int w, unsigned int h, float scale = 1.0f,
                          float shift = 0.0f);
        }
    }
}


namespace common = cu::common;

__device__ unsigned int common::device::absi(int a){ return max(-a, a); }

__device__ bool common::device::sameSign(int* x, int* y) { return ((*x > 0) ^ (*y < 0)); }

__device__ unsigned char common::device::pix_UC4toUC1_Y(uchar4* in) { return (unsigned char)(.2989f * in->x + .587f * in->y + .114f * in->z); }

__global__ void common::global::uchar4texToUchar1(unsigned char* out, unsigned int w, unsigned int h) {
    /// Converts uchar4 texture to uchar1 array

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    float iX = (float)idx;
    float iY = (float)idy;

    uchar4 pixelIn = tex2D((improcTex1), iX, iY);
    out[idy * w + idx] = (device::pix_UC4toUC1_Y(&pixelIn));
}

__global__ void common::global::uchar1toFloat(unsigned char* in, float* out, unsigned int w, unsigned int h, float scale, float shift) {
    /// Converts uchar1 array to float array
    /// Size is deccribed by w and h
    /// out = in*scale+shift

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    out[idy * w + idx] = (float)in[idy * w + idx] * scale + shift;
}

__device__ void common::device::loadChunkToShared_UC1_16x16_b3(
    unsigned char* img, unsigned char s[22][22], const unsigned int cols, const unsigned int rows, const int x, const int y, const int iX, const int iY) {
    /// Loads 16x16 window with border 3 from 1D array (2D)image to shared memory
    /// img[] is input image, s[][] is 2D shared memory array
    /// iX and iY are threadIdx+border values
    /// x and y are image pixel related to threadIdx
    /// use __syncthreads(); after

    // --- Fill shared memory
    s[iX][iY] = img[y * cols + x];

    if (iX < 6) {
        if (x < 3) {
            s[iX - 3][iY] = 0;
            if (iY < 6)
                s[iX - 3][iY - 3] = 0;
        } else {
            s[iX - 3][iY] = img[y * cols + x - 3];
            if (y < 3)
                s[iX - 3][iY - 3] = 0;
            else
                s[iX - 3][iY - 3] = img[(y - 3) * cols + (x - 3)];
        }
    }
    if (iX >= blockDim.x) {
        if (x >= cols - 3) {
            s[iX + 3][iY] = 0;
            if (iY >= blockDim.y)
                s[iX + 3][iY + 3] = 0;
        } else {
            s[iX + 3][iY] = img[y * cols + (x + 3)];
            if (y >= rows - 3)
                s[iX + 3][iY + 3] = 0;
            else
                s[iX + 3][iY + 3] = img[(y + 3) * cols + x + 3];
        }
    }
    if (iY < 6) {
        if (y < 3) {
            s[iX][iY - 3] = 0;
            if (iX >= blockDim.x)
                s[iX + 3][iY - 3] = 0;
        } else {
            s[iX][iY - 3] = img[(y - 3) * cols + x];
            if (x >= cols - 3)
                s[iX + 3][iY - 3] = 0;
            else
                s[iX + 3][iY - 3] = img[(y - 3) * cols + x + 3];
        }
    }
    if (iY >= blockDim.y) {
        if (y >= rows - 3) {
            s[iX][iY + 3] = 0;
            if (iX < 6)
                s[iX - 3][iY + 3] = 0;
        } else {
            s[iX][iY + 3] = img[(y + 3) * cols + x];
            if (x < 3)
                s[iX - 3][iY + 3] = 0;
            else
                s[iX - 3][iY + 3] = img[(y + 3) * cols + (x - 3)];
        }
    }
}

__device__ unsigned int common::device::__usad4(unsigned int A, unsigned int B, unsigned int C) {
    unsigned int result;
#if (__CUDA_ARCH__ >= 300) // Kepler (SM 3.x) supports a 4 vector SAD SIMD
    asm("vabsdiff4.u32.u32.u32.add"
        " %0, %1, %2, %3;"
        : "=r"(result)
        : "r"(A), "r"(B), "r"(C));
#else // SM 2.0            // Fermi  (SM 2.x) supports only 1 SAD SIMD, so there are 4 instructions
    asm("vabsdiff.u32.u32.u32.add"
        " %0, %1.b0, %2.b0, %3;"
        : "=r"(result)
        : "r"(A), "r"(B), "r"(C));
    asm("vabsdiff.u32.u32.u32.add"
        " %0, %1.b1, %2.b1, %3;"
        : "=r"(result)
        : "r"(A), "r"(B), "r"(result));
    asm("vabsdiff.u32.u32.u32.add"
        " %0, %1.b2, %2.b2, %3;"
        : "=r"(result)
        : "r"(A), "r"(B), "r"(result));
    asm("vabsdiff.u32.u32.u32.add"
        " %0, %1.b3, %2.b3, %3;"
        : "=r"(result)
        : "r"(A), "r"(B), "r"(result));
#endif
    return result;
}
