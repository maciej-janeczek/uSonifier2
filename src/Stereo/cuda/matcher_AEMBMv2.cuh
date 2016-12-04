#pragma once
#include "common.cuh"


namespace cu {
    namespace stereo{
        namespace macher {
            namespace global {
                __global__ void matcher_AEMBMv2(unsigned char* edges, unsigned char* out, const int w, const int h, const int minDisp, const int maxDisp);
            }
            namespace device {

            };
        }
    }
}

__global__ void cu::stereo::macher::global::matcher_AEMBMv2(unsigned char* edges, unsigned char* out, const int w, const int h, const int minDisp, const int maxDisp){
    ///Alternative version of the Matcher, to be tested
    ///
    /*const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    const int iX = threadIdx.x + 16;
    const int iY = threadIdx.y + 16;

    __shared__ unsigned char e_s[8][32];
    __shared__ unsigned int results[40][64];
    unsigned int left[5][2];
    unsigned int rightPix;
    unsigned int leftPix;
    unsigned int cost = 0;
    unsigned int bestCost = 99999999;
    unsigned int bestDisp = 0;
    // Load edges to shared (will be used dispMax-dispMin times)
    e_s[threadIdx.y][threadIdx.x] = edges[idy*w+idx];

    // Load left image to shared (will be used dispMax-dispMin times
    #pragma unroll
    for(int x = 0; x < 2; x++){
        #pragma unroll
        for(int y = 0; y < 5; y++){
            left[y][x] = tex2D((texLeft), idx-32+32*x, idy-16+8*y);
        }
    }
    __syncthreads();
    #pragma  unroll
    for(int d = 0; d < 128; d+=2){
        #pragma unroll
        for(int x = 0; x < 2; x++){
            #pragma unroll
            for(int y = 0; y < 5; y++){
                leftPix = left[y][x];
                rightPix = tex2D((texRight), idx-32+32*x+d, idy-16+8*y);
                asm("vabsdiff4.u32.u32.u32.add" " %0, %1, %2, %3;" : "=r"(cost) : "r"(leftPix), "r"(rightPix), "r"(0));
                results[y*8+threadIdx.y][x*32+threadIdx.x] = cost;
            }
        }
        __syncthreads();
        //#pragma unroll
        for (int y=0; y<5; y++)
        {
            cost = 0;
            //#pragma unroll
            for (int i=-5; i<=5 ; i++)
            {
                cost += results[iY-16+8*y][iX+i];
            }
            __syncthreads();
            results[iY-16+8*y][iX] = cost;
            __syncthreads();
        }
        // sum cost vertically
        cost = 0;
        //#pragma unroll
        for (int i=-5; i<=5 ; i++)
        {
            cost += results[iY+i][iX];
        }

        if(cost < bestCost){
            bestCost = cost;
            bestDisp = d;
        }//
    }
    out[idy*w+idx] = bestDisp;
*/
}