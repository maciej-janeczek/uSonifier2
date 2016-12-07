#include "common.cuh"


namespace cu {
    namespace stereo{
        namespace macher {
            namespace global {
                __global__ void match3x3(unsigned short* out, const int w, const int h);
                __global__ void match8x8(unsigned short* out, const int w, const int h);
                __global__ void matcher_AEMBMv1(unsigned char* edges, unsigned short* block2, unsigned int* block32, float* out, const int w, const int h);
                __global__ void match8to32(unsigned short* out8, unsigned int* out32, const int w, const int h);
            }
            namespace device {

            };
        }
    }
}

namespace macher = cu::stereo::macher;

__global__ void macher::global::match8x8(unsigned short* out, const int w, const int h){

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    const int imgShift = 128*(blockIdx.y*w/8+(4*blockIdx.x)+threadIdx.y);

    __shared__ unsigned short results[8][32];
    __shared__ unsigned int right[8][32+64];
    __shared__ unsigned short costs[4][64];
    uchar4 lchar4;
    uchar4 rchar4;
    unsigned char lchar[4];
    unsigned char rchar[4];
    unsigned int leftPix;
    unsigned int cost;
    unsigned short costShort;
    unsigned int zero = 0;

    //if(idx+maxDisp > w-1) return;

    lchar4 = tex2D((texLeft), idx, idy);
    lchar[0] = lchar4.x;
    lchar[1] = lchar4.y;
    lchar[2] = lchar4.z;
    lchar[3] = 0;
    leftPix = *((unsigned int*)lchar);

    rchar4 = tex2D((texRight), idx, idy);
    rchar[0] = rchar4.x;
    rchar[1] = rchar4.y;
    rchar[2] = rchar4.z;
    rchar[3] = 0;
    right[threadIdx.y][threadIdx.x] = *((unsigned int *) rchar);

    rchar4 = tex2D((texRight), idx+32, idy);
    rchar[0] = rchar4.x;
    rchar[1] = rchar4.y;
    rchar[2] = rchar4.z;
    rchar[3] = 0;
    right[threadIdx.y][threadIdx.x+32] = *((unsigned int *) rchar);

    rchar4 = tex2D((texRight), idx+64, idy);
    rchar[0] = rchar4.x;
    rchar[1] = rchar4.y;
    rchar[2] = rchar4.z;
    rchar[3] = 0;
    right[threadIdx.y][threadIdx.x+64] = *((unsigned int *) rchar);

    /*rchar4 = tex2D((texRight), idx+96, idy);
    rchar[0] = rchar4.x;
    rchar[1] = rchar4.y;
    rchar[2] = rchar4.z;
    rchar[3] = 0;
    right[threadIdx.y][threadIdx.x+96] = *((unsigned int *) rchar);

    rchar4 = tex2D((texRight), idx+128, idy);
    rchar[0] = rchar4.x;
    rchar[1] = rchar4.y;
    rchar[2] = rchar4.z;
    rchar[3] = 0;
    right[threadIdx.y][threadIdx.x+128] = *((unsigned int *) rchar);*/
    __syncthreads();

    //Calculate SADs

#pragma unroll
    for(int d = 0; d < 64; d+=1) {
        asm("vabsdiff4.u32.u32.u32.add" " %0, %1, %2, %3;" : "=r"(cost) : "r"(leftPix), "r"(right[threadIdx.y][threadIdx.x+d]), "r"(zero));
        results[threadIdx.y][threadIdx.x] = cost;

        __syncthreads();
        if(threadIdx.y == 0){
            costShort = results[threadIdx.y+1][threadIdx.x];
            costShort += results[threadIdx.y+2][threadIdx.x];
            costShort += results[threadIdx.y+3][threadIdx.x];
            costShort += results[threadIdx.y+4][threadIdx.x];
            costShort += results[threadIdx.y+5][threadIdx.x];
            costShort += results[threadIdx.y+6][threadIdx.x];
            costShort += results[threadIdx.y+7][threadIdx.x];
            results[threadIdx.y  ][threadIdx.x] += costShort;
        }
        __syncthreads();
        if(threadIdx.x < 4 && threadIdx.y == 0){
            costShort =  results[0][8*threadIdx.x];
            costShort += results[0][8*threadIdx.x+1];
            costShort += results[0][8*threadIdx.x+2];
            costShort += results[0][8*threadIdx.x+3];
            costShort += results[0][8*threadIdx.x+4];
            costShort += results[0][8*threadIdx.x+5];
            costShort += results[0][8*threadIdx.x+6];
            costShort += results[0][8*threadIdx.x+7];
            costs[threadIdx.x][d] = costShort;
        }
    }
    __syncthreads();
    if(threadIdx.y < 4){
        out[imgShift+threadIdx.x] = costs[threadIdx.y][threadIdx.x];
        out[imgShift+threadIdx.x+32] = costs[threadIdx.y][threadIdx.x+32];
        //out[imgShift+threadIdx.x+64] = costs[threadIdx.y][threadIdx.x+64];
        //out[imgShift+threadIdx.x+96] = costs[threadIdx.y][threadIdx.x+96];
    }
}

__global__ void macher::global::match3x3(unsigned short* out, const int w, const int h) {

    const int idx = blockIdx.x * 64 + threadIdx.x;
    const int idy = blockIdx.y * 8 + threadIdx.y;

    __shared__ unsigned short results[10][66];
    __shared__ unsigned int right[10][66 + 64];

    uchar4 lchar4;
    uchar4 rchar4;
    unsigned char lchar[4];
    unsigned char rchar[4];
    unsigned int leftPix;
    unsigned int cost;
    unsigned short costShort;
    unsigned int zero = 0;

    lchar4 = tex2D((texLeft), idx - 1, idy - 1);
    lchar[0] = lchar4.x;
    lchar[1] = lchar4.y;
    lchar[2] = lchar4.z;
    lchar[3] = 0;
    leftPix = *((unsigned int *) lchar);

    rchar4 = tex2D((texRight), idx - 1, idy - 1);
    rchar[0] = rchar4.x;
    rchar[1] = rchar4.y;
    rchar[2] = rchar4.z;
    rchar[3] = 0;
    right[threadIdx.y][threadIdx.x] = *((unsigned int *) rchar);

    rchar4 = tex2D((texRight), idx + 65, idy - 1);
    rchar[0] = rchar4.x;
    rchar[1] = rchar4.y;
    rchar[2] = rchar4.z;
    rchar[3] = 0;
    right[threadIdx.y][threadIdx.x + 66] = *((unsigned int *) rchar);

    /*if (threadIdx.x < 62) {
        rchar4 = tex2D((texRight), idx + 131, idy - 1);
        rchar[0] = rchar4.x;
        rchar[1] = rchar4.y;
        rchar[2] = rchar4.z;
        rchar[3] = 0;
        right[threadIdx.y][threadIdx.x + 132] = *((unsigned int *) rchar);
    }*/

    if(idx < 1 || idx > w - 63 || idy < 1 || idy > h - 1){
        return;
    }
    //Calculate SADs
#pragma unroll
    for(int d = 0; d < 64; d+=1) {
        __syncthreads();
        asm("vabsdiff4.u32.u32.u32.add" " %0, %1, %2, %3;" : "=r"(cost) : "r"(leftPix), "r"(right[threadIdx.y][threadIdx.x+d]), "r"(zero));
        results[threadIdx.y][threadIdx.x] = cost;

        __syncthreads();

        if(threadIdx.x < 64 && threadIdx.y < 8 ){
            costShort  = results[threadIdx.y  ][threadIdx.x  ];
            costShort += results[threadIdx.y+1][threadIdx.x  ];
            costShort += results[threadIdx.y+2][threadIdx.x  ];
            costShort += results[threadIdx.y+0][threadIdx.x+1];
            costShort += results[threadIdx.y+1][threadIdx.x+1];
            costShort += results[threadIdx.y+2][threadIdx.x+1];
            costShort += results[threadIdx.y+0][threadIdx.x+2];
            costShort += results[threadIdx.y+1][threadIdx.x+2];
            costShort += results[threadIdx.y+2][threadIdx.x+2];
            out[d*(w*h)+idy*w+idx] = costShort;
        }
    }
}

__global__ void macher::global::match8to32(unsigned short* in8, unsigned int* out32, const int w, const int h) {

    const int idx = 2*blockIdx.x;
    const int idy = 2*blockIdx.y;
    unsigned int cost;

    __shared__ unsigned int s_in8[9][64];
    __shared__ unsigned int s_out32[4][64];

    if(idx < 1 || idx > (w/8)-1 || idy < 1 || idy > (h/8)-1) return;
#pragma unroll
    for(int x = 0; x < 3; x++){
#pragma unroll
        for(int y = 0; y < 3; y++){
            cost  = in8[128*((idy-2+2*y)*w/8+(idx-2+2*x))+threadIdx.x];
            cost += in8[128*((idy-1+2*y)*w/8+(idx-2+2*x))+threadIdx.x];
            cost += in8[128*((idy-2+2*y)*w/8+(idx-1+2*x))+threadIdx.x];
            s_in8[y*3+x][threadIdx.x] = cost + in8[128*((idy-1)*w/8+(idx-1))+threadIdx.x];
        }
    }
    __syncthreads();
#pragma unroll
    for(int x = 0; x < 2; x++){
#pragma unroll
        for(int y = 0; y < 2; y++){
            cost  = s_in8[(y  )*3+x  ][threadIdx.x];
            cost += s_in8[(y+1)*3+x  ][threadIdx.x];
            cost += s_in8[(y  )*3+x+1][threadIdx.x];
            s_out32[y*2+x][threadIdx.x] = cost + s_in8[(y+1)*3+x+1][threadIdx.x];
            out32[128*((idy+y)*(w/8)+(idx+x))+threadIdx.x] = s_out32[y*2+x][threadIdx.x];
        }
    }
}

__global__ void macher::global::matcher_AEMBMv1(unsigned char* edges, unsigned short* block2, unsigned int* block32, float* out, const int w, const int h){
    ///Main version
    ///
    __shared__ unsigned int s_block32[4][128];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;


    const int idx32_0 = 128*((2*blockIdx.y  )*(w/8)+2*blockIdx.x+threadIdx.y/8)+16*(threadIdx.y%8)+threadIdx.x;
    const int idx32_1 = 128*((2*blockIdx.y+1)*(w/8)+2*blockIdx.x+threadIdx.y/8)+16*(threadIdx.y%8)+threadIdx.x;
    const float xx32 = threadIdx.x/16.0f;
    const float yy32 = threadIdx.y/16.0f;
    const float x32 = 1.0f-xx32;
    const float y32 = 1.0f-yy32;
    unsigned int cost32;
    unsigned int cost;
    unsigned int bestCost = 99999999;
    unsigned int disp = 0;
    unsigned int e32 = 15+edges[idy*w+idx];
    unsigned int e2 =  32*(45 - e32);

    s_block32[threadIdx.y/8][16*(threadIdx.y%8)+threadIdx.x] =   block32[idx32_0];
    s_block32[2+threadIdx.y/8][16*(threadIdx.y%8)+threadIdx.x] = block32[idx32_1];

    if(idx < 17 || idx > w-64-1 || idy < 17 || idy >= (h-17)){
        out[idy * w + idx] = 0.0f;
        return;
    }
    __syncthreads();

#pragma unroll
    for(int d = 0; d < 64; d+=1) {
        cost32  = x32  * y32  * s_block32[0][d];
        cost32 += x32  * yy32 * s_block32[2][d];
        cost32 += xx32 * y32  * s_block32[1][d];
        cost32 += xx32 * yy32 * s_block32[3][d];

        cost = e2 * block2[d*(w*h)+idy*w+idx] + e32*cost32;

        if(cost < bestCost){
            bestCost = cost;
            disp = d;
        }
    }

    if(disp == 0 || disp == 63){
        out[idy * w + idx] = 0.0f;
        return;
    }

    cost32  = x32  * y32  * s_block32[0][disp-1];
    cost32 += x32  * yy32 * s_block32[2][disp-1];
    cost32 += xx32 * y32  * s_block32[1][disp-1];
    cost32 += xx32 * yy32 * s_block32[3][disp-1];

    unsigned int prevCost = e2 * block2[(disp-1)*(w*h)+idy*w+idx] + e32*cost32;

    cost32  = x32  * y32  * s_block32[0][disp+1];
    cost32 += x32  * yy32 * s_block32[2][disp+1];
    cost32 += xx32 * y32  * s_block32[1][disp+1];
    cost32 += xx32 * yy32 * s_block32[3][disp+1];

    unsigned int nextCost = e2 * block2[(disp+1)*(w*h)+idy*w+idx] + e32*cost32;

    if(   bestCost < 2000000  &&
            (0.05f < (float)(prevCost-bestCost)/(bestCost) || 0.05f < (float)(nextCost-bestCost)/(bestCost))
                    /*(bestCost < prevCost*0.95f)  (bestCost < nextCost*0.95f)
        /*(bestCost < bestPrevCost*0.5f && common::device::absi((int)prevDisp - disp) != 1))*/){
        float inter = (float)(((int)prevCost)-((int)nextCost))/(2*(((int)prevCost)-2*((int)bestCost)+((int)nextCost)));
        out[idy * w + idx] = ((float)disp+inter)/255.0f;
    }else{
        out[idy * w + idx] = 0.0f;
    }

}
