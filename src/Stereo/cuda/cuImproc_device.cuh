#include <cuda_runtime.h>


namespace improc {
    namespace cu {
        namespace device {
            __device__ void sobelCellFirstY(unsigned char s[22][22], unsigned int iX, unsigned int iY, int* res);
            __device__ void sobelCellFirstX(unsigned char s[22][22], unsigned int iX, unsigned int iY, int* res);
            __device__ void sobelCellSecondY(unsigned char s[22][22], unsigned int iX, unsigned int iY, int* res);
            __device__ void sobelCellSecondX(unsigned char s[22][22], unsigned int iX, unsigned int iY, int* res);
        }
    }
}


__device__ void improc::cu::device::sobelCellFirstY(unsigned char s[22][22], unsigned int iX, unsigned int iY, int* res) {
    unsigned char inP[4], inN[4];
    inP[0] = s[iX - 1][iY + 1];
    inP[1] = s[iX][iY + 1];
    inP[2] = inP[1];
    inP[3] = s[iX + 1][iY + 1];

    inN[0] = s[iX - 1][iY - 1];
    inN[1] = s[iX][iY - 1];
    inN[2] = inN[1];
    inN[3] = s[iX + 1][iY - 1];
    unsigned int c = 0;
    asm("vsub4.s32.u32.u32.add"
        "%0,%1,%2,%3;"
        : "=r"(*res)
        : "r"(*((unsigned int*)inP)), "r"(*((unsigned int*)inN)), "r"(c));
    /*// Y first
    sumYf-=  s[iX-1][iY-1];
    sumYf-=2*s[iX  ][iY-1];
    sumYf-=  s[iX+1][iY-1];
    sumYf+=  s[iX-1][iY+1];
    sumYf+=2*s[iX  ][iY+1];
    sumYf+=  s[iX+1][iY+1];*/
}

__device__ void improc::cu::device::sobelCellFirstX(unsigned char s[22][22], unsigned int iX, unsigned int iY, int* res) {
    unsigned char inP[4], inN[4];
    unsigned int c = 0;
    inP[0] = s[iX + 1][iY - 1];
    inP[1] = s[iX + 1][iY];
    inP[2] = inP[1];
    inP[3] = s[iX + 1][iY + 1];

    inN[0] = s[iX - 1][iY - 1];
    inN[1] = s[iX - 1][iY];
    inN[2] = inN[1];
    inN[3] = s[iX - 1][iY + 1];
    asm("vsub4.s32.u32.u32.add"
        "%0,%1,%2,%3;"
        : "=r"(*res)
        : "r"(*((unsigned int*)inP)), "r"(*((unsigned int*)inN)), "r"(c));
}

__device__ void improc::cu::device::sobelCellSecondY(unsigned char s[22][22], unsigned int iX, unsigned int iY, int* res) {
    int tmp = 0;
    int c = 0;
    unsigned char inP[4], inN[4];

    // sum of coeff 1 and -2
    inP[0] = s[iX - 2][iY - 2];
    inP[1] = s[iX + 2][iY - 2];
    inP[2] = s[iX - 2][iY + 2];
    inP[3] = s[iX + 2][iY + 2];

    inN[0] = s[iX - 2][iY];
    inN[1] = inN[0];
    inN[2] = s[iX + 2][iY];
    inN[3] = inN[1];
    asm("vsub4.s32.u32.u32.add"
        "%0,%1,%2,%3;"
        : "=r"(*res)
        : "r"(*((unsigned int*)inP)), "r"(*((unsigned int*)inN)), "r"(c));

    // sum of coeff 4 and -8
    inP[0] = s[iX - 1][iY - 2];
    inP[1] = s[iX + 1][iY - 2];
    inP[2] = s[iX - 1][iY + 2];
    inP[3] = s[iX + 1][iY + 2];

    inN[0] = s[iX - 1][iY];
    inN[1] = inN[0];
    inN[2] = s[iX + 1][iY];
    inN[3] = inN[2];
    asm("vsub4.s32.u32.u32.add"
        "%0,%1,%2,%3;"
        : "=r"(tmp)
        : "r"(*((unsigned int*)inP)), "r"(*((unsigned int*)inN)), "r"(*res));
    *res += tmp;

    // sum of coeff 6 and -12
    inP[0] = s[iX][iY - 2];
    inP[1] = inP[0];
    inP[2] = s[iX][iY + 2];
    inP[3] = inP[2];

    inN[0] = s[iX][iY];
    inN[1] = inN[0];
    inN[2] = inN[0];
    inN[3] = inN[0];

    asm("vsub4.s32.u32.u32.add"
        "%0,%1,%2,%3;"
        : "=r"(tmp)
        : "r"(*((unsigned int*)inP)), "r"(*((unsigned int*)inN)), "r"(c));
    *res += 3 * tmp;

    /*
    umYs+=  s[iX-2][iY-2];
    sumYs+=4*s[iX-1][iY-2];
    sumYs+=6*s[iX  ][iY-2];
    sumYs+=4*s[iX+1][iY-2];
    sumYs+=  s[iX+2][iY-2];

    sumYs-= 2*s[iX-2][iY];
    sumYs-= 8*s[iX-1][iY];
    sumYs-=12*s[iX  ][iY];
    sumYs-= 8*s[iX+1][iY];
    sumYs-= 2*s[iX+2][iY];

    sumYs+=  s[iX-2][iY+2];
    sumYs+=4*s[iX-1][iY+2];
    sumYs+=6*s[iX  ][iY+2];
    sumYs+=4*s[iX+1][iY+2];
    sumYs+=  s[iX+2][iY+2];
     */
}

__device__ void improc::cu::device::sobelCellSecondX(unsigned char s[22][22], unsigned int iX, unsigned int iY, int* res) {
    int tmp = 0;
    int c = 0;
    unsigned char inP[4], inN[4];

    // sum of coeff 1 and -2
    inP[0] = s[iX - 2][iY - 2];
    inP[1] = s[iX + 2][iY - 2];
    inP[2] = s[iX - 2][iY + 2];
    inP[3] = s[iX + 2][iY + 2];

    inN[0] = s[iX][iY - 2];
    inN[1] = inN[0];
    inN[2] = s[iX][iY + 2];
    inN[3] = inN[1];
    asm("vsub4.s32.u32.u32.add"
        "%0,%1,%2,%3;"
        : "=r"(*res)
        : "r"(*((unsigned int*)inP)), "r"(*((unsigned int*)inN)), "r"(c));

    // sum of coeff 4 and -8
    inP[0] = s[iX - 2][iY - 1];
    inP[1] = s[iX - 2][iY + 1];
    inP[2] = s[iX + 2][iY - 1];
    inP[3] = s[iX + 2][iY + 1];

    inN[0] = s[iX][iY - 1];
    inN[1] = inN[0];
    inN[2] = s[iX][iY + 1];
    inN[3] = inN[2];
    asm("vsub4.s32.u32.u32.add"
        "%0,%1,%2,%3;"
        : "=r"(tmp)
        : "r"(*((unsigned int*)inP)), "r"(*((unsigned int*)inN)), "r"(*res));
    *res += tmp;

    // sum of coeff 6 and -12
    inP[0] = s[iX - 2][iY];
    inP[1] = inP[0];
    inP[2] = s[iX + 2][iY];
    inP[3] = inP[2];

    inN[0] = s[iX][iY];
    inN[1] = inN[0];
    inN[2] = inN[0];
    inN[3] = inN[0];

    asm("vsub4.s32.u32.u32.add"
        "%0,%1,%2,%3;"
        : "=r"(tmp)
        : "r"(*((unsigned int*)inP)), "r"(*((unsigned int*)inN)), "r"(c));
    *res += 3 * tmp;

    /*
        sumXs+=  s[iX-2][iY-2];
        sumXs+=4*s[iX-2][iY-1];
        sumXs+=6*s[iX-2][iY  ];
        sumXs+=4*s[iX-2][iY+1];
        sumXs+=  s[iX-2][iY+2];

                sumXs-= 2*s[iX][iY-2];
        sumXs-= 8*s[iX][iY-1];
        sumXs-=12*s[iX][iY  ];
        sumXs-= 8*s[iX][iY+1];
        sumXs-= 2*s[iX][iY+2];

        sumXs+=  s[iX+2][iY-2];
        sumXs+=4*s[iX+2][iY-1];
        sumXs+=6*s[iX+2][iY  ];
        sumXs+=4*s[iX+2][iY+1];
        sumXs+=  s[iX+2][iY+2];
     */
}
