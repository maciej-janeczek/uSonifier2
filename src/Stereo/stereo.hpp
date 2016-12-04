#pragma once

#include "cuda/cuStereo.h"
#include "dataTypes/view.hpp"


namespace stereo {
    class Macher {
    private:
        View* view;
        cu::stereo::stereoParams params;
        cu::stereo::tempArr temps;

    public:
        Macher(View* view, cu::stereo::stereoParams params);
        Macher(View* view, unsigned int minDisp, unsigned int maxDisp, unsigned int edgeMultiplier, unsigned int threshold);
        ~Macher();
        void perform_AEMBM();
        void perform_NEW();
    };
}

