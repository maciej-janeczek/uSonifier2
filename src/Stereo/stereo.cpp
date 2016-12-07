#include "stereo.hpp"

stereo::Macher::Macher(View* view, cu::stereo::stereoParams params) {
    this->view = view;
    this->params.edgeMultiplier = params.edgeMultiplier;
    this->params.maxDisp = params.maxDisp;
    this->params.minDisp = params.minDisp;
    this->params.threshold = params.threshold;
    this->temps.sizes[0] = 3;
    this->temps.sizes[1] = 3;
    this->temps.sizes[2] = 3;

    cu::stereo::allocTemps(this->temps, view->size.getWidth(), view->size.getHeight(), view->size.getChannels());
}

stereo::Macher::Macher(View* view, unsigned int minDisp, unsigned int maxDisp, unsigned int edgeMultiplier, unsigned int threshold) {
    this->view = view;
    this->params.edgeMultiplier = edgeMultiplier;
    this->params.maxDisp = maxDisp;
    this->params.minDisp = minDisp;
    this->params.threshold = threshold;
    this->temps.sizes[0] = 3;
    this->temps.sizes[1] = 3;
    this->temps.sizes[2] = 3;
    std::cout << "Macher created for W=" << view->size.getWidth() << " H=" << view->size.getHeight() << " C=" << view->size.getChannels() << std::endl;
    cu::stereo::allocTemps(this->temps, view->size.getWidth(), view->size.getHeight(), view->size.getChannels());
}

stereo::Macher::~Macher() { cu::stereo::freeTemps(temps); }

void stereo::Macher::perform_AEMBM() {
    cu::stereo::match_AEMBM(view->data_gpu.left, view->data_gpu.right, view->data_gpu.disparity, this->temps, this->params, 640, 480, 3);
}
