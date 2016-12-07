#include "view.hpp"
#include "../cuda/cuStereo.h"

View::View(unsigned int width, unsigned int height, unsigned int channels,
           unsigned char *left_cpu, unsigned char *right_cpu)
    : View(stereo::Size2d(width, height, channels), left_cpu, right_cpu) {
  camera = nullptr;
}

View::View(stereo::Size2d size, unsigned char *left_cpu,
           unsigned char *right_cpu)
    : size(size), left_cpu(left_cpu), right_cpu(right_cpu), angles(0),
      origin(0) {
  cu::init(&(this->data_gpu), size.getHeight(), size.getWidth(),
           size.getChannels());
}

View::View(cam::Camera *camera) {
    this->camera = camera;
    this->size = camera->size;
    this->left_cpu = camera->left.data;
    this->right_cpu = camera->right.data;
    this->angles = glm::vec3(0, 0, 0);
    this->origin = glm::vec3(0, 0, 0);
    cu::init(&(this->data_gpu), size.getHeight(), size.getWidth(),
           size.getChannels());
    std::cout << "View created for W=" << size.getWidth() << " H=" << size.getHeight() << " C=" << size.getChannels() << std::endl;

}

View::~View() {
  cudaFree(data_gpu.left);
  cudaFree(data_gpu.right);
}

int View::update_Old(glm::vec3 angles) {
  return cu::update(&(this->data_gpu), this->left_cpu, this->right_cpu);
}

int View::updateFromCam() {
  camera->update();
  cu::update(&(this->data_gpu), camera->left.data, camera->right.data);
}

int View::grab(float *output) { return cu::grab(&data_gpu, output); }
