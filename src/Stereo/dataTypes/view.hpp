#pragma one

#include "../../Camera/Camera.h"
#include "cuda/cuTypes.h"
#include "size2d.hpp"
#include <glm/glm.hpp>

class View {
public:
  View(unsigned int newWidth, unsigned int newHeight, unsigned int newChannels,
       unsigned char *left_cpu, unsigned char *right_cpu);
  View(stereo::Size2d size, unsigned char *left_cpu, unsigned char *right_cpu);
  View(cam::Camera *camera);
  ~View();
  int update_Old(glm::vec3 angles);
  int updateFromCam();
  int grab(float *output);

  stereo::Size2d size = stereo::Size2d(0, 0, 0);
  cu::data_gpu data_gpu;
  cam::Camera *camera;
  unsigned char *left_cpu;
  unsigned char *right_cpu;
  glm::vec3 angles;
  glm::vec3 origin;
};
