#pragma once

#include "../../Camera/Camera.h"
#include "cuda/cuTypes.h"
#include "size2d.hpp"
#include <glm/glm.hpp>
#include "../../pch.h"
#include "../../Scene/Obstacle.h"

class View {
public:
    View(unsigned int newWidth, unsigned int newHeight, unsigned int newChannels,
       unsigned char *left_cpu, unsigned char *right_cpu);
    View(stereo::Size2d size, unsigned char *left_cpu, unsigned char *right_cpu);
    View(cam::Camera *camera, float distMin, float distMax, float width);
    ~View();
    int update_Old(glm::vec3 angles);
    int updateFromCam();
    int grab(float *output);
    vector<Obstacle>* getObstacles();

    stereo::Size2d size = stereo::Size2d(0, 0, 0);
    cu::data_gpu data_gpu;
    cam::Camera *camera;
    unsigned char *left_cpu;
    unsigned char *right_cpu;
    glm::vec3 angles;
    glm::vec3 origin;
    float distMax;
    float distMin;
    float width;
    int areaWidth;  /// area of the considered depth image in pixels (Width)
    int areaDepth;  /// area of the considered depth image in pixels (Depth)

    cv::Mat depthRectPrev;
    cv::Mat depthPrev;
    void depthSegmentation();


private:
    cv::Mat depth;
    cv::Mat depthTH;
    vector<Obstacle> obstacles;
    float* disparity;
};
