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

View::View(cam::Camera *camera, float distMax, float width) {
    this->camera = camera;
    this->size = camera->size;
    this->left_cpu = camera->left.data;
    this->right_cpu = camera->right.data;
    this->angles = glm::vec3(0, 0, 0);
    this->origin = glm::vec3(0, 0, 0);
    cu::init(&(this->data_gpu), size.getHeight(), size.getWidth(),
           size.getChannels());
    std::cout << "View created for W=" << size.getWidth() << " H=" << size.getHeight() << " C=" << size.getChannels() << std::endl;

    this->width = width;
    this->distMax = distMax;
    this->areaDepth = (int)(this->distMax*100)/2;
    this->areaWidth = (int)(this->width*100)/2;
    this->depth = cv::Mat(this->areaDepth, this->areaWidth, CV_8UC1);
    this->depthTH = cv::Mat(this->areaDepth, this->areaWidth, CV_8UC1);
    this->disparity  = (float*)malloc(this->size.getWidth()*this->size.getHeight()*sizeof(float));
    std::cout << "Depth created: Width=" << this->width <<"m; Depth="<< this->distMax << "m"<< std::endl;

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

void View::depthSegmentation(){
    cu::grab(&data_gpu, this->disparity);
    cv::Mat disp = cv::Mat(480, 640, CV_32FC1, this->disparity);
    cv::medianBlur(disp, disp, 3);
    memset((void*)(this->depth.data), 0, this->areaDepth * this->areaWidth * sizeof(unsigned char));
    double u0 = camera->paramsRect.lU.at<double>(0, 2);
    unsigned char* d = this->depth.data;
    for (unsigned int u = 0; u < size.getWidth(); ++u) {
        for (unsigned int v = 0; v < this->size.getHeight(); ++v) {
            double value = this->disparity[v*size.getWidth()+u];
            if(value > 1) {
                double Z = (camera->paramsDepth.focalLength*camera->paramsDepth.baseline)/(value);
                if(Z < 5.0f) {
                    double X = (float) ((u - u0) * Z / camera->paramsDepth.focalLength);

                    int depth2d_y = (((this->distMax)-Z)*100)/2;
                    int depth2d_x = ((X+(this->width/2))*100)/2;
                    if(depth2d_y < this->areaDepth && depth2d_y >= 0 && depth2d_x < this->areaWidth && depth2d_x >=0){
                        int idx = (int)(this->width*100)/2*depth2d_y+depth2d_x;
                        if((int)d[idx] + (64-value)/8+1 < 255){
                            d[idx] += (64-value)/16+1;
                        }else{
                            d[idx] = 255;
                        }
                    }
                }
            }
        }
    }
    Mat element = getStructuringElement( MORPH_ELLIPSE,
                                         Size( 2*1 + 1, 2*1+1 ),
                                         Point( 1, 1 ) );

    cv::medianBlur(this->depth, this->depth, 3);
    cv::dilate(this->depth, this->depth, element);
    cv::erode(this->depth, this->depth, element);
    cv::erode(this->depth, this->depth, element);

    depth.convertTo(depthPrev, CV_8UC1);

    cv::threshold(this->depth, this->depthTH, 150, 255, THRESH_BINARY);
    cv::threshold(this->depth, this->depth, 50, 255, THRESH_BINARY);

    obstacles.clear();
    cv::Rect rect;
    for (unsigned int x = 0; x < this->areaWidth; ++x) {
        for (unsigned int y = 0; y < this->areaDepth; ++y) {
            if(depthTH.at<uchar>(y, x) && depth.at<uchar>(y, x) == 255){
                floodFill(this->depth, cv::Point(x, y), Scalar(100), &rect);
                if(rect.area() > 100)
                    obstacles.push_back(Obstacle(rect));
            }
        }
    }

    std::cout << "No. of objects=" << obstacles.size() << std::endl;
    depth.convertTo(depthRectPrev, CV_8UC1);
    for (auto &o : obstacles) // access by reference to avoid copying
    {
        //std::cout << "Rect at: x=" << o.x <<" y=" << o.z_dist << " w=" << o.width << std::endl;
        cv::rectangle(depthPrev, o.bBox, cv::Scalar(255, 255, 255));
    }


}

vector<Obstacle>* View::getObstacles(){ return &obstacles; }