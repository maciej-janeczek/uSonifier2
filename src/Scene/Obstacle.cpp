#include "Obstacle.h"


Obstacle::Obstacle(cv::Rect bBox){
    this->bBox = bBox;
    this->width = bBox.width/50.0f;
    this->type = obst::TYPE::obstacle;
    this->x = ((bBox.x+bBox.width/2)-100)/50.0f;
    this->z_dist = (250-(bBox.y+bBox.height))/50.0f;
    this->dist = sqrt(this->x*this->x+this->z_dist*this->z_dist);
}

Obstacle::~Obstacle(){

}

float Obstacle::isOnPath(float width) {
    ///Returns information whether the obstacle is on walk path of specific width and returns its distance to path
    float dist = abs(this->x) - this->width / 2.0f;
    if (dist > 0)
        return dist;
    else
        return 0.0f;
}

