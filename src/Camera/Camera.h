//
// Created by Maciej Janeczek on 04.12.16.
//

#ifndef USONIFIER_CAMERA_H
#define USONIFIER_CAMERA_H


#include <opencv2/opencv.hpp>
#include "DUO/PID/cameractrl.h"
#include "../Stereo/dataTypes/size2d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace cam{
    struct paramsStereo{
        float focalLength; ///[mm/mm] = (focal length) / (pixel size)
        float baseline; ///[mm]
    };

    struct paramsRectif{
        cv::Mat lU, rU, lR, rR, R, T;
        cv::Mat mapL1, mapL2, mapR1, mapR2;
    };

    enum type{
        IMAGE,
        DUO
    };

    enum color{
        GRAY,
        RGB
    };

    enum duoParamsSet{
        CIRCLES1, CIRCLES2, CHESSBOARD1, CHESSBOARD2
    };

    class Camera {
    public:
        cv::Mat left;
        cv::Mat right;
        stereo::Size2d size = stereo::Size2d(640, 480, 1);
        int fps;
        paramsStereo paramsDepth;
        paramsRectif paramsRect;

    private:
        int autoRectif;
        int autoBrightness;
        int autoHistEqualize;
        int denoiseImages;
        int opened;
        CameraCtrl *cameraCtrl;
        cam::type type;
        cam::color color;

    public:
        Camera(cam::type type, stereo::Size2d size,
               cam::color color, int fps, int rectify,
               int brightness, int histEqualize, int denoiseImages);
        //Camera(cam::type type, cam::color color, cam::paramsStereo paramsS, cam::paramsRectif paramsR,
        //      int enableAutoRectify = 1, int enableAutoPID = 1);

        ~Camera();

        void initRectif();
        void update();

    private:

        void initRectificationParams(cam::duoParamsSet set);

    };
}

#endif //USONIFIER_CAMERA_H
