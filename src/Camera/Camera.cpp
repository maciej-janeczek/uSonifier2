//
// Created by Maciej Janeczek on 04.12.16.
//

#include "Camera.h"

#include "DUO/duo.h"
#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>

cam::Camera::Camera(cam::type type, stereo::Size2d size, cam::color color,
                    int fps, int rectify, int brightness, int histEqualize,
                    int denoiseImages) {

    this->autoBrightness = brightness;
    this->autoRectif = rectify;
    this->autoHistEqualize = histEqualize;
    this->denoiseImages = denoiseImages;
    this->color = color;
    this->type = type;
    this->fps = fps;
    this->opened = 0;

    switch (this->type) {
    case cam::type::DUO:
        this->size = stereo::Size2d(640, 480, 4);
        this->paramsDepth.baseline = 0.03;
        this->paramsDepth.focalLength = 333.33;
        /// Open DUO camera and start capturing
        //printf("DUOLib Version:       v%s\n", GetLibVersion());
         if(!OpenDUOCamera(this->size.getWidth(), this->size.getHeight(),
         this->fps))
        {
            printf("Could not open DUO camera\n");
        }else{
        opened = 1;
        }

        if (color == cam::color::GRAY) {
            left = cv::Mat(size.getHeight(), size.getWidth(), CV_8UC4);
            right = cv::Mat(size.getHeight(), size.getWidth(), CV_8UC4);
        } else {
            printf("DUO camera can only operate with Grayscale images, set "
                 "color::GRAY\n");
        }

        /// Initialize PID
        if (this->autoBrightness) {
            cameraCtrl = new CameraCtrl(&left, SetExposure, SetGain, SetLed, false);
            SetGain(100);
            SetExposure(100);
            SetLed(0);
            printf("Audo brightness adjustment initialised\n");
        }

        /// Initialize Rectifocation
        if (this->autoRectif) {

            this->initRectificationParams(duoParamsSet::CIRCLES2);
            cv::Mat Q, R1, R2, P1, P2;
            cv::stereoRectify(paramsRect.lU, paramsRect.lR, paramsRect.rU,
                             paramsRect.rR, Size(640, 480), paramsRect.R,
                            paramsRect.T, R1, R2, P1, P2, Q);
            cv::fisheye::initUndistortRectifyMap(
              paramsRect.lU, paramsRect.lR, R1, P1,
              Size(size.getWidth(), size.getHeight()), CV_16SC2, paramsRect.mapL1,
              paramsRect.mapL2);
            cv::fisheye::initUndistortRectifyMap(
              paramsRect.rU, paramsRect.rR, R2, P2,
              Size(size.getWidth(), size.getHeight()), CV_16SC2, paramsRect.mapR1,
              paramsRect.mapR2);
            printf("Rectification initialised\n");
        }
        break;

    case cam::type::IMAGE:

        break;
  }
}

cam::Camera::~Camera() { delete (cameraCtrl); }

void cam::Camera::update() {
    /// Capture DUO frame
    PDUOFrame pFrameData = NULL;
    if (opened) {
        PDUOFrame pFrameData = GetDUOFrame();
        if (pFrameData == NULL)
            return;
        /// update images from camera
        for(int x = 0; x < 640*480; x++){
            this->left.data[4*x] = pFrameData->leftData[x];
            this->left.data[4*x+1] = pFrameData->leftData[x];
            this->left.data[4*x+2] = pFrameData->leftData[x];


            this->right.data[4*x] = pFrameData->rightData[x];
            this->right.data[4*x+1] = pFrameData->rightData[x];
            this->right.data[4*x+2] = pFrameData->rightData[x];
        }

        //cudaMemcpy(this->left.data, pFrameData->leftData,
        //           this->size.getSize() * sizeof(unsigned char), cudaMemcpyHostToHost);
        //cudaMemcpy(this->right.data, pFrameData->rightData,
        //           this->size.getSize() * sizeof(unsigned char), cudaMemcpyHostToHost);
    }



    /// rectify images based on initialised params
    if (autoRectif) {
        cv::remap(left, left, paramsRect.mapL1, paramsRect.mapL2, INTER_AREA);
        cv::remap(right, right, paramsRect.mapR1, paramsRect.mapR2, INTER_AREA);
    }

    /// adjust PID
    if (autoBrightness) {
        cameraCtrl->Update();
    }

    /// denoise
    if (denoiseImages) {
        medianBlur(left, left, 3);
        medianBlur(right, right, 3);
    }

    /// equalize histograms
    if (autoHistEqualize) {
        cv::equalizeHist(left, left);
        cv::equalizeHist(right, right);
    }
}

void cam::Camera::initRectificationParams(cam::duoParamsSet set) {
    paramsRect.lU = Mat::zeros(3, 3, CV_64FC1);
    paramsRect.rU = Mat::zeros(3, 3, CV_64FC1);
    paramsRect.lR = Mat::zeros(1, 4, CV_64FC1);
    paramsRect.rR = Mat::zeros(1, 4, CV_64FC1);
    paramsRect.R = Mat::zeros(3, 3, CV_64FC1);
    paramsRect.T = Mat::zeros(3, 1, CV_64FC1);
  switch (set) {
  case cam::duoParamsSet::CHESSBOARD1:
    paramsRect.lU.at<double>(0, 0) = 385.781537922;
    paramsRect.lU.at<double>(1, 1) = 386.770182219;
    paramsRect.lU.at<double>(0, 2) = 325.28735922;
    paramsRect.lU.at<double>(1, 2) = 219.111593446;
    paramsRect.lU.at<double>(2, 2) = 1.0;

    paramsRect.lR.at<double>(0, 0) = -0.115369267631;
    paramsRect.lR.at<double>(0, 1) = 0.0;

    paramsRect.rU.at<double>(0, 0) = 382.188010972;
    paramsRect.rU.at<double>(1, 1) = 382.624359906;
    paramsRect.rU.at<double>(0, 2) = 305.959504;
    paramsRect.rU.at<double>(1, 2) = 236.717857874;
    paramsRect.rU.at<double>(2, 2) = 1.0;

    paramsRect.rR.at<double>(0, 0) = -0.113198501368;
    paramsRect.rR.at<double>(0, 1) = 0.0;

    paramsRect.R.at<double>(0, 0) = 0.999978503734;
    paramsRect.R.at<double>(0, 1) = 0.00585549331363;
    paramsRect.R.at<double>(0, 2) = 0.00295046925795;
    paramsRect.R.at<double>(1, 0) = -0.00586456451925;
    paramsRect.R.at<double>(1, 1) = 0.999978074523;
    paramsRect.R.at<double>(1, 2) = 0.00307528161444;
    paramsRect.R.at<double>(2, 0) = -0.00293239727657;
    paramsRect.R.at<double>(2, 1) = -0.0030925187247;
    paramsRect.R.at<double>(2, 2) = 0.999990918646;

    paramsRect.T.at<double>(0, 0) = -28.9602486665;
    paramsRect.T.at<double>(1, 0) = 0.325331067345;
    paramsRect.T.at<double>(2, 0) = 0.659922450245;

    break;

  case cam::duoParamsSet::CHESSBOARD2:
    paramsRect.lU.at<double>(0, 0) = 385.675380919;
    paramsRect.lU.at<double>(1, 1) = 385.954329526;
    paramsRect.lU.at<double>(0, 2) = 324.171669424;
    paramsRect.lU.at<double>(1, 2) = 223.929227171;
    paramsRect.lU.at<double>(2, 2) = 1.0;

    paramsRect.lR.at<double>(0, 0) = -0.111450237218;
    paramsRect.lR.at<double>(0, 1) = -0.0177005620878;

    paramsRect.rU.at<double>(0, 0) = 382.23167021;
    paramsRect.rU.at<double>(1, 1) = 382.439655001;
    paramsRect.rU.at<double>(0, 2) = 305.931305581;
    paramsRect.rU.at<double>(1, 2) = 241.47113977;
    paramsRect.rU.at<double>(2, 2) = 1.0;

    paramsRect.rR.at<double>(0, 0) = -0.11242560225;
    paramsRect.rR.at<double>(0, 1) = -0.00762108671039;

    paramsRect.R.at<double>(0, 0) = 0.999989431215;
    paramsRect.R.at<double>(0, 1) = 0.00459342955651;
    paramsRect.R.at<double>(0, 2) = -0.000194586152118;
    paramsRect.R.at<double>(1, 0) = -0.00459281664531;
    paramsRect.R.at<double>(1, 1) = 0.999984828646;
    paramsRect.R.at<double>(1, 2) = 0.00304113696687;
    paramsRect.R.at<double>(2, 0) = 0.000208552448411;
    paramsRect.R.at<double>(2, 1) = -0.00304021112722;
    paramsRect.R.at<double>(2, 2) = 0.9999953568;

    paramsRect.T.at<double>(0, 0) = -30.2446439979;
    paramsRect.T.at<double>(1, 0) = 0.382365480057;
    paramsRect.T.at<double>(2, 0) = 1.35237960952;

    break;

  case cam::duoParamsSet::CIRCLES1:
    paramsRect.lU.at<double>(0, 0) = 385.781537922;
    paramsRect.lU.at<double>(1, 1) = 386.770182219;
    paramsRect.lU.at<double>(0, 2) = 325.28735922;
    paramsRect.lU.at<double>(1, 2) = 219.111593446;
    paramsRect.lU.at<double>(2, 2) = 1.0;

    paramsRect.lR.at<double>(0, 0) = -0.115369267631;
    paramsRect.lR.at<double>(0, 1) = 0.0;

    paramsRect.rU.at<double>(0, 0) = 382.188010972;
    paramsRect.rU.at<double>(1, 1) = 382.624359906;
    paramsRect.rU.at<double>(0, 2) = 305.959504;
    paramsRect.rU.at<double>(1, 2) = 236.717857874;
    paramsRect.rU.at<double>(2, 2) = 1.0;

    paramsRect.rR.at<double>(0, 0) = -0.113198501368;
    paramsRect.rR.at<double>(0, 1) = 0.0;

    paramsRect.R.at<double>(0, 0) = 0.999978503734;
    paramsRect.R.at<double>(0, 1) = 0.00585549331363;
    paramsRect.R.at<double>(0, 2) = 0.00295046925795;
    paramsRect.R.at<double>(1, 0) = -0.00586456451925;
    paramsRect.R.at<double>(1, 1) = 0.999978074523;
    paramsRect.R.at<double>(1, 2) = 0.00307528161444;
    paramsRect.R.at<double>(2, 0) = -0.00293239727657;
    paramsRect.R.at<double>(2, 1) = -0.0030925187247;
    paramsRect.R.at<double>(2, 2) = 0.999990918646;

    paramsRect.T.at<double>(0, 0) = -28.9602486665;
    paramsRect.T.at<double>(1, 0) = 0.325331067345;
    paramsRect.T.at<double>(2, 0) = 0.659922450245;

    break;

  case cam::duoParamsSet::CIRCLES2:
    paramsRect.lU.at<double>(0, 0) = 385.798795301;
    paramsRect.lU.at<double>(1, 1) = 386.784692895;
    paramsRect.lU.at<double>(0, 2) = 325.339356744;
    paramsRect.lU.at<double>(1, 2) = 219.118364841;
    paramsRect.lU.at<double>(2, 2) = 1.0;

    paramsRect.lR.at<double>(0, 0) = -0.118873900504;
    paramsRect.lR.at<double>(0, 1) = 0.00876671684306;

    paramsRect.rU.at<double>(0, 0) = 382.270707025;
    paramsRect.rU.at<double>(1, 1) = 382.730131838;
    paramsRect.rU.at<double>(0, 2) = 305.901833445;
    paramsRect.rU.at<double>(1, 2) = 236.768029997;
    paramsRect.rU.at<double>(2, 2) = 1.0;

    paramsRect.rR.at<double>(0, 0) = -0.120308217576;
    paramsRect.rR.at<double>(0, 1) = 0.0176862800281;

    paramsRect.R.at<double>(0, 0) = 0.999977472977;
    paramsRect.R.at<double>(0, 1) = 0.00586909963887;
    paramsRect.R.at<double>(0, 2) = 0.00325687094129;
    paramsRect.R.at<double>(1, 0) = -0.00587875460569;
    paramsRect.R.at<double>(1, 1) = 0.99997833056;
    paramsRect.R.at<double>(1, 2) = 0.00296287948043;
    paramsRect.R.at<double>(2, 0) = -0.00323941093183;
    paramsRect.R.at<double>(2, 1) = -0.00298195908063;
    paramsRect.R.at<double>(2, 2) = 0.999990307021;

    paramsRect.T.at<double>(0, 0) = -28.9635579257;
    paramsRect.T.at<double>(1, 0) = 0.32315785343;
    paramsRect.T.at<double>(2, 0) = 0.650483833158;
    break;
  }
}
