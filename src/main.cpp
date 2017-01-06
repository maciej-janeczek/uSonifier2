#include "pch.h"
#include "CSoundManager/CSoundManager.h"
#include "Stereo/stereo.hpp"
#include "Camera/Camera.h"
#include <iostream>

#define FL_BL 10.0 //FOCAL_LENGTH*BASELINE*1000


int main(int argc, char **argv){
    /// Initialize DUO MLX camera
    cam::Camera *camera;
    camera = new cam::Camera(cam::type::DUO, stereo::Size2d(640, 480, 1), cam::color::GRAY, 10, 1, 1, 0, 1);

    /// Create View context with binded Camera
    View* view;
    view = new View(camera, 0.1f, 5.0f, 4.0f);

    /// Create depth calculation for particular View
    stereo::Macher* macher;
    macher = new stereo::Macher(view, 0, 63, 1, 80);

    /// Create Scene and bind view to it
    Scene* scene = new Scene(view);

    /// Create CsoundManager and bind Scene to it
    char* csdfile = "../resources/MovingPlane.csd";
    CSoundManager csound(scene, csdfile);
    csound.Start();

    /// Init display window
	namedWindow( "Display", WINDOW_AUTOSIZE);


    /// Start pipeline with csound working in separate thread
    float* dataOut = (float*)malloc(640 * 480 * sizeof(float));
	while((cvWaitKey(5) & 0xff) != 27)
	{
        /// image from camera -> disparity -> u-disparity -> depth segmentation -> get obstacles
        view->updateFromCam();
        macher->perform_AEMBM();
        view->depthSegmentation();
        scene->updateFromView();

        /// display image
        //view->grab(dataOut);
        //cv::Mat image3 = cv::Mat(480, 640, CV_32FC1, dataOut);
        //cv::Mat image1 = cv::Mat(480, 640, CV_8UC4, view->left_cpu);
        imshow("Display1", view->depthPrev);
        //imshow("Display2", view->depthRectPrev);
        //imshow("Display3", image1);
	}
	waitKey(0);

    csound.Stop();

    delete (camera);
    delete (view);
    delete (macher);
    return 1;
}
