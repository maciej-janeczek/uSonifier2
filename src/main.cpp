#include "pch.h"
#include "CSoundManager/CSoundManager.h"
#include "Stereo/stereo.hpp"
#include "Camera/Camera.h"
#include <iostream>

#define FL_BL 10.0 //FOCAL_LENGTH*BASELINE*1000


int main(void){
    /// Initialize DUO MLX camera
    cam::Camera *camera;
    camera = new cam::Camera(cam::type::DUO, stereo::Size2d(640, 480, 1), cam::color::GRAY, 30, 0, 0, 0, 0);

    /// Create View context with binded Camera
    View* view;
    view = new View(camera);

    /// Create depth calculation for particular View
    stereo::Macher macher(view, 0, 63, 1, 80);
	//cudaDeviceProp devProp;
   	//cudaGetDeviceProperties(&devProp,0);

	
	namedWindow( "Display", WINDOW_AUTOSIZE);
	
	while((cvWaitKey(5) & 0xff) != 27)
	{
        camera->update();
        camera->rectify();
        imshow("Display", camera->left);
	}
	waitKey(0);
	
}


	/*

	//char* csdfile = "/home/ubuntu/Documents/DUO3D-ARM-v1.0.50.26/DUOSDK/Samples/OpenCV/Sample-01-cvShowImage/csound.csd";
	//Scene *scene = new Scene(SCENE_MIN_DEPTH, SCENE_MAX_DEPTH);
	//CSoundManager *cs = new CSoundManager(scene, csdfile);
	//cs->Start();
	//cs->Stop();
}



/*    double fxL = 385.675380919;
    double fyL = 385.954329526;
    double cxL = 324.171669424;
    double cyL = 223.929227171;
    lU.at<double>(0, 0) = fxL;
    lU.at<double>(1, 1) = fyL;
    lU.at<double>(0, 2) = cxL;
    lU.at<double>(1, 2) = cyL;
    lU.at<double>(2, 2) = 1.0;
    
    double r1L = -0.111450237218;
    double r2L = -0.0177005620878;
    Mat lR = Mat::zeros(1, 4, CV_64FC1);
    lR.at<double>(0,0) = r1L;
    lR.at<double>(0,1) = r2L;
    
    double fxR = 382.23167021;
    double fyR = 382.439655001;
    double cxR = 305.931305581;
    double cyR = 241.47113977;
    rU.at<double>(0, 0) = fxR;
    rU.at<double>(1, 1) = fyR;
    rU.at<double>(0, 2) = cxR;
    rU.at<double>(1, 2) = cyR;
    rU.at<double>(2, 2) = 1.0;
    
    double r1R = -0.11242560225;
    double r2R = -0.00762108671039;
    Mat rR = Mat::zeros(1, 4, CV_64FC1);
    rR.at<double>(0,0) = r1R;
    rR.at<double>(0,1) = r2R;
    
    Mat R = Mat::zeros(3, 3, CV_64FC1);
    R.at<double>(0, 0) = 0.999989431215;
    R.at<double>(0, 1) = 0.00459342955651;
    R.at<double>(0, 2) = -0.000194586152118;
    R.at<double>(1, 0) = -0.00459281664531;
    R.at<double>(1, 1) = 0.999984828646;
    R.at<double>(1, 2) = 0.00304113696687;
    R.at<double>(2, 0) = 0.000208552448411;
    R.at<double>(2, 1) = -0.00304021112722;
    R.at<double>(2, 2) = 0.9999953568;
   
    Mat T = Mat::zeros(3, 1, CV_64FC1);
    T.at<double>(0, 0) = -30.2446439979;
    T.at<double>(1, 0) = 0.382365480057;
    T.at<double>(2, 0) = 1.35237960952;
    */
    
    //CIRCLES 1
    /*double fxL = 385.781537922;
    double fyL = 386.770182219;
    double cxL = 325.28735922;
    double cyL = 219.111593446;
    lU.at<double>(0, 0) = fxL;
    lU.at<double>(1, 1) = fyL;
    lU.at<double>(0, 2) = cxL;
    lU.at<double>(1, 2) = cyL;
    lU.at<double>(2, 2) = 1.0;
    
    double r1L = -0.115369267631;
    double r2L = 0.0;
    Mat lR = Mat::zeros(1, 4, CV_64FC1);
    lR.at<double>(0,0) = r1L;
    lR.at<double>(0,1) = r2L;
    
    double fxR = 382.188010972;
    double fyR = 382.624359906;
    double cxR = 305.959504;
    double cyR = 236.717857874;
    rU.at<double>(0, 0) = fxR;
    rU.at<double>(1, 1) = fyR;
    rU.at<double>(0, 2) = cxR;
    rU.at<double>(1, 2) = cyR;
    rU.at<double>(2, 2) = 1.0;
    
    double r1R = -0.113198501368;
    double r2R = 0.0;
    Mat rR = Mat::zeros(1, 4, CV_64FC1);
    rR.at<double>(0,0) = r1R;
    rR.at<double>(0,1) = r2R;
    
    Mat R = Mat::zeros(3, 3, CV_64FC1);
    R.at<double>(0, 0) = 0.999978503734;
    R.at<double>(0, 1) = 0.00585549331363;
    R.at<double>(0, 2) = 0.00295046925795;
    R.at<double>(1, 0) = -0.00586456451925;
    R.at<double>(1, 1) = 0.999978074523;
    R.at<double>(1, 2) = 0.00307528161444;
    R.at<double>(2, 0) = -0.00293239727657;
    R.at<double>(2, 1) = -0.0030925187247;
    R.at<double>(2, 2) = 0.999990918646 ;
   
    Mat T = Mat::zeros(3, 1, CV_64FC1);
    T.at<double>(0, 0) = -28.9602486665;
    T.at<double>(1, 0) = 0.325331067345;
    T.at<double>(2, 0) = 0.659922450245;
    */
    
    
        //CIRCLES 2
    /*    double fxL = 385.798795301;
    double fyL = 386.784692895;
    double cxL = 325.339356744;
    double cyL = 219.118364841;
    lU.at<double>(0, 0) = fxL;
    lU.at<double>(1, 1) = fyL;
    lU.at<double>(0, 2) = cxL;
    lU.at<double>(1, 2) = cyL;
    lU.at<double>(2, 2) = 1.0;
    
    double r1L = -0.118873900504;
    double r2L = 0.00876671684306;
    Mat lR = Mat::zeros(1, 4, CV_64FC1);
    lR.at<double>(0,0) = r1L;
    lR.at<double>(0,1) = r2L;
    
    double fxR = 382.270707025;
    double fyR = 382.730131838;
    double cxR = 305.901833445;
    double cyR = 236.768029997;
    rU.at<double>(0, 0) = fxR;
    rU.at<double>(1, 1) = fyR;
    rU.at<double>(0, 2) = cxR;
    rU.at<double>(1, 2) = cyR;
    rU.at<double>(2, 2) = 1.0;
    
    double r1R = -0.120308217576;
    double r2R = 0.0176862800281;
    Mat rR = Mat::zeros(1, 4, CV_64FC1);
    rR.at<double>(0,0) = r1R;
    rR.at<double>(0,1) = r2R;
    
    Mat R = Mat::zeros(3, 3, CV_64FC1);
    R.at<double>(0, 0) = 0.999977472977;
    R.at<double>(0, 1) = 0.00586909963887;
    R.at<double>(0, 2) = 0.00325687094129;
    R.at<double>(1, 0) = -0.00587875460569;
    R.at<double>(1, 1) = 0.99997833056;
    R.at<double>(1, 2) = 0.00296287948043;
    R.at<double>(2, 0) = -0.00323941093183;
    R.at<double>(2, 1) = -0.00298195908063;
    R.at<double>(2, 2) = 0.999990307021 ;
   
    Mat T = Mat::zeros(3, 1, CV_64FC1);
    T.at<double>(0, 0) = -28.9635579257;
    T.at<double>(1, 0) = 0.32315785343;
    T.at<double>(2, 0) = 0.650483833158;
    */
    
