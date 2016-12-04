#ifndef CAMERACTRL_H
#define CAMERACTRL_H

#include <opencv2/opencv.hpp>
#include <stdio.h>
using namespace cv;

// definitions of function pointers
typedef void (*SetExposureType)(float exposure);
typedef void (*SetGainType)(float gain);
typedef void (*SetDiodesType)(float power);

class CameraCtrl
{
	public:
		/** 
		\brief Constructor
		\param[in] img -- pointer to the OpenCV image
		\param[in] SetExposure -- pointer to the function setting the exposure
		\param[in] SetGain -- pointer to the function setting the gain 
		\param[in] kP -- proportional coefficient for PI controller
		\param[in] kI -- integration coefficient for PI controller
		\param[in] debugOn -- if true, then debug informations are displayed*/
		CameraCtrl( cv::Mat * img, SetExposureType SetExposure, SetGainType SetGain, SetDiodesType SetDiodes, bool debugOn=false );

		/**
		\brief Function controlling the exposure and gain*/
		void Update();

	private:
		// ----- variables 
		SetDiodesType	SetDiodes;
		SetExposureType SetExposure;
		SetGainType		SetGain;
		cv::Mat *		img;		
		bool			debugOn;
		float			intError;
		float			histeresis;

		bool			firstRun;

		float			KP;
		float			KI;
		// ----- constants
//		static const float	KP;
//		static const float	KI;
		static const float  KD;
		static const float	MAX_PI_ERROR;
		static const float	MIN_PI_ERROR;
		//static const float	HISTERESIS;
		static const int	SET_POINT;
		static const int	ROI_WIDTH;
		static const int	ROI_HEIGHT;

		// ----- methods
		int		CalculateAverage( int x1, int y1, int x2, int y2 );
		void	DebugRectangle( int xC, int yC, int width, int height, int value );
		void	DebugInfo( float avg, float exposure, float gain, float diodes );
		float	PI( float error );
};

#ifdef CAMERACTRL_CPP
// ----- constant static variables
//const float CameraCtrl::KP				= 0.05;		// proportional constant in PI controller
//const float CameraCtrl::KI				= 0.2;		// integration constant in PI controller
const float	CameraCtrl::MAX_PI_ERROR	= 300;		// maximum output value from the controller
const float	CameraCtrl::MIN_PI_ERROR	=   0;		// minimum output value from the controller
//const float CameraCtrl::HISTERESIS		=  50;		// histeresis
const int	CameraCtrl::SET_POINT		= 100;		// target average value to which controller should regulate
const int	CameraCtrl::ROI_WIDTH		= 175;		// window width
const int	CameraCtrl::ROI_HEIGHT		= 125;		// window height
#endif
#endif

