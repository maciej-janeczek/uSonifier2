#pragma once
#include "../pch.h"
#include "../Scene/Scene.h"
#include <csound/csound.hpp>
#include <csound/csPerfThread.hpp>
#include "ScanTimer.h"
#include <functional>

class CSoundManager{
	public:
		CSoundManager(Scene* newScene, char* csdfile);
		~CSoundManager();
		void Start();
		void Stop();
        void sonifyObstacles();
        void sonifyMarkers();
	private:
		void callback();

        void playNote(Obstacle o);
        ///Notes simplifiers
        int getInstrument();
        double getVolume(double distance);
        double getAzimuth(float x, float z);
        double getElevation(float x, float z);
        double getStiffness(double distanceFromPath, double pathWidth);

		
	
	private:
		Csound *cs;
		unsigned int instr;
        ScanTimer* timer;
		CsoundPerformanceThread* csThread;
		Scene* scene;
        int callbackStatus;


};

