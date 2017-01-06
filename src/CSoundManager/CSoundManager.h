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
		
	
	private:
		Csound *cs;
		unsigned int instr;
        ScanTimer* timer;
		CsoundPerformanceThread* csThread;
		Scene* scene;
        int callbackStatus;


};

