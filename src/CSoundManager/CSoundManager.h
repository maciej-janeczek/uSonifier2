
#include "../pch.h"
#include "../Scene/Scene.h"
#include <functional>
#include <csound/csound.hpp>
#include <csound/csPerfThread.hpp>

//void sonify(Csound*, CsoundPerformanceThread*, Scene* , float , unsigned int* );

class CSoundManager{
	public:
		CSoundManager(Scene* newScene, char* csdfile);
		~CSoundManager();
		void Start();
		void Stop();
        void sonify(Csound* cs, CsoundPerformanceThread* csThread, Scene* scene, float time, unsigned int* freeInstr);
	private:
		void call(unsigned int);
		
	
	private:
		Csound *cs;
		unsigned int instr;
		unsigned int* pInstr;
		unsigned int scanTimer;
		unsigned int* pScanTimer;
		CsoundPerformanceThread* perfThread;
		Scene* scene;


};

