

#include "CSoundManager.h"
#include <chrono>
#include <thread>

int mark1flag;
int mark2flag;
int mark3flag;

CSoundManager::CSoundManager(Scene* newScene, char* csdfile){
	cs = new Csound();
	scene = newScene;
	
	pScanTimer = &scanTimer;
	pInstr = &instr;
	Csound *cs = new Csound();
 	cs->Compile(csdfile); 	
 	mark1flag = 0;
	mark2flag = 0;
	mark3flag = 0;
}

CSoundManager::~CSoundManager(){
	
}


void CSoundManager::Start(){	

	instr = 1;
	scanTimer = 0;
	perfThread = new CsoundPerformanceThread(cs);
	perfThread->Play();
	callback(sonify, 20);
}

void CSoundManager::Stop(){
	
	perfThread->Stop();
	perfThread->Join();
	cs->Stop();
}

void CSoundManager::callback(function<void(Csound*, CsoundPerformanceThread*, Scene*, float, unsigned int*)> func, unsigned int interval)
{
	
    std::thread([func, interval, this]() {
        while (true)
        {
        	
        	auto start = std::chrono::system_clock::now();
        	*(this->pScanTimer)+=interval; 
        	if(*(this->pScanTimer) <= ACTIVE_TIME){
				float scaledTime = (float)(*(this->pScanTimer))/ACTIVE_TIME;	
        		func(this->cs, this->perfThread, this->scene, scaledTime, this->pInstr);
        	}
        	if(*(this->pScanTimer) > ACTIVE_TIME && *(this->pScanTimer) < FULL_SCAN){
        		// clear
        	}
        	if(*(this->pScanTimer) >= FULL_SCAN){
        		*(this->pScanTimer)= 0;
        		*(this->pInstr) = 1;
        		mark1flag = 0;
				mark2flag = 0;
				mark3flag = 0;
        	}
			auto stop = std::chrono::system_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            std::this_thread::sleep_for(std::chrono::milliseconds(interval-(int)duration.count()));
        }
    }).detach();
}

void sonify(Csound* cs, CsoundPerformanceThread* csThread, Scene* scene, float time, unsigned int* freeInstr)
{
	int depth = scene->maxDepth*100*time;
	int area = scene->maxDepth*100*UPDATE_TIME/ACTIVE_TIME;
	int minDepth = depth - area/2;
	int maxDepth = depth + area/2;
	if(depth>100 && mark1flag == 0){
		int instr  = (*freeInstr);
		mark1flag = 1;
		double params[13] = { static_cast<double>(instr), 0, 0.03, 0.7, 1, 1, 500, 8.03, 0.01, 0.5, 2000, 0.5, 0.9 };
		cs->SetChannel(("vol" + std::to_string(instr)).c_str(), 0.7*(1.0f - sqrt((100.0f/100.0f) / scene->maxDepth)));
		cs->SetChannel(("azimuth" + std::to_string(instr)).c_str(), 0.0f);
		csThread->ScoreEvent(0, 'i', 13, params);
		(*freeInstr)++;
	}
	if(depth>250 && mark2flag == 0){
		int instr  = (*freeInstr);
		mark2flag = 1;
		double params[13] = { static_cast<double>(instr), 0, 0.03, 0.7, 1, 1, 500, 8.03, 0.01, 0.5, 2000, 0.5, 0.9 };
		cs->SetChannel(("vol" + std::to_string(instr)).c_str(), 0.7*(1.0f - sqrt((250.0f/100.0f) / scene->maxDepth)));
		cs->SetChannel(("azimuth" + std::to_string(instr)).c_str(), 0.0f);
		csThread->ScoreEvent(0, 'i', 13, params);
		(*freeInstr)++;
	}
	if(depth>450 && mark3flag == 0){
		int instr  = (*freeInstr);
		mark3flag = 1;
		double params[13] = { static_cast<double>(instr), 0, 0.03, 0.7, 1, 1, 500, 8.03, 0.01, 0.5, 2000, 0.5, 0.9 };
		cs->SetChannel(("vol" + std::to_string(instr)).c_str(), 0.7*(1.0f - sqrt((450.0f/100.0f) / scene->maxDepth)));
		cs->SetChannel(("azimuth" + std::to_string(instr)).c_str(), 0.0f);
		csThread->ScoreEvent(0, 'i', 13, params);
		(*freeInstr)++;
	}
	for(auto o : scene->getScene()){
		//std::cout<<minDepth<<" "<<maxDepth<<" "<<o->z<<std::endl;
		if((*freeInstr) == 10) break;
		if(o->z > minDepth && o->z <= maxDepth){
			if(o->type == OBSTACLE){
					//double params[13] = {1, 0, 0.3, 1, 1, 1, 200, 0.05, 0.01, 50, 6000, 0.5, 0.02};
			
				
				int instr  = (*freeInstr);
				double duration = (ACTIVE_TIME) / 1000.0f*(1.0f / scene->maxDepth);
				double aStif = (STIFF_MIN - STIFF_MAX) / (WIDTH_MAX - WIDTH_MIN);
				double stiffness = aStif*(o->width/100.0f) + (STIFF_MIN - WIDTH_MAX * aStif);
				double aSVel = (STRVEL_MAX - STRVEL_MIN) / (scene->maxDepth - scene->minDepth);
				double strikeVelocity = aSVel*(depth/100.0f)+(STRVEL_MIN - scene->maxDepth*aSVel);
				double params[13] = { static_cast<double>(instr), 0, 0.5, 0.2 , 1, 1, stiffness, 0.05, 0.01, 50, strikeVelocity, 0.8, 0.02 };
				cs->SetChannel(("vol" + std::to_string(instr)).c_str(), 0.7*(1.0f - sqrt((o->z/100.0f) / scene->maxDepth)));
				cs->SetChannel(("azimuth" + std::to_string(instr)).c_str(), atan2(4*o->x, (depth)) * TO_DEGREES);
				//player->SetControl(("elev" + to_string(instr)).c_str(), atan2(i->getOriginY(), distance) * TO_DEGREES);
				//player->SetControl(("mute" + to_string(instr)).c_str(), 0);
				//std::cout<<o->z<<" "<<duration<< " " << stiffness <<" "<< atan2(2*o->x, (depth)) * TO_DEGREES<<" "<< 0.7*(1.0f - sqrt((o->z/100.0f) / 5.0f))<<std::endl;
			
				csThread->ScoreEvent(0, 'i', 13, params);
				(*freeInstr)++;
			
			}
			if(o->type == WALL){
					//double params[13] = {1, 0, 0.3, 1, 1, 1, 200, 0.05, 0.01, 50, 6000, 0.5, 0.02};
				std::cout<<"Wall obstacle at z="<<o->z<< std::endl;
				
				int instr  = (*freeInstr);
				
				float angle = abs(atan(o->a)*TO_DEGREES);
				//calculating x coordinate of crossed wall
				float tanAngle = o->a;
				//division by zero secure
				if (abs(tanAngle) < 0.001) {
					if (tanAngle < 0) tanAngle = -0.001f;
					else tanAngle = 0.001f;
				}
				//float x = -(plane.w + plane.z*z) / plane.x;
				
				float timeLength = (abs(depth) / (float)scene->maxDepth)*(ACTIVE_TIME / 1000.0f);
	
				if (timeLength < 0.2) timeLength = 0.2f;
				if (timeLength > 0.5) timeLength = 0.5f;
				float scanSpeed = 2;
				scanSpeed = (abs(angle)) / 4;
				if (scanSpeed < 2) scanSpeed = 2;
				float stiffness = (float)(STIFF_WALL);
				double params[13] = { static_cast<double>(instr), 0, timeLength, 2, 1, 1, stiffness, 0.04, scanSpeed, 100, 500, 0.2, 0.8 };
				cs->SetChannel(("vol" + std::to_string(instr)).c_str(), 0.7*(1.0f - sqrt((o->z/100.0f) / scene->maxDepth)));
	
				
				cs->SetChannel(("azimuth" +std::to_string(instr)).c_str(), atan2(4*o->x, o->z) * TO_DEGREES);
				
				//player->SetControl(("elev" + to_string(i->getInstrument())).c_str(), atan2(i->getOriginY(), z) * TO_DEGREES);
				//player->SetControl(("mute" + to_string(i->getInstrument())).c_str(), 0);
				csThread->ScoreEvent(0, 'i', 13, params);
			
			}  
		}
				
		//csThread->ScoreEvent(0, 'i', 13, params);
	
	
	}
    		
}
