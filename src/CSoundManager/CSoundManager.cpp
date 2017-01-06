#include "CSoundManager.h"
#include <chrono>
#include <thread>

CSoundManager::CSoundManager(Scene* newScene, char* csdfile){
	cs = new Csound();
	scene = newScene;
	//cs = new Csound();
 	cs->Compile(csdfile);
}

CSoundManager::~CSoundManager(){
    delete timer;
}

void CSoundManager::Start(){	

	instr = 1;
	csThread = new CsoundPerformanceThread(cs);
    csThread->Play();
    timer = new ScanTimer(20, scene->view->distMax, scene->view->distMin, timer::TYPE::GROWING_SPHERE,1500, 2000);
    std::thread([this] {callback();} ).detach();
}

void CSoundManager::Stop(){
    callbackStatus = 0;
	csThread->Stop();
	csThread->Join();
	cs->Stop();
}

void CSoundManager::callback()
{
    int markersPlayed = 0;
    this->callbackStatus = 1;
    while (callbackStatus)
    {
        timer::STATUS status = this->timer->update();

        if(status == timer::STATUS::ACTIVE){
            if(markersPlayed == 0){
                sonifyMarkers();
                markersPlayed = 1;
            }
            sonifyObstacles();
        }
        else if(status == timer::STATUS::RESET){
            markersPlayed = 0;
            this->instr = 1;
            continue;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(this->timer->getSleepTime()));
    }
}


void CSoundManager::sonifyObstacles()
{

}

void CSoundManager::sonifyMarkers() {
    double time1 = 0.1/(scene->view->distMax) * (1.5);
    float vol1 = pow((float)(1.0f - (0.1) / scene->view->distMax), 2.0f)/2.0f;
    double params1[5] = { static_cast<double>(31), time1, 0.5, 0.7, 31};
    cs->SetChannel(("vol" + std::to_string(31)).c_str(),  0.7*vol1);
    cs->SetChannel(("azimuth" + std::to_string(31)).c_str(), 0.0f);
    csThread->ScoreEvent(0, 'i', 5, params1);


    double time2 = 2.0/(scene->view->distMax) * (1.5);
    double params2[5] = { static_cast<double>(32), time2, 0.5, 0.7, 31};
    float vol2 = pow((float)(1.0f - (2.0) / scene->view->distMax), 2.0f)/2.0f;
    cs->SetChannel(("vol" + std::to_string(32)).c_str(),  0.7*vol2);
    cs->SetChannel(("azimuth" + std::to_string(32)).c_str(), 0.0f);
    csThread->ScoreEvent(0, 'i', 5, params2);


    double time3 = 4.0/(scene->view->distMax) * (1.5);
    double params3[5] = { static_cast<double>(33), time3, 0.5, 0.7, 31};
    float vol3 = pow((float)(1.0f - (4.0) / scene->view->distMax), 2.0f)/2.0f;
    cs->SetChannel(("vol" + std::to_string(33)).c_str(),  0.7*vol3);
    cs->SetChannel(("azimuth" + std::to_string(33)).c_str(), 0.0f);
    csThread->ScoreEvent(0, 'i', 5, params3);
}


