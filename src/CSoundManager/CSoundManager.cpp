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
    timer = new ScanTimer(20, scene->view->distMin, scene->view->distMax, timer::TYPE::GROWING_SPHERE,1500, 2000);
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
    vector<Obstacle>* obstacles = this->scene->getScene();
    for (auto &o : *obstacles)
    {
        ///sonify obstacle if it is near surface (within range determined by the interval
        if(o.dist >= timer->rangeMin && o.dist < timer->rangeMax){
            playNote(o);
        }
    }
}

void CSoundManager::playNote(Obstacle o){
    int i = getInstrument();
    if(i){
        double volume = getVolume(o.dist);
        double stiffness = getStiffness(o.isOnPath(0.8), 0.8);
        double azimuth = getAzimuth(o.x, o.z_dist);
        double elevation = getElevation(o.x, o.z_dist);

        cs->SetChannel(("vol" + std::to_string(i)).c_str(),  volume);
        cs->SetChannel(("azimuth" + std::to_string(i)).c_str(),  azimuth);
        cs->SetChannel(("elev" + std::to_string(i)).c_str(),  elevation);
        //std::cout << "I:" << i << " Vol:" << volume << " Stiff:" << stiffness << " Azimuth:" << azimuth << std::endl;
        double params[15] = { static_cast<double>(i), 0, volume, 0.2,
                              1, 1, stiffness, 0.05, 0.01, stiffness, 10000, 0.8, 0.02, 0, 0};
        csThread->ScoreEvent(0, 'i', 15, params);
    }

}

void CSoundManager::sonifyMarkers() {
    double time1 = 0.1/(scene->view->distMax) * (1.5);
    double vol1 = getVolume(0.1);
    double params1[5] = { static_cast<double>(31), time1, 0.5, 0.7, 31};
    cs->SetChannel(("vol" + std::to_string(31)).c_str(),  0.4*vol1);
    cs->SetChannel(("azimuth" + std::to_string(31)).c_str(), 0.0f);
    csThread->ScoreEvent(0, 'i', 5, params1);


    double time2 = 2.0/(scene->view->distMax) * (1.5);
    double params2[5] = { static_cast<double>(32), time2, 0.5, 0.7, 31};
    double vol2 = getVolume(2.0);
    cs->SetChannel(("vol" + std::to_string(32)).c_str(),  0.4*vol2);
    cs->SetChannel(("azimuth" + std::to_string(32)).c_str(), 0.0f);
    csThread->ScoreEvent(0, 'i', 5, params2);


    double time3 = 4.0/(scene->view->distMax) * (1.5);
    double params3[5] = { static_cast<double>(33), time3, 0.5, 0.7, 31};
    double vol3 = getVolume(4.0);
    cs->SetChannel(("vol" + std::to_string(33)).c_str(),  0.4*vol3);
    cs->SetChannel(("azimuth" + std::to_string(33)).c_str(), 0.0f);
    csThread->ScoreEvent(0, 'i', 5, params3);
}

int CSoundManager::getInstrument(){
    if(instr == 10) return 0;
    return instr++;
}

///Notes simplifiers
double CSoundManager::getVolume(double distance){
    return pow((double)(1.0 - (distance) / scene->view->distMax), 2.0)/2.0;
}

double CSoundManager::getAzimuth(float x, float z){
    return tan(x/z)*TO_DEGREES;
}

double CSoundManager::getElevation(float x, float z){
    ///Assumed that all obstacles are on the head lvl
    return 0.0;
    //TODO:
    //Obstacles located on the ground

}

double CSoundManager::getStiffness(double distanceFromPath, double pathWidth){
    double min = 50.0;
    double max = 300.0;
    double distMax = (2.0-pathWidth/2.0);
    return (min-max) * (distanceFromPath/distMax) + max;
}


