//
// Created by maciek on 06.01.17.
//
#include "ScanTimer.h"

ScanTimer::ScanTimer(int interval, float minDistance, float maxDistance, timer::TYPE type, float activeTime, float scanTime) {
    this->interval = interval;
    this->minDistance = minDistance;
    this->maxDistance = maxDistance;
    this->type = type;
    this->scanStartTime = std::chrono::system_clock::now();
    this->currentTime = this->scanStartTime;
    this->activeTime = activeTime;
    this->scanTime = scanTime;
}

ScanTimer::~ScanTimer() {

}

timer::STATUS ScanTimer::update(){
    ///Get status of the scan and update ramges
    currentTime = std::chrono::system_clock::now();
    long time = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - scanStartTime).count();

    if(time <= activeTime){
        float timePercNear = (float)(time-interval/2)/activeTime;
        rangeMin = (maxDistance-minDistance) * timePercNear + minDistance;
        float timePercFar = (float)(time+interval/2)/activeTime;
        rangeMax = (maxDistance-minDistance) * timePercFar + minDistance;
        return timer::STATUS::ACTIVE;

    }else if(time < scanTime){
        rangeMax = maxDistance;
        rangeMin = maxDistance;
        return timer::STATUS::DELAY;

    }else{
        this->scanStartTime = std::chrono::system_clock::now();
        return timer::STATUS::RESET;
    }
}

long ScanTimer::getSleepTime(){
    auto current = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current - currentTime);
    return interval-duration.count();
}


