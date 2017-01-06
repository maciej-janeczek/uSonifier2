//
// Created by maciek on 06.01.17.
//
#pragma once
#include <chrono>

namespace timer {
    enum TYPE {
        GROWING_SPHERE,
        MOVING_PLANE
    };

    enum STATUS {
        ACTIVE,
        DELAY,
        RESET
    };
}

class ScanTimer {
public:
    ScanTimer(int interval, float minDistance, float maxDistance, timer::TYPE type, float activeTime, float scanTime);
    ~ScanTimer();
    timer::STATUS update();
    long getSleepTime();

    float rangeMin;
    float rangeMax;

private:
    timer::TYPE type;
    int interval;
    float minDistance;
    float maxDistance;
    float scanTime;
    float activeTime;
    std::chrono::time_point<std::chrono::system_clock> scanStartTime;
    std::chrono::time_point<std::chrono::system_clock> currentTime;
};


