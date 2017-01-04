#pragma once
#include "../pch.h"

namespace obst{
	enum TYPE{
		obstacle,
		wall
	};
}
class Obstacle{
	public:
		Obstacle(cv::Rect bBox);
		~Obstacle();
        float isOnPath(float width);
	public:
		float x;
		float z_dist;
		float width;
		float dist;

		obst::TYPE type;
        cv::Rect bBox;
};
