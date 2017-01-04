#pragma once

#include "../pch.h"
#include "../Stereo/dataTypes/view.hpp"

using namespace cv;
using namespace std;

class Scene{
	public:
		Scene(View* view);
		~Scene();
		vector<Obstacle>* getScene();
		void updateFromView();
	public:
		View* view;
	private:
		vector<Obstacle> obstacles;
};
