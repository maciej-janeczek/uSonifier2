
#include "../pch.h"
#include "Obstacle.h"
#include "../Stereo/dataTypes/view.hpp"

using namespace cv;
using namespace std;

class Scene{
	public:
		Scene(View* view, int minDepth, int maxDept);
		~Scene();
		vector<Obstacle*> getScene();
		void updateFromView();
	public:
		View* view;
		int maxDepth;
		int minDepth;
	private:
		vector<Obstacle*> obstacles;
};
