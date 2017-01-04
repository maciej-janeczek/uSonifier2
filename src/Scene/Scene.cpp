#include "Scene.h"

#define OBSTACLE 1
#define WALL 2

Scene::Scene(View* view){
	this->view = view;
}

Scene::~Scene(){
}

vector<Obstacle>* Scene::getScene(){

	return &obstacles;
}

void Scene::updateFromView(){
    vector<Obstacle>* p_obstacles = view->getObstacles();
	obstacles.clear();
    obstacles.reserve(p_obstacles->size());
    copy(p_obstacles->begin(),p_obstacles->end(),back_inserter(obstacles)); /// Create deep copy of the vector

    if(DEBUG_EN){
        for (auto &o : obstacles) // access by reference to avoid copying
        {
            std::cout << "Obstacle: x=" << o.x <<" z=" << o.z_dist << " width=" << o.width << " Dist=" << o.dist << std::endl;
        }
    }

}
