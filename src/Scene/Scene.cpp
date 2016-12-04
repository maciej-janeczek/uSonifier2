#include "Scene.h"

#define OBSTACLE 1
#define WALL 2

Scene::Scene(int minDepth, int maxDepth){
	this->minDepth = minDepth;
	this->maxDepth = maxDepth;
}

Scene::~Scene(){
}

vector<Obstacle*> Scene::getScene(){
	return obstacles;//Should be coppied
}

void Scene::update(vector<vector<Point>*> points){
	obstacles.clear();
	
	for(size_t i = 0;i<points.size(); i++){
		vector<Point>* pPoints;
		pPoints = points[i];
		Obstacle *obstacle = new Obstacle();
		int sumX = 0;
		int sumZ = 0;
		int maxX = -600;
		int maxXz = 0;
		int minX = 600;
		int minXz = 0;
		int maxZ = 0;
		int maxZx = 0;
		int minZ = 700;
		int minZx = 0;
		for(size_t j = 0;j<pPoints->size(); j++){
			Point* pPoint;
			pPoint = &((*pPoints)[j]);
			int x = (pPoint->x);
			int z = (300-(pPoint->y));
			if(x>maxX) {
				maxX = x;
				maxXz = z;
			}
			if(x<minX){
			 	minX = x;
			 	minXz = z;
			 }
			if(z>maxZ) {
				maxZ = z;
				maxZx = x;
			}
			if(z<minZ){ 
				minZ = z;
				minZx = x;	
			}
			sumX+=x;
			sumZ+=z;
		}
	
		obstacle->z = (sumZ/float(pPoints->size())+minZ);
		obstacle->x = (sumX/float(pPoints->size()))-160;
		obstacle->dist = int(sqrt(obstacle->x*obstacle->x+obstacle->z*obstacle->z));
		
		obstacle->minZ = minZ;
		obstacle->width = maxX-minX;
		obstacle->depth = maxZ-minZ;
		obstacle->minX = minX;
		obstacle->type = OBSTACLE;
		obstacle->avX = (maxX+minX)/2;
		if(obstacle->z>200){
			int dimX = sqrt((maxX-minX)*(maxX-minX)+(maxXz-minXz)*(maxXz-minXz));
			int dimZ = sqrt((maxZ-minZ)*(maxZ-minZ)+(maxZx-minZx)*(maxZx-minZx));
			int dim = sqrt(obstacle->width*obstacle->width+obstacle->depth*obstacle->depth);
			float az = ((float)(maxZ-minZ))/(maxZx-minZx);
			float ax = ((float)(maxXz-minXz))/(maxX-minX);	
			
			if((dimX > 100 || dimZ > 100 || dim >130)) {
				obstacle->type = WALL;
				
				float bz = maxZ - az*maxZx;
				float bx = maxXz- ax*maxX;
				if(ax<az){
					obstacle->a = ax;
					obstacle->b = bx;
				}
				else{
					obstacle->a = ax;
					obstacle->b = bx;
				}
				
			}
		}	
		obstacles.push_back(obstacle);
	}
}
