
#include "caffe/cc/core/cc_utils.h"
#include <import-staticlib.h>
#include <Windows.h>
#include <vector>
#include <map>

using namespace cc;
using namespace std;

class MyLayer : public AbstractCustomLayer{
public:
	SETUP_LAYERFUNC(MyLayer)

	MyLayer(){

	}

	virtual ~MyLayer(){
		printf("Îö¹¹ÁË.\n");
	}

	virtual const char* type(){
		return "MyPrint";
	}

	virtual void setup(const char* name, const char* type, const char* param_str, int phase, vector<Blob*>& bottom, vector<Blob*>& top){
		map<string, string> param = parseParamStr(param_str);

		top[0]->Reshape(1, 3, 224, 224);
		top[1]->Reshape(1, 1, 1, 1);
		printf("phase = %d\n", phase);
	}

	virtual void forward(vector<Blob*>& bottom, vector<Blob*>& top){

	}

	virtual void backward(vector<Blob*>& bottom, vector<bool>& propagate_down, vector<Blob*>& top){

	}

	virtual void reshape(vector<Blob*>& bottom, vector<Blob*>& top){

	}
};

void main(){

	installRegister();
	INSTALL_LAYER(MyLayer);

	SetCurrentDirectoryA("I:/cc");

	WPtr<Solver> solver = loadSolverFromPrototxt("solver.prototxt");
	while (solver->iter() < solver->max_iter()){
		solver->step(1);
		printf("loss: %f\n", solver->smooth_loss());
	}
}