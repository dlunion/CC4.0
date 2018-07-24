

#include "caffe/cc/core/cc.h"
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include <math.h>
#include <iostream>
#include <caffe/sgd_solvers.hpp>
#include "caffe/util/signal_handler.h"

#ifdef USE_CC_PYTHON
#include <memory>

#undef _DEBUG
#include <Python.h>
#endif

using namespace std;
using namespace cv;

namespace cc{

#define cvt(p)	((caffe::Solver<float>*)p)
#define ptr		(cvt(this->_native))

	template <typename Dtype>
	caffe::Solver<Dtype>* GetSolver(const caffe::SolverParameter& param) {
		//return new SGDSolver<Dtype>(param);
		caffe::SolverParameter_SolverType type = param.solver_type();

		switch (type) {
		case caffe::SolverParameter_SolverType_SGD:
			return new caffe::SGDSolver<Dtype>(param);
		case caffe::SolverParameter_SolverType_NESTEROV:
			return new caffe::NesterovSolver<Dtype>(param);
		case caffe::SolverParameter_SolverType_ADAGRAD:
			return new caffe::AdaGradSolver<Dtype>(param);
		case caffe::SolverParameter_SolverType_RMSPROP:
			return new caffe::RMSPropSolver<Dtype>(param);
		case caffe::SolverParameter_SolverType_ADADELTA:
			return new caffe::AdaDeltaSolver<Dtype>(param);
		case caffe::SolverParameter_SolverType_ADAM:
			return new caffe::AdamSolver<Dtype>(param);
		default:
			LOG(FATAL) << "Unknown SolverType: " << type;
		}
		return (caffe::Solver<Dtype>*) NULL;
	}

	void Solver::setNative(void* native){
		this->_native = native;
	}

	void* Solver::getNative(){
		return ptr;
	}

	CCAPI void CCCALL releaseSolver(Solver* solver){
		if (solver){
			void* p = solver->getNative();
			if (p) delete cvt(p);
		}
	}

	void Solver::Restore(const char* resume_file){
		ptr->Restore(resume_file);
	}

	void Solver::Snapshot(const char* filepath, bool save_solver_state){
		ptr->Snapshot(filepath, save_solver_state);
	}

#ifdef USE_PROTOBUF
	caffe::SolverParameter& Solver::solver_param(){
		return ptr->param_;
	}
#endif

	Solver::Solver(){
		static caffe::SignalHandler singalHandler(
			caffe::SolverAction::STOP,
			caffe::SolverAction::SNAPSHOT);
		this->signalHandler_ = &singalHandler;
	}

	void Solver::setBaseLearningRate(float rate){
		ptr->param_.set_base_lr(rate);
	}

	float Solver::getBaseLearningRate(){
		return ptr->param_.base_lr();
	}

	void Solver::postSnapshotSignal(){
		ptr->postSnapshotSignal();
	}

	bool Solver::getParam(const char* path, Value& val){
		return getMessageValue(&ptr->param_, path, val);
	}

	MessageHandle Solver::param(){
		return &ptr->param_;
	}

	void Solver::TestAll(){
		ptr->TestAll();
	}

	Solver::~Solver(){
	}

#ifdef USE_PROTOBUF
	CCAPI Solver* CCCALL newSolverFromProto(const caffe::SolverParameter* solver_param){
		return GetSolver<float>(*solver_param)->ccSolver();
	}
#endif

	static Solver* buildSolver(caffe::SolverParameter& solver_param){
		// Set device id and mode
		if (solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
			LOG(INFO) << "Use GPU with device ID " << solver_param.device_id();
			caffe::Caffe::SetDevice(solver_param.device_id());
			caffe::Caffe::set_mode(caffe::Caffe::GPU);
		}
		else {
			LOG(INFO) << "Use CPU.";
			caffe::Caffe::set_mode(caffe::Caffe::CPU);
		}
		return GetSolver<float>(solver_param)->ccSolver();
	}

	bool loadSolverNetFromString(caffe::SolverParameter& solver_param, const char* netstring = 0){
		if (netstring){
			//如果提供字符串，则优先使用这个
			solver_param.clear_net();
			solver_param.clear_net_param();
			solver_param.clear_train_net();
			solver_param.clear_test_net();
			solver_param.clear_test_net_param();
			solver_param.clear_train_net_param();
			caffe::NetParameter netp;
			caffe::ReadNetParamsFromTextStringOrDie(netstring, &netp);
			solver_param.mutable_net_param()->CopyFrom(netp);
		}
		return true;
	}

#ifdef USE_CC_PYTHON
	static bool initPython(){
		if (Py_IsInitialized()) return true;

		Py_Initialize();
		return Py_IsInitialized();
	}

	static void destoryPython(){
		Py_Finalize();
	}

	static void getpathDirName(const char* path, string& dir, string& name){
		char dir_[260] = { 0 };
		char filebuf_[260] = { 0 };
		const char* p0 = strrchr(path, '\\');
		const char* p1 = strrchr(path, '/');
		char* p = 0;
		if (p0 && p1){
			p = (char*)(p0 > p1 ? p0 : p1);
		}
		else{
			p = (char*)(p0 ? p0 : p1);
		}

		if (!p){
			strcpy(dir_, ".");
			strcpy(filebuf_, path);
		}
		else{
			strncpy(dir_, path, p - path);
			strcpy(filebuf_, p + 1);
		}
		dir = dir_;
		name = filebuf_;
	}

	class Pyobj{
	public:
		Pyobj(){
		}

		static void Deleter(PyObject* obj){
			if (obj){
				Py_DECREF(obj);
			}
		}

		Pyobj(PyObject* obj){
			native_.reset(obj, Deleter);
		}

		operator PyObject* (){
			return this->native_.get();
		}

		PyObject* get(){
			return this->native_.get();
		}

	private:
		std::shared_ptr<PyObject> native_;
	};

	static string getfuncName(const string& name){
		int p = name.rfind(':');
		if (p != -1)
			return name.substr(p + 1);

		return name;
	}

#define thisModule		(*((Pyobj*)this->module_))
	CCPython::CCPython(){
		initPython();
		module_ = new Pyobj();
	}

	CCPython::~CCPython(){
		delete ((Pyobj*)module_);
	}
	 
	bool CCPython::load(const char* file){
		string name, dir;
		getpathDirName(file, dir, name);
		PyRun_SimpleString("import sys");
		PyRun_SimpleString(format("sys.path.append(r'%s')", dir.c_str()).c_str());
		thisModule = PyImport_ImportModule(name.c_str());
		if (!thisModule.get()){
			lasterror_ = format("load module fail: %s\n", name.c_str()).c_str();
			return false;
		}
		return true;
	}

	CCString CCPython::callstringFunction(const CCString& name, CCString& errmsg){
		Pyobj pDict = PyModule_GetDict(thisModule);
		if (!pDict) {
			errmsg = "get dict error.\n";
			return "";
		}

		Pyobj keys = PyDict_Keys(pDict);
		Pyobj pFunc = PyDict_GetItemString(pDict, name.c_str());
		if (!pFunc || !PyCallable_Check(pFunc)) {
			errmsg = format("can't find function: %s", name.c_str()).c_str();
			return "";
		}

		char* str = 0;
		Pyobj returnval = PyObject_CallFunction(pFunc, 0);
		PyArg_Parse(returnval, "s", &str);
		return str;
	}

	CCString CCPython::train_ptototxt(){
		return callstringFunction(getfuncName(__FUNCTION__).c_str(), lasterror_);
	}

	CCString CCPython::deploy_prototxt(){
		return callstringFunction(getfuncName(__FUNCTION__).c_str(), lasterror_);
	}

	CCString CCPython::solver(){
		return callstringFunction(getfuncName(__FUNCTION__).c_str(), lasterror_);
	}

	CCString CCPython::last_error(){
		return lasterror_;
	}

	CCAPI Solver* CCCALL loadSolverFromPython(const char* pythonfile){
		initPython();
		Solver* solver_ptr = 0;
		{
			CCPython pp;
			bool ok = pp.load(pythonfile);
			CHECK(ok) << pp.last_error();
			if (!ok) return 0;

			string solver = pp.solver();
			string train = pp.train_ptototxt();
			 
			caffe::SolverParameter solver_param;
			CHECK(caffe::ReadProtoFromTextString(solver, &solver_param));
			CHECK(loadSolverNetFromString(solver_param, train.c_str()));
			solver_ptr = buildSolver(solver_param);
		}
		return solver_ptr;
	}
#endif

	CCAPI Solver* CCCALL loadSolverFromPrototxtString(const char* solver_prototxt_string, const char* netstring){
		caffe::SolverParameter solver_param;
		caffe::ReadProtoFromTextString(solver_prototxt_string, &solver_param);
		if (!loadSolverNetFromString(solver_param, netstring))
			return 0;
		
		return buildSolver(solver_param);
	}

	CCAPI Solver* CCCALL loadSolverFromPrototxt(const char* solver_prototxt, const char* netstring){
		caffe::SolverParameter solver_param;
		caffe::ReadProtoFromTextFileOrDie(solver_prototxt, &solver_param);
		if (!loadSolverNetFromString(solver_param, netstring))
			return 0;

		return buildSolver(solver_param);
	}

	void Solver::installActionSignalOperator(){
		ptr->SetActionFunction(((caffe::SignalHandler*)this->signalHandler_)->GetActionFunction());
	}

	void Solver::step(int iters){
		ptr->Step(iters);
	}

	int Solver::max_iter(){
		return ptr->param().max_iter();
	}

	float Solver::smooth_loss(){
		return ptr->smooth_loss();
	}

	int Solver::num_test_net(){
		return ptr->test_nets().size();
	}
	
	Net* Solver::test_net(int index){
		if (index < 0 || index >= num_test_net())
			return 0;

		return ptr->test_nets()[index]->ccNet();
	}

	Net* Solver::net(){
		return ptr->net()->ccNet();
	}

	int Solver::iter(){
		return ptr->iter();
	}

	void Solver::Solve(){
		installActionSignalOperator();
		ptr->Solve();
	}
}