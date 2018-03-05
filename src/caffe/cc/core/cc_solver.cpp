

#include "caffe/cc/core/cc.h"
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include <math.h>
#include <iostream>
#include <caffe/sgd_solvers.hpp>
#include "caffe/util/signal_handler.h"

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

	void Solver::Snapshot(){
		ptr->Snapshot();
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

	Solver::~Solver(){
	}

#ifdef USE_PROTOBUF
	CCAPI Solver* CCCALL newSolverFromProto(const caffe::SolverParameter* solver_param){
		return GetSolver<float>(*solver_param)->ccSolver();
	}
#endif

	CCAPI Solver* CCCALL loadSolverFromPrototxt(const char* solver_prototxt){
		caffe::SolverParameter solver_param;
		caffe::ReadProtoFromTextFileOrDie(solver_prototxt, &solver_param);

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