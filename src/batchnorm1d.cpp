#include "batchnorm1d.h"
#include <torch/utils.h>

BatchNorm1d::BatchNorm1d(size_t dims, double eps, double momentum){
	this->momentum = torch::tensor(momentum);
	this->eps = torch::tensor(eps);
	this->bias = torch::zeros(dims);
	this->gain = torch::ones(dims);
	this->running_mean = torch::zeros(dims);
	this->running_var = torch::ones(dims);
}


torch::Tensor BatchNorm1d::operator()(torch::Tensor x) override {
	torch::Tensor xvar, xmean;
	if(this->training){
		xvar = x.var(0, true);
		xmean = x.mean(0, true);
	}else {
		xvar = this->running_var;
		xmean = this->running_mean;
	}
	auto xhat =   (x - xmean) / torch::sqrt(xvar + this->eps);
	this->out = this->gain * xhat + bias;
	if(this->training){
		torch::NoGradGuard no_grad;
		this->running_var = (1 - this->momentum) * this->running_var + this->momentum * xvar;
		this->running_mean = (1 - this->momentum) * this->running_mean + this->momentum * xmean;
	}
	return this->out;
}

torch::Tensor BatchNorm1d::parameters() override {
	return torch::zeros(0);
}
