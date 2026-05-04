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


