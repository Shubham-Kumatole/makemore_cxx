#include "linear.h"

Linear::Linear(int fan_in, int fan_out, bool bias){
	this->weight = (torch::randn({fan_in, fan_out}, g) / std::sqrt(fan_in)).detach().requires_grad_(true);
	if(bias){
		this->bias = torch::zeros(fan_out).requires_grad_(true);
	}else {
		this->bias = torch::Tensor();
	}
}
