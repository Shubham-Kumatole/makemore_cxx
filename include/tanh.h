#include <torch/torch.h>
#include "ilayer.h"

class Tanh: public ILayer {
public:
	Tanh(){};
	torch::Tensor out;
	torch::Tensor operator()(torch::Tensor x) override {
		this->out = x.tanh();
		return this->out;
	}

	torch::Tensor parameters() override {
		return torch::zeros(0);
	}	
};
