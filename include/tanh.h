#include <torch/torch.h>
#include "ilayer.h"

class Tanh: public ILayer {
public:
	Tanh(){};
	torch::Tensor operator()(torch::Tensor x) override {
		this->out = x.tanh();
		return this->out;
	}

	std::vector<torch::Tensor*> parameters() override {
		return {}; 
	}	
};
