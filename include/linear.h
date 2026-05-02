#include <torch/torch.h>
#include "ilayer.h"

class Linear : public ILayer {
public:
	torch::Tensor weight, bias;
	Linear(int fan_in, int fan_out, bool bias);

	torch::Tensor operator()(torch::Tensor x) override{
		this->out = x.matmul(this->weight);
		if(this->bias.defined())
			this->out = (this->out + this->bias).detach().requires_grad_(true);
		return this->out;
	}

	torch::Tensor parameters() override {
		if(this->bias.defined()){
			return torch::concat({this->weight, this->bias});
		}
		return this->weight;
	}
};

