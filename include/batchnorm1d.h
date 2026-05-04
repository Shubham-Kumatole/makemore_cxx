#include <torch/torch.h>
#include "ilayer.h"

class BatchNorm1d: public ILayer {
public:
	torch::Tensor eps, gain, bias, running_mean, running_var, momentum;
	BatchNorm1d(size_t dim, double eps = 1e-5, double momentum = 0.1);
	std::vector<torch::Tensor*> parameters() override{
		return {&gain, &bias};
	}
	torch::Tensor operator()(torch::Tensor x) override  {
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
};
