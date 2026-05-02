#pragma once
#include <torch/torch.h>

class ILayer{
public:
	torch::Tensor out;
	bool training = true;
	torch::Generator g = torch::make_generator<at::CPUGeneratorImpl>(2147483647);
	virtual torch::Tensor parameters() = 0;
	virtual torch::Tensor operator()(torch::Tensor x) = 0;
	virtual ~ILayer() = default;
};
