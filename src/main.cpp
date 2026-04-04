#include <ATen/ops/softmax.h>
#include <c10/core/TensorOptions.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/nn/modules/activation.h>
#include <torch/torch.h>
#include <torch/types.h>
#include <torch/utils.h>
#define INSERT_NEW_LINE() std::cout << std::endl
 
 
 


// #include "bigram.h"
#include "common.h"

#define USE_BIGRAM_MODEL 0
#define USE_MLP_MODEL 1

#if USE_MLP_MODEL
	#include "mlp.h"
#endif

int main() {

#if USE_BIGRAM_MODEL
  bigram_model();
#elif USE_MLP_MODEL
	MLP mlp;
	mlp.build_dataset();
	mlp();
	mlp.train_model(100000);
#endif

// #ifdef DEBUG
//   std::cout << xs.size(0);
//   INSERT_NEW_LINE();
//   std::cout << ys.size(0);
//   INSERT_NEW_LINE();
//   std::cout << xenc.size(0);
//   INSERT_NEW_LINE();
//   matplotlibcpp::backend("TkAgg");
//   torch::Tensor one_hots =
//       xenc.slice(0, 0, 10, 1).to(c10::kCPU).to(torch::kFloat32).contiguous();
//   matplotlibcpp::imshow(one_hots.data_ptr<float>(), 10, 27, 1);
//   matplotlibcpp::show();
// #endif
  return 0;
}

#undef USE_BIGRAM_MODEL