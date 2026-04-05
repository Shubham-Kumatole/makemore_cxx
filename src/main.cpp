#define INSERT_NEW_LINE() std::cout << std::endl

// #include "bigram.h"
#include "common.h"

#define USE_BIGRAM_MODEL 0
#define USE_MLP_MODEL 1

#if USE_BIGRAM_MODEL
#include "bigram.h"
#elif USE_MLP_MODEL
#include "mlp.h"
#endif

int main() {

#if USE_BIGRAM_MODEL
  bigram_model();
#elif USE_MLP_MODEL
  MLP mlp(5, 10, 100);
  mlp.build_dataset();
  mlp();
  mlp.init_weights();
  mlp.plot_activations_of_weights();
//   mlp.train_model(200000);
//   mlp.sample_model(20);
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
  //       xenc.slice(0, 0, 10,
  //       1).to(c10::kCPU).to(torch::kFloat32).contiguous();
  //   matplotlibcpp::imshow(one_hots.data_ptr<float>(), 10, 27, 1);
  //   matplotlibcpp::show();
  // #endif
  return 0;
}

#undef USE_BIGRAM_MODEL