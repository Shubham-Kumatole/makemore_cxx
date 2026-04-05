#include "bn.h"
#include "matplotlibcpp.h"
#include <torch/headeronly/core/DeviceType.h>
#include <torch/types.h>

// void BN::init_weights() {
//     std::cout << "override init weights called " << std::endl;
//     MLP::init_weights();
// }


void BN::plot_histogram_of_weights(){
    assert(W1.defined());
    auto w_normal = W1.to(c10::kCPU).to(torch::kFloat32).contiguous();
    matplotlibcpp::backend("Agg");
    matplotlibcpp::figure_size(1600, 1600);
    matplotlibcpp::imshow(w_normal.data_ptr<float>(), W1.size(0), W1.size(1), 1,  {{"cmap", "Blues"}});
    matplotlibcpp::save("../weight_activations.png");
}