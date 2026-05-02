#include "common.h"
#include "matplotlibcpp.h"
#include <torch/torch.h>
#include <vector>
#ifndef INSERT_NEW_LINE
#define INSERT_NEW_LINE() std::cout << std::endl
#endif
class MLP {
protected:
  torch::Tensor X, Y;
  int VOCABULARY_SIZE = 27;
  int EMBEDDING_SPACE_DIM = 2;
  int CONTEXT_SIZE = 3;
  int NUM_HIDDEN_NEURONS = 100;
  int BATCH_SIZE = 32;
  torch::Tensor W1, W2, b2, C;
  bool starting_fresh = true;
  torch::Generator g = at::make_generator<at::CPUGeneratorImpl>(2147483647);
  torch::Tensor Xtr, Ytr, Xdev, Ydev, Ytest, Xtest, bnmean, bnstd, bngain, bnbias;
  std::vector<int> stepi;
  std::vector<double> lossi;
  std::unordered_map<char, int> stoi;
  std::unordered_map<int, char> itos;

public:
  MLP(int context_size = 3, int embedding_space_dim = 2,
      int num_hidden_neurons = 100)
      : EMBEDDING_SPACE_DIM(embedding_space_dim), CONTEXT_SIZE(context_size),
        NUM_HIDDEN_NEURONS(num_hidden_neurons) {}
  void build_dataset();

  void test_forward_pass();
  void operator()();
  void init_weights();
  void clear_weights();
  void train_model(int num_training_loops);
  void clear_grads();
  void update_params(double learning_rate);
  void plot_losses();
  void validate_loss();
  void test_loss();
  void train_loss();
  void sample_model(int num_iters);
  void plot_activations_of_weights();
};
