#include "mlp.h"
#include "matplotlibcpp.h"
#include <ATen/core/grad_mode.h>
#include <ATen/ops/multinomial.h>
#include <torch/nn/functional/loss.h>

void MLP::build_dataset() {
  std::vector<std::string> words;
  std::vector<int> Y1, X1;
  assert(read_inputs(words));
  create_vocabulary(words, stoi, itos);
  for (auto word : words) {
    std::queue<int> context;
    for (int i = 0; i < CONTEXT_SIZE; i++)
      context.push(0);
    int ix = 0;
    word = word + ".";
    for (char &c : word) {
      ix = stoi[c];
      Y1.push_back(ix);
      //   std::vector<int> vect;
      std::queue<int> temp_context(context);
      while (!temp_context.empty()) {
        // vect.push_back(std::move(temp_context.front()));
        X1.push_back(std::move(temp_context.front()));
        temp_context.pop();
      }
      context.pop();
      context.push(ix);
    }
  }
  X = torch::tensor(X1).reshape({-1, CONTEXT_SIZE});
  Y = torch::tensor(Y1);
}

void MLP::test_forward_pass() {
  std::cout << X.sizes() << std::endl;
  torch::Tensor C = torch::randn({VOCABULARY_SIZE, EMBEDDING_SPACE_DIM});
  torch::Tensor emb = C.index({X});
  torch::Tensor W1 =
      torch::randn({CONTEXT_SIZE * EMBEDDING_SPACE_DIM, NUM_HIDDEN_NEURONS});
  torch::Tensor b1 = torch::randn(NUM_HIDDEN_NEURONS);
  torch::Tensor W2 = torch::randn({NUM_HIDDEN_NEURONS, VOCABULARY_SIZE});
  torch::Tensor b2 = torch::randn(VOCABULARY_SIZE);
  torch::Tensor h = torch::matmul(emb.view({-1, 6}), W1) + b1;
  torch::Tensor logits = torch::matmul(h.tanh(), W2) + b2;
  std::cout << logits.sizes() << std::endl;
  torch::Tensor counts = logits.exp();
  std::cout << counts << std::endl;
  torch::Tensor probs = counts / counts.sum(1, true);
  torch::Tensor loss =
      -probs.index({torch::range(0, Y.size(0) - 1, 1).to(torch::kI64), Y})
           .log()
           .mean();

  std::cout << loss << std::endl;
}

void MLP::operator()() {
  Xtr = X.slice(0, 0, 0.8 * X.size(0), 1);
  Ytr = Y.slice(0, 0, 0.8 * Y.size(0), 1);
  Xdev = X.slice(0, 0.8 * X.size(0), 0.9 * X.size(0), 1);
  Ydev = Y.slice(0, 0.8 * Y.size(0), 0.9 * Y.size(0), 1);
  Xtest = X.slice(0, 0.9 * X.size(0), X.size(0), 1);
  Ytest = Y.slice(0, 0.9 * Y.size(0), Y.size(0), 1);
  std::cout << Xtr.sizes();
  INSERT_NEW_LINE();
  std::cout << Ytr.sizes();
  INSERT_NEW_LINE();
  std::cout << Xdev.sizes();
  INSERT_NEW_LINE();
  std::cout << Ydev.sizes();
  INSERT_NEW_LINE();
  std::cout << Xtest.sizes();
  INSERT_NEW_LINE();
  std::cout << Ytest.sizes();
  INSERT_NEW_LINE();
  assert(Xtr.size(0) + Xdev.size(0) + Xtest.size(0) == X.size(0));
  std::cout << "dataset is split into train validate and test sets with "
               "80/10/10'%' share each"
            << std::endl;
}

void MLP::init_weights() {
  starting_fresh = false;
  c10::TensorOptions t = c10::TensorOptions();
  t = t.requires_grad(true);
  this->W1 = torch::randn({CONTEXT_SIZE * EMBEDDING_SPACE_DIM, NUM_HIDDEN_NEURONS}, g,
                    t);
  this->b1 = torch::randn(NUM_HIDDEN_NEURONS, g, t);
  this->W2 = torch::randn({NUM_HIDDEN_NEURONS, VOCABULARY_SIZE}, g, t);
  this->b2 = torch::randn(VOCABULARY_SIZE, g, t);
  this->C = torch::randn({VOCABULARY_SIZE, EMBEDDING_SPACE_DIM}, g, t);
}

void MLP::clear_weights() {
  starting_fresh = true;
  lossi.clear();
  stepi.clear();
}

void MLP::clear_grads() {
  W1.mutable_grad() = torch::Tensor();
  W2.mutable_grad() = torch::Tensor();
  b1.mutable_grad() = torch::Tensor();
  b2.mutable_grad() = torch::Tensor();
  C.mutable_grad() = torch::Tensor();
}
void MLP::train_model(int num_training_loops) {
  if (starting_fresh) {
    this->init_weights();
  }
  for (int i = 0; i < num_training_loops; i++) {
    torch::Tensor ix = torch::randint(0, Xtr.size(0), {BATCH_SIZE});
    torch::Tensor emb = C.index({Xtr.index({ix})});
    torch::Tensor h = emb.view({-1, CONTEXT_SIZE * EMBEDDING_SPACE_DIM})
                          .matmul(W1)
                          .add(b1)
                          .tanh();
    torch::Tensor logits = h.matmul(W2).add(b2);
    torch::Tensor loss =
        at::cross_entropy_loss(logits, Ytr.index({ix})) +
        0.001 * (W1.square().mean() + b1.square().mean() + W2.square().mean() +
                 b2.square().mean() + C.square().mean());
    clear_grads();
    loss.backward();
    auto lr = i < 100000 ? 0.1 : 0.01;
    update_params(lr);
    stepi.push_back(i);
    lossi.push_back(loss.log10().item().toDouble());
  }
  plot_losses();
  train_loss();
  validate_loss();
  test_loss();
}

void MLP::update_params(double learning_rate) {
  torch::NoGradGuard no_grad;
  W1.data() += -learning_rate * W1.grad();
  W2.data() += -learning_rate * W2.grad();
  b1.data() += -learning_rate * b1.grad();
  b2.data() += -learning_rate * b2.grad();
  C.data() += -learning_rate * C.grad();
}

void MLP::plot_losses() {
  matplotlibcpp::backend("Agg");
  matplotlibcpp::plot(stepi, lossi);
  matplotlibcpp::save("../mlp.png");
}

void MLP::validate_loss() {
  torch::NoGradGuard no_grad;
  auto emb = C.index({Xdev});
  auto h = emb.view({-1, CONTEXT_SIZE * EMBEDDING_SPACE_DIM})
               .matmul(W1)
               .add(b1)
               .tanh();
  auto logits = h.matmul(W2) + b2;
  auto loss = torch::nn::functional::cross_entropy(logits, Ydev);
  std::cout << "Validation dataset loss = " << loss.item().toDouble()
            << std::endl;
}

void MLP::train_loss() {
  torch::NoGradGuard no_grad;
  auto emb = C.index({Xtr});
  auto h = emb.view({-1, CONTEXT_SIZE * EMBEDDING_SPACE_DIM})
               .matmul(W1)
               .add(b1)
               .tanh();
  auto logits = h.matmul(W2) + b2;
  auto loss = torch::nn::functional::cross_entropy(logits, Ytr);
  std::cout << "training dataset loss = " << loss.item().toDouble()
            << std::endl;
}

void MLP::test_loss() {
  torch::NoGradGuard no_grad;
  auto emb = C.index({Xtest});
  auto h = emb.view({-1, CONTEXT_SIZE * EMBEDDING_SPACE_DIM})
               .matmul(W1)
               .add(b1)
               .tanh();
  auto logits = h.matmul(W2) + b2;
  auto loss = torch::nn::functional::cross_entropy(logits, Ytest);
  std::cout << "test dataset loss = " << loss.item().toDouble() << std::endl;
}

void MLP::sample_model(int num_iters) {
  torch::NoGradGuard no_grad;
  std::vector<int> vect(CONTEXT_SIZE, 0);
  while (num_iters--) {
    torch::Tensor ctxt = torch::tensor(vect);
    while (true) {
      auto emb = C.index({ctxt});
      auto h = emb.view({-1, CONTEXT_SIZE * EMBEDDING_SPACE_DIM})
                   .matmul(W1)
                   .add(b1)
                   .tanh();
      auto logits = h.matmul(W2).add(b2);
      auto counts = logits.exp();
      auto probs = counts / counts.sum(1, true);
      auto ypred = torch::multinomial(probs, 1);
      if (ypred[0].item().toInt() == 0)
        break;
      std::cout << itos[ypred.item().toInt()];
      ctxt = torch::cat({ctxt.slice(0, 1), ypred[0]}, 0);
    }
    std::cout << std::endl;
  }
}