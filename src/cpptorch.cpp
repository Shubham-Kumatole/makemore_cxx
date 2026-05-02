#include "cpptorch.h"
#include <ATen/core/grad_mode.h>
#include <memory>
#include <torch/utils.h>

void build_dataset(torch::Tensor &X, torch::Tensor& Y) {
  std::vector<std::string> words;
  std::vector<int> Y1, X1;
  assert(read_inputs(words));
  std::unordered_map<char, int> stoi;
  std::unordered_map<int, char> itos;
  create_vocabulary(words, stoi, itos);
  for (auto word : words) {
    std::queue<int> context;
    for (int i = 0; i < block_size; i++)
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
  X = torch::tensor(X1).reshape({-1, block_size});
  Y = torch::tensor(Y1);
}

void pytorch_like_training(){
	std::vector<PLayer> layers = {};
  layers.push_back(std::make_shared<Linear>(block_size* n_embed, n_hidden, false));
  layers.push_back(std::make_shared<BatchNorm1d>(n_hidden));
  layers.push_back(std::make_shared<Tanh>());
  layers.push_back(std::make_shared<Linear>(n_hidden, n_hidden, false));
  layers.push_back(std::make_shared<BatchNorm1d>(n_hidden));
  layers.push_back(std::make_shared<Tanh>());
  layers.push_back(std::make_shared<Linear>(n_hidden, n_hidden, false));
  layers.push_back(std::make_shared<BatchNorm1d>(n_hidden));
  layers.push_back(std::make_shared<Tanh>());
  layers.push_back(std::make_shared<Linear>(n_hidden, n_hidden, false));
  layers.push_back(std::make_shared<BatchNorm1d>(n_hidden));
  layers.push_back(std::make_shared<Tanh>());
  layers.push_back(std::make_shared<Linear>(n_hidden, n_hidden, false));
  layers.push_back(std::make_shared<BatchNorm1d>(n_hidden));
  layers.push_back(std::make_shared<Tanh>());
  layers.push_back(std::make_shared<Linear>(n_hidden, vocab_size, false));
  layers.push_back(std::make_shared<BatchNorm1d>(vocab_size));
  int num_layers = layers.size();
  {
    torch::NoGradGuard no_grad;
    auto last_layer = std::static_pointer_cast<BatchNorm1d>(layers[num_layers - 1]);
    last_layer->gain = (last_layer->gain * 0.1).detach();
    for(auto &layer: layers){
      if(std::shared_ptr<Linear> ll = std::dynamic_pointer_cast<Linear>(layer)){
        ll->weight = (ll->weight * 5/3).detach();
      }
    }
  }
	torch::Generator g = torch::make_generator<at::CPUGeneratorImpl>(2147483647);
  torch::Tensor C = torch::randn({vocab_size, n_embed}, g);
  auto parameters = torch::concat(C);
  for(auto &layer: layers){
    parameters = torch::concat({parameters, layer->parameters()});
  }
  for(auto &param : parameters){
    param = param->requires_grad_(true);
  }
  auto max_steps = 200000;
  std::vector<double> lossi, ud;  
  torch::Tensor X, Y;
  build_dataset(X, Y);
  torch::Tensor Xtr = X.slice(0, 0, 0.8 * X.size(0), 1);
  torch::Tensor Ytr = Y.slice(0, 0, 0.8 * Y.size(0), 1);
  torch::Tensor Xdev = X.slice(0, 0.8 * X.size(0), 0.9 * X.size(0), 1);
  torch::Tensor Ydev = Y.slice(0, 0.8 * Y.size(0), 0.9 * Y.size(0), 1);
  torch::Tensor Xtest = X.slice(0, 0.9 * X.size(0), X.size(0), 1);
  torch::Tensor Ytest = Y.slice(0, 0.9 * Y.size(0), Y.size(0), 1); 
  for(int i = 0; i < max_steps; i++){
    auto ix = torch::randint(0, Xtr.size(0), {batch_size}, g);
    auto xb = Xtr.index({ix});
    auto yb = Ytr.index({ix});
    torch::Tensor emb = C.index({xb});
    torch::Tensor embcat = emb.view({ emb.size(0), -1 });
    for(auto &layer:layers){
      embcat = (*layer)(embcat);
    }
    torch::Tensor loss = at::cross_entropy_loss(embcat, yb);
    for(auto &p : parameters){
      p.out.retain_grad();
    }
    for(auto &p : parameters){
      p.mutable_grad() = torch::Tensor();
    }
    loss.backward();
    double lr = i < 150000 ? 0.1 : 0.01;
    for(auto &p : parameters){
      p.data() += -lr * p.grad();
    }
    if(i%10000 == 0){
      printf("%7d/%7d : %.4f", i, max_steps, loss.item().toFloat());
    }
    lossi.push_back(loss.item().toFloat());
    if(i > 1000)break; 
  }
}


