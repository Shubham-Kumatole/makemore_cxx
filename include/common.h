#include <torch/torch.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <fstream>

bool read_inputs(std::vector<std::string>& words);
void create_vocabulary(std::vector<std::string>& words, 
                       std::unordered_map<char,int>& stoi,
                       std::unordered_map<int,char>& itos);
void create_dataset(std::vector<std::string>& words,
                    torch::Tensor& xs, torch::Tensor& ys,
                    std::unordered_map<char,int>& stoi);