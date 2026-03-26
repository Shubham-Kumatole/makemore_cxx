#include "matplotlibcpp.h"
#include <ATen/core/Generator.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/log.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/multinomial.h>
#include <ATen/ops/one_hot.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/xlogy_ops.h>
#include <c10/core/TensorOptions.h>
#include <fstream>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/torch.h>
#include <torch/types.h>
#include <torch/utils.h>
#include <unordered_map>
#define INSERT_NEW_LINE() std::cout<<std::endl

bool read_inputs(std::vector<std::string> & words){
    std:: cout << "reading names from file \n" << std::endl;
    std::ifstream input;
    input.open("../names.txt", std::ifstream::in);
    std::string word;
    while(input >> word && input.good()){
        words.push_back(word);
    }
    words.push_back(word);
    #ifdef DEBUG
    std::cout<<words.size()<<std::endl;
    INSERT_NEW_LINE();
    #endif
    return words.size() > 0;
}

void create_vocabulary(std::vector<std::string> &words, std::unordered_map<char, int> &stoi, std::unordered_map<int, char> &itos){
    int index =0;
    stoi['.'] = index;
    itos[index++] = '.';
    for(auto &word: words){
        for(auto ch:word){
            if(stoi.find(ch) == stoi.end()){
                stoi[ch] = ch - 'a' + 1;
                itos[ch - 'a' + 1] = ch;
            }
        }
    }
    #ifdef DEBUG
    std::cout << "Total number of tokens = " << stoi.size() <<std::endl;
    INSERT_NEW_LINE();
    #endif
}

void create_dataset(std::vector<std::string> &words, torch::Tensor &xs,torch::Tensor &ys, std::unordered_map<char, int> &stoi){
    char curr_char;
    std::vector<int> xs1, ys1;
    for(auto word:words){
        curr_char = '.';
        for(char ch: word){
            xs1.push_back(stoi[curr_char]);
            ys1.push_back(stoi[ch]);
            curr_char = ch;
        }
        xs1.push_back(stoi[curr_char]);
        ys1.push_back(stoi['.']);
    }
    
    xs = torch::tensor(xs1);
    ys = torch::tensor(ys1);
}

void plot_bigram(torch::Tensor& N, std::unordered_map<int, char>& itos){
    matplotlibcpp::backend("Agg"); // use Agg to avoid the macOS threading crash

    auto N_normalized = N.to(torch::kCPU).to(torch::kFloat32).contiguous();
    matplotlibcpp::figure_size(1600, 1600);
    matplotlibcpp::imshow(
        N_normalized.data_ptr<float>(),
        27, 27, 1,
        {{"cmap", "Blues"}}
    );

    for(int i = 0; i < 27; i++){
        for(int j = 0; j < 27; j++){
            std::string bigram = "";
            bigram += itos[i];
            bigram += itos[j];
            std::string count = std::to_string((int)N[i][j].item().toFloat());
            matplotlibcpp::text(j, i, bigram, {{"ha","center"},{"va","bottom"},{"color","gray"}});
            matplotlibcpp::text(j, i, count,  {{"ha","center"},{"va","top"},   {"color","gray"}});
        }
    }

    matplotlibcpp::axis("off");
    matplotlibcpp::save("../bigram.png");
}


void bigram_model(){
    std::vector<std::string> words;
    assert(read_inputs(words));
    std::unordered_map<char, int> stoi;
    std::unordered_map<int, char> itos;
    create_vocabulary(words, stoi, itos);
    torch::Tensor xs, ys;
    create_dataset(words,xs, ys, stoi);
    torch::Tensor N = torch::zeros({27, 27}, torch::kInt32);
    for(auto i = 0; i < xs.size(0); i++){
        N[xs[i]][ys[i]] += 1;
    }
    plot_bigram(N, itos);
    torch::Tensor P = N.add(1).to(torch::kFloat32);
    P = P/P.sum(1, true);
    int ix1, ix2;
    torch::Tensor prob, log_likelihood = torch::tensor(0.0) , n = torch::tensor(0);
    for(auto w: words){
        std::string ch = "." + w + ".";
        for(auto i=1; i < ch.size(); i++){
            ix1 = stoi[ch[i-1]];
            ix2 = stoi[ch[i]];
            prob = P[ix1][ix2];
            log_likelihood += torch::log(prob);
            n+=1;
        }
    }
    std::cout << "Log Likelihood = " << log_likelihood << std::endl;
    torch::Tensor nll = -log_likelihood;
    std::cout << "Negative log likelihood = " << nll << std::endl;
    std::cout << "loss = " << nll/n << std::endl;
    std::cout << std::endl;
    auto g = at::make_generator<at::CPUGeneratorImpl>(2147483647);
    c10::TensorOptions t = c10::TensorOptions();
    t = t.requires_grad( true);
    torch::Tensor W = torch::randn({27,27}, g, t);
    torch::Tensor xenc, logits, counts, probs, ts, loss;
    for(int i = 0; i < 150; i++){
        xenc = torch::one_hot(xs, 27).to(torch::kFloat32);
        logits = torch::matmul(xenc, W);
        counts = logits.exp();
        probs = counts / counts.sum(1, true);
        ts = torch::arange(xs.size(0));
        loss = -probs.index({ts, ys}).log().mean() + 0.01 * W.pow(2.0).mean();
        W.mutable_grad() = torch::Tensor();
        loss.backward();
        {
            torch::NoGradGuard no_grad;
            W.data() -= 50.0 * W.grad();
        }
    }
    INSERT_NEW_LINE();
    INSERT_NEW_LINE();
    std::cout << loss << std::endl;
    INSERT_NEW_LINE();
    INSERT_NEW_LINE();
    std::cout << " Running inference loop five times " << std::endl;
    for(int i = 0; i < 5; i++){
        int ix = 0;
        do{
            torch::NoGradGuard no_grad;
            xenc = torch::one_hot( torch::tensor(ix), 27).to(torch::kFloat32);
            logits = torch::matmul(xenc, W);
            counts = logits.exp();
            probs = counts / counts.sum();
            ix = torch::multinomial(probs, 1, true, g).item().toInt();
            std::cout << itos[ix];
        } while(ix != 0);
        std::cout << std::endl;
    }
    return;
}

int main(){
    std::vector<std::string> words;
    assert(read_inputs(words));
    std::unordered_map<char, int> stoi;
    std::unordered_map<int, char> itos;
    create_vocabulary(words, stoi, itos);
    torch::Tensor xs, ys;
    create_dataset(words,xs, ys, stoi);
    bigram_model();
#ifdef DEBUG
    std::cout << xs.size(0);
    INSERT_NEW_LINE();
    std::cout << ys.size(0);
    INSERT_NEW_LINE();
#endif
    torch::Tensor xenc = torch::one_hot(xs, stoi.size());
#ifdef DEBUG
    std::cout << xenc.size(0);
    INSERT_NEW_LINE();
    matplotlibcpp::backend("TkAgg");
    torch::Tensor one_hots = xenc.slice(0, 0, 10,1).to(c10::kCPU).to(torch::kFloat32).contiguous();
    matplotlibcpp::imshow(one_hots.data_ptr<float>(), 10, 27, 1);
    matplotlibcpp::show();
#endif
    return 0;
}