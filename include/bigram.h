
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
        for(size_t i=1; i < ch.size(); i++){
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
    for(int i = 0; i < 2000; i++){
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