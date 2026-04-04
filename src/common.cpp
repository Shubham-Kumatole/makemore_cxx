#include "common.h"
bool read_inputs(std::vector<std::string> & words){
    std:: cout << "reading names from file \n" << std::endl;
    std::ifstream input;
    input.open("./names.txt", std::ifstream::in);
    std::string word;
    while(input >> word && input.good()){
        words.push_back(word);
    }
    if(word.length()){
        words.push_back(word);
    }
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
