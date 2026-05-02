#include "ilayer.h"
#include "tanh.h"
#include "linear.h"
#include "batchnorm1d.h"
#include "common.h"

using PLayer = std::shared_ptr<ILayer>; 

const int block_size = 3;
const int n_embed = 10;
const int n_hidden = 200;
const int batch_size = 32;
const int vocab_size = 27;

void pytorch_like_training();
