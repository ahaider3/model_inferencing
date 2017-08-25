#include "nn.h"
#include<vector>

int main(){
  std::vector<int> layers = std::vector<int>();
  layers.push_back(784);
  layers.push_back(512);
  layers.push_back(128);
  layers.push_back(1);

  int batch_size = 16;
  NN nn = NN(layers.size(), layers, batch_size);
  float * batch = (float*)mkl_malloc(784 * batch_size * sizeof(float), 64);
  assert(batch != NULL);
  for (int i =0; i < 784*batch_size; i ++)
    batch[i] = 1;
  float * pred = nn.forward_pass(batch);
  std::cout<< pred[0] << "\n";
}
