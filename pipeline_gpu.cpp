#include "nn_gpu.h"
#include<vector>
#include <time.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>

int main(int argc, char**argv){
  std::vector<int> layers = std::vector<int>();
  layers.push_back(std::stoi(argv[3]));
  layers.push_back(512);
  layers.push_back(128);
  layers.push_back(1);
  int num_iter = std::stoi(argv[1]);
  int batch_size = std::stoi(argv[2]);

  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
 NN nn = NN(layers.size(), layers, batch_size);
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
 
  float * batch = (float*)malloc(layers[0] * batch_size * sizeof(float));
  assert(batch != NULL);
  for (int i =0; i < layers[0]*batch_size; i ++)
    batch[i] = 1;

  float * pred;

  std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
  for (int i = 0; i < num_iter; i ++){

    pred = nn.forward_pass(batch);
}
  std::chrono::steady_clock::time_point t4= std::chrono::steady_clock::now();
  

//  std::cout << "Total Time: " << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t1).count() <<std::endl;
  std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t4- t3).count()/float(num_iter) <<std::endl;
//  std::cout << "NN Construct: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() <<std::endl;
//  std::cout << "Batch Creation: " << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() <<std::endl;




}
