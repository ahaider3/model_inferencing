#include<vector>
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include "cublas_v2.h"

class NN {

  private:
     int num_layers;
     int batch_size;
     int * sizes;
     float ** weights;
     float ** biases;
     float ** outputs;
     float ** d_weights;
     float ** d_biases;
     float ** d_outputs;


  public:

  NN(int num_layers, std::vector<int> sizes, int batch_size){
     this->num_layers = num_layers;
     this->batch_size = batch_size;
     this->sizes = new int[num_layers];
     for (int i = 0; i < num_layers; i ++) 
	this->sizes[i] = sizes[i];
     this->weights = new float*[num_layers];
     this->outputs = new float*[num_layers];


    
//     std::cout<< "NUM LAYERS: " << this->num_layers << "\n" <<
// 	"BATCH SIZE: " << this->batch_size << "\n" ;

     this->biases = new float*[num_layers];

     this->d_weights = new float*[num_layers];
     this->d_biases = new float*[num_layers];
     this->d_outputs = new float*[num_layers];

     int prev_size = this->sizes[0];

     for (int i = 1; i < num_layers; i ++){
	float * layer = (float *)malloc(prev_size * this->sizes[i] * sizeof(float));
	this->weights[i] = layer;
	float * bias = (float *)malloc(batch_size * this->sizes[i] * sizeof(float));
	this->biases[i] = bias;
	float * out = (float *)malloc(batch_size * this->sizes[i] * sizeof(float));
	this->outputs[i] = out;
     	assert(this->weights[i] != NULL &&
	       this->outputs[i] != NULL &&
	       this->biases[i] != NULL);
	for(int j = 0;j < this->sizes[i] * prev_size; j++){
		this->weights[i][j] = 1.0;
	}
	for(int j = 0;j < this->sizes[i] * batch_size; j++){
		this->outputs[i][j] = 1.0;
		this->biases[i][j] = 1.0;

	}
	float * d_layer, * d_bias, *d_out;
	cudaMalloc(&d_layer, prev_size * this->sizes[i] * sizeof(float));
	this->d_weights[i] = d_layer;
	cudaMalloc(&d_bias, batch_size * this->sizes[i] * sizeof(float));
	this->d_biases[i] = d_bias;
	cudaMalloc(&d_out, batch_size * this->sizes[i] * sizeof(float));
	this->d_outputs[i] = d_out;


	prev_size = this->sizes[i];
        
     }

 /*    for (int i =1; i < num_layers; i ++){
       float ** layer = new float*[prev_size];
       for (int j = 0; j < prev_size; j ++){
         layer[j] = new float[this->sizes[i]];
       }
       this->biases[i] = new float[this->sizes[i]];
       this->weights[i] = layer;
     }*/
   }

  float *  forward_pass(float * batch){
//    float * in = (float *)mkl_malloc(2 * 784 * sizeof(float), 64);
 
    
    int k = this->sizes[0];
    int m = this->batch_size;
    float * input = batch;
    for (int i =1 ; i < this->num_layers; i ++){
      int n = this->sizes[i];
      input = this->outputs[i];
      
      k = n;
    }
    return input;
  }
  
       
  ~NN(){
    for(int i = 1; i < this->num_layers; i ++){
	free(this->weights[i]);
	free(this->outputs[i]);
	free(this->biases[i]);
    }
   }

};
  

