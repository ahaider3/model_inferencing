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
     float * input_batch;

     cublasHandle_t handle;

  public:

  NN(int num_layers, std::vector<int> sizes, int batch_size){
     this->num_layers = num_layers;
     this->batch_size = batch_size;
     this->sizes = new int[num_layers];
     for (int i = 0; i < num_layers; i ++) 
	this->sizes[i] = sizes[i];
     this->weights = new float*[num_layers];
     this->outputs = new float*[num_layers];

     cublasCreate(&(this->handle));
     cudaMalloc(&input_batch, batch_size * this->sizes[0] * sizeof(float));
    
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
	float * d_layer, * d_bias, *d_out;
	cudaMalloc(&d_layer, prev_size * this->sizes[i] * sizeof(float));
	this->d_weights[i] = d_layer;
	cudaMalloc(&d_bias, batch_size * this->sizes[i] * sizeof(float));
	this->d_biases[i] = d_bias;
	cudaMalloc(&d_out, batch_size * this->sizes[i] * sizeof(float));
	this->d_outputs[i] = d_out;

	for(int j = 0;j < this->sizes[i] * prev_size; j++){
		this->weights[i][j] = 1.0;
	}
	for(int j = 0;j < this->sizes[i] * batch_size; j++){
		this->outputs[i][j] = 1.0;
		this->biases[i][j] = 1.0;

	}
	cudaMemcpy(this->d_weights[i], this->weights[i], this->sizes[i] * prev_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_biases[i], this->biases[i], this->sizes[i] * batch_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_outputs[i], this->outputs[i], this->sizes[i] * batch_size * sizeof(float), cudaMemcpyHostToDevice);


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
    
    int k = this->sizes[0];
    int m = this->batch_size;
    float alf = 1;
    cudaMemcpy(this->input_batch, batch, k * m * sizeof(float), cudaMemcpyHostToDevice);
    float * input = this->input_batch;
    for (int i =1 ; i < this->num_layers; i ++){
      int n = this->sizes[i];
      gpu_mult(input, this->d_weights[i], this->d_outputs[i], m, k, n);
      cublasSaxpy(this->handle, m *n,  &alf, this->d_biases[i], 1, this->d_outputs[i], 1);

      k = n;
    }
    return input;
  }
  
  void gpu_mult(const float * A, const float * B, float * C, int m, int k, int n){
        float alf = 1;
        float * alpha = & alf;
	float bet = 0;
	float * beta = &bet; 
        cublasSgemm(this->handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, m, B, k, beta, C, m);
  } 
       
  ~NN(){
    for(int i = 1; i < this->num_layers; i ++){
	free(this->weights[i]);
	free(this->outputs[i]);
	free(this->biases[i]);

//	cublasDestroy(this->handle);
    }
   }

};
  

