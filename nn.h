#include<vector>
#include<mkl.h>
#include <iostream>
#include <cassert>


class NN {

  private:
     int num_layers;
     int batch_size;
     int * sizes;
     float ** weights;
     float ** biases;
     float ** outputs;
     float ** outputs1;

  public:

  NN(int num_layers, std::vector<int> sizes, int batch_size){
     this->num_layers = num_layers;
     this->batch_size = batch_size;
     this->sizes = new int[num_layers];
     for (int i = 0; i < num_layers; i ++) 
	this->sizes[i] = sizes[i];
     this->weights = new float*[num_layers];
     this->outputs = new float*[num_layers];
     std::cout<< "NUM LAYERS: " << this->num_layers << "\n" <<
 	"BATCH SIZE: " << this->batch_size << "\n" ;

     this->biases = new float*[num_layers];
     int prev_size = this->sizes[0];

     for (int i = 1; i < num_layers; i ++){
	float * layer = (float *)mkl_malloc(prev_size * this->sizes[i] * sizeof(float), 64);
	this->weights[i] = layer;
	float * bias = (float *)mkl_malloc(batch_size * this->sizes[i] * sizeof(float), 64);
	this->biases[i] = bias;
	float * out = (float *)mkl_malloc(batch_size * this->sizes[i] * sizeof(float), 64);
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
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		m, n, k,
		1.0, input, k, this->weights[i],
		n, 1, this->outputs[i], n);
      std::cout<< "Finished GEMM for Layer: " << i << "\n";
      cblas_saxpy(m * this->sizes[i],1, this->biases[i],0,
		this->outputs[i], 0);
      input = this->outputs[i];
      
      k = n;
    }
    return input;
  }
  
       
  ~NN(){
    for(int i = 1; i < this->num_layers; i ++){
	mkl_free(this->weights[i]);
	mkl_free(this->outputs[i]);
	mkl_free(this->biases[i]);
    }
   }

};
  

