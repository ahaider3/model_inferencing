#include<vector>
#include<mkl.h>



class NN {

  private:
     int num_layers;
     int * sizes;
     float *** weights;
     float ** biases;
  public:

  NN(int num_layers, std::vector<int> sizes){
    
     this->num_layers = num_layers;
     this->sizes = &sizes[0];
     this->weights = new float**[num_layers];
     this->biases = new float*[num_layers];
     int prev_size = this->sizes[0];
     

     for (int i =1; i < num_layers; i ++){
       float ** layer = new float*[prev_size];
       for (int j = 0; j < prev_size; j ++){
         layer[j] = new float[this->sizes[i]];
       }
       this->biases[i] = new float[this->sizes[i]];
       this->weights[i] = layer;
     }
   }

  float * forward_pass(float ** batch){
    cblas_dgemm(CblasRowMajor, CblasNoTrans,CblasNoTrans,
      1, 1, 1, 1, weights[0], 
    return; 
  }
    
  
       
  

};
     
