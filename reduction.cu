%%writefile reduction.cu
#include <stdio.h>
#include <stdlib.h>


// CUDA Practice: Level 2 Reduction
// The problem: Add up ALL elements of an array using the GPU.
// array = [1, 2, 3, 4, 5, 6, 7, 8]
// result = 36  (sum of everything)

__global__ void AddElementArray(int* input,int* result, int n) {

// 1. declare the temporary shared memory: 
 __shared__ int  sharedData[256];

// 2. position in the block
int ThreadIndex = threadIdx.x;

// 3. Global index
int i = blockIdx.x * blockDim.x + threadIdx.x;


// 4. load array element into a shared memrory
// 256 threads running. each thread load only one element. 
sharedData[ThreadIndex] = input[i];


// 5.Wait for all the threads
__syncthreads();


// 6.Only threads whose ThreadID is divisible by [2 * stride] participate
for (int stride = 1; stride < blockDim.x; stride *= 2) {
     if (ThreadIndex % (stride * 2) == 0) {  
     sharedData[ThreadIndex] += sharedData[ThreadIndex + stride];
    } 

__syncthreads();   // wait until the end
    }
    
// write the results
if (ThreadIndex == 0) {  
result[blockIdx.x] = sharedData[0]; 

 }
} // closing kernel


// main() = runs on CPU, handles memory and setup
    
int main ( )  {

int  n  =  1 <<  20; // --> 2^20 = 1,048,576 elements 

size_t size = n * sizeof(int);  // size of all the integers. 


// allocate memory for the CPU
int* inputHost  = (int*)malloc(size); 
int* outputHost = (int*)malloc(size);



int threadsPerBlock =  256;
int block = n / threadsPerBlock;

// fill input host with values
 for (int i = 0; i < n; i++) {
 inputHost[i] = i;
 }

// compute the sum in the CPU
int cpu_sum = 0;
for (int i = 0; i < n; i++) {
cpu_sum   +=  inputHost[i];   
}

// allocate memory for the GPU 
int * inputDevice, * outputDevice;
cudaMalloc(&inputDevice, size);
cudaMalloc(&outputDevice, size);


// copy data from CPU to GPU 
cudaMemcpy(inputDevice, inputHost, size, cudaMemcpyHostToDevice);

// run the kernel 
AddElementArray <<< block, threadsPerBlock>>> (
   inputDevice,
   outputDevice,
    n
 );

// copy back 
cudaMemcpy(outputHost, outputDevice, size, cudaMemcpyDeviceToHost);

// ===========================
// Compare the CPU and GPU 
// ===========================
int gpu_sum = 0; 
for (int i = 0; i < block; i++)  {
gpu_sum  += outputHost[i];   
}

if (cpu_sum  == gpu_sum) {
  printf("The GPU computation is correct");  
} else {
  printf("GPU computation is wrong");
}

// free GPU allocated memory
cudaFree(inputDevice);
cudaFree(outputDevice);

// free CPU allocated memory: 
free(inputHost);
free(outputHost);
return 0;
  }
