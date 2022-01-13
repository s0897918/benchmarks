#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "fp16_conversion.h"

using namespace std;

// #define FP16MM

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

inline
cublasStatus_t checkCublas(cublasStatus_t result)
{
  if (result != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cublasGetErrorString(result));
    assert(result == CUBLAS_STATUS_SUCCESS);
  }
  return result;
}

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on CPU
void CPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
	int a=1;

    for(int i = 0; i < nr_rows_A * nr_cols_A; i++){
		A[i] = (float)rand()/(float)(RAND_MAX/a);
	}
}

void CPU_fill_rand_double(double *A, int nr_rows_A, int nr_cols_A) {
	int a=1;

    for(int i = 0; i < nr_rows_A * nr_cols_A; i++){
		A[i] = (double)rand()/(double)(RAND_MAX/a);
	}
}


int main(int argc, char ** argv)
{



    if (argc < 2) {
    
	cout << "ERROR ARGS PROGRAM: \n" << endl;
	
	return 1;
    }

    std::string exe_file = argv[0];
    std::string data_type = argv[1];


    cout << exe_file << " " <<data_type << endl;

	if (data_type.compare("fp32") == 0) {
       cout << "data type is fp32, cublasSgemm test result:" << endl;
    }
    else if (data_type.compare("fp64") == 0) {
       cout << "data type is fp64, cublasDgemm test result:" << endl;
    }
	else {
	   cout << "ERROR ARGS" << endl;
	   return 1;
	}
    

  int min_m_k_n = 2;
  int max_m_k_n = 4096*8;
  int repeats = 1;
  int verbose = 1;
 
  if(verbose) 
    cout << "running with" 
	 << " min_m_k_n: " << min_m_k_n
	 << " max_m_k_n: " << max_m_k_n
	 << " repeats: " << repeats
	 << endl;

  cublasStatus_t stat;
  cublasHandle_t handle;

  checkCublas(cublasCreate(&handle));

  if(verbose) cout << "allocating device variables" << endl;
  
  // Allocate 3 arrays on CPU
  if  (data_type.compare("fp32") == 0 ) 
  {
	  float *h_A = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(float));
	  float *h_B = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(float));
	  float *h_C = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(float));
	  
	  CPU_fill_rand(h_A, max_m_k_n, max_m_k_n);
	  CPU_fill_rand(h_B, max_m_k_n, max_m_k_n);
	  CPU_fill_rand(h_C, max_m_k_n, max_m_k_n);


	   float *d_A, *d_B, *d_C;
	   checkCuda(cudaMallocManaged(&d_A, max_m_k_n * max_m_k_n * sizeof(float)));
	   checkCuda(cudaMallocManaged(&d_B, max_m_k_n * max_m_k_n * sizeof(float)));
	   checkCuda(cudaMallocManaged(&d_C, max_m_k_n * max_m_k_n * sizeof(float)));

	   checkCuda(cudaMemcpy(d_A,h_A,max_m_k_n * max_m_k_n * sizeof(float),cudaMemcpyHostToDevice));
	   checkCuda(cudaMemcpy(d_B,h_B,max_m_k_n * max_m_k_n * sizeof(float),cudaMemcpyHostToDevice));
	   checkCuda(cudaMemcpy(d_C,h_C,max_m_k_n * max_m_k_n * sizeof(float),cudaMemcpyHostToDevice));

	   int lda, ldb, ldc, m, n, k;
	   const float alf = 1.0f;
	   const float bet = 0.0f;
	   const float *alpha = &alf;
	   const float *beta = &bet;

		cout << "start run gemm on GPUs" << endl;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		for(int size = min_m_k_n; size <= max_m_k_n; size=size*2)
		{
			double sum = 0.0;

			cudaEventRecord(start, 0);
			m=n=k=size;
			lda = m;
			ldb = k;
			ldc = m;

			stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
		
			cudaEventRecord(stop,0);
			cudaEventSynchronize(stop);
			if(stat != CUBLAS_STATUS_SUCCESS)
			{
				cerr << "cublasSgemmBatched failed" << endl;
				exit(1);
			}
			assert(!cudaGetLastError());
		  
			float elapsed;
			cudaEventElapsedTime(&elapsed, start, stop);
			elapsed /= 1000.0f;
			sum += elapsed;
			cout << "float64: size "  << size << " average: " << sum << " s "<< endl;
		}
		
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);

		free(h_A);
		free(h_B);
		free(h_C);  
  }
  else if( data_type.compare("fp64") == 0 ) 
  {
	  double *h_A = (double *)malloc(max_m_k_n * max_m_k_n * sizeof(double));
	  double *h_B = (double *)malloc(max_m_k_n * max_m_k_n * sizeof(double));
	  double *h_C = (double *)malloc(max_m_k_n * max_m_k_n * sizeof(double));
	  
	  CPU_fill_rand_double(h_A, max_m_k_n, max_m_k_n);
	  CPU_fill_rand_double(h_B, max_m_k_n, max_m_k_n);
	  CPU_fill_rand_double(h_C, max_m_k_n, max_m_k_n);

	   double *d_A, *d_B, *d_C;
	   checkCuda(cudaMallocManaged(&d_A, max_m_k_n * max_m_k_n * sizeof(double)));
	   checkCuda(cudaMallocManaged(&d_B, max_m_k_n * max_m_k_n * sizeof(double)));
	   checkCuda(cudaMallocManaged(&d_C, max_m_k_n * max_m_k_n * sizeof(double)));

	   checkCuda(cudaMemcpy(d_A,h_A,max_m_k_n * max_m_k_n * sizeof(double),cudaMemcpyHostToDevice));
	   checkCuda(cudaMemcpy(d_B,h_B,max_m_k_n * max_m_k_n * sizeof(double),cudaMemcpyHostToDevice));
	   checkCuda(cudaMemcpy(d_C,h_C,max_m_k_n * max_m_k_n * sizeof(double),cudaMemcpyHostToDevice));

	   int lda, ldb, ldc, m, n, k;
	   const double alf = 1.0f;
	   const double bet = 0.0f;
	   const double *alpha = &alf;
	   const double *beta = &bet;

		cout << "start run gemm on GPUs" << endl;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		for(int size = min_m_k_n; size <= max_m_k_n; size=size*2)
		{
			double sum = 0.0;

			cudaEventRecord(start, 0);
			m=n=k=size;
			lda = m;
			ldb = k;
			ldc = m;

			stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
		
			cudaEventRecord(stop,0);
			cudaEventSynchronize(stop);
			if(stat != CUBLAS_STATUS_SUCCESS)
			{
				cerr << "cublasSgemmBatched failed" << endl;
				exit(1);
			}
			assert(!cudaGetLastError());
		  
			float elapsed;
			cudaEventElapsedTime(&elapsed, start, stop);
			elapsed /= 1000.0f;
			sum += elapsed;
			cout << "float64: size "  << size << " average: " << sum << " s "<< endl;
		}
		
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);

		free(h_A);
		free(h_B);
		free(h_C);
  }
  
  return 0;
}