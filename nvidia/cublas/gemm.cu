#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;
#define debug 0
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

cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}


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

void print_debug_info(float *hh, int size)
{
	cout << "Debug info:" << endl;
	for (int i=0; i<size; i++) {
		for (int j=0; j<size; j++)
		{
			cout << hh[i*size + j] << " "; 
		}
		cout << endl;
	}
}

int main(int argc, char ** argv)
{
  	int min_m_k_n = 64;
  	int max_m_k_n = 4096*8;
  	int repeats = 1;

    cout << "Running with" 
	 << " min_m_k_n: " << min_m_k_n
	 << " max_m_k_n: " << max_m_k_n
	 << " repeats: " << repeats
	 << endl;

	cublasStatus_t stat;
	cublasHandle_t handle;

	checkCublas(cublasCreate(&handle));

	cudaEvent_t start_all, stop_all;
	cudaEvent_t start_kernel, stop_kernel;
	cudaEvent_t start_memory, stop_memory;
	cudaEventCreate(&start_all);
	cudaEventCreate(&stop_all);
  	cudaEventCreate(&start_kernel);
	cudaEventCreate(&stop_kernel);
	cudaEventCreate(&start_memory);
	cudaEventCreate(&stop_memory);
	
	for(int size = min_m_k_n; size <= max_m_k_n; size=size*2)
	{
		double time_kernel = 0.0;
		double time_memory = 0.0;
		double time_all = 0.0;
		float elapsed;

		cudaEventRecord(start_all, 0);

		/*memory*/
		cout << "start allocating memory on host, matrix size: " << size << endl;
		cudaEventRecord(start_memory, 0);
		float *h_A = (float *)malloc(size * size * sizeof(float));
		float *h_B = (float *)malloc(size * size * sizeof(float));
		float *h_C = (float *)malloc(size * size * sizeof(float));
		CPU_fill_rand(h_A, size, size);
		CPU_fill_rand(h_B, size, size);
		cudaEventRecord(stop_memory, 0);
		cudaEventSynchronize(stop_memory);
		elapsed = 0.0;
		cudaEventElapsedTime(&elapsed, start_memory, stop_memory);
		elapsed /= 1000.0f;
		time_memory = elapsed;
		/*memory end*/

		#if debug
		print_debug_info (h_A, size);
		print_debug_info (h_B, size);
		#endif

		/*kernel*/

		int lda, ldb, ldc, m, n, k;
		const float alf = 1.0f;
		const float bet = 0.0f;
		const float *alpha = &alf;
		const float *beta = &bet;

		m=n=k=size;
		lda = m;
		ldb = k;
		ldc = m;


		cout << "start run cublasSgemm on GPUs, matrix size: " << size << endl;
		cudaEventRecord(start_kernel, 0);
		float *d_A, *d_B, *d_C;
		checkCuda(cudaMallocManaged(&d_A, size * size * sizeof(float)));
		checkCuda(cudaMallocManaged(&d_B, size * size * sizeof(float)));
		checkCuda(cudaMallocManaged(&d_C, size * size * sizeof(float)));
		checkCuda(cudaMemcpy(d_A, h_A, size * size * sizeof(float), cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(d_B, h_B, size * size * sizeof(float), cudaMemcpyHostToDevice));
		stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
		checkCuda(cudaMemcpy(h_C,d_C,size * size * sizeof(float),cudaMemcpyDeviceToHost));			
		cudaEventRecord(stop_kernel,0);
		cudaEventSynchronize(stop_kernel);

		#if debug
		print_debug_info (h_C, size);
		#endif
		if(stat != CUBLAS_STATUS_SUCCESS)
		{
			cerr << "cublasSgemmBatched failed" << endl;
			exit(1);
		}
		assert(!cudaGetLastError());

		elapsed = 0.0;
		cudaEventElapsedTime(&elapsed, start_kernel, stop_kernel);
		elapsed /= 1000.0f;
		time_kernel = elapsed;
		double perf = ((double)(size*size)/time_kernel)/1000000.0f;
		/*kernel end*/

		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);

		free(h_A);
		free(h_B);
		free(h_C); 

		cudaEventRecord(stop_all, 0);
		cudaEventSynchronize(stop_all);
		cudaEventElapsedTime(&elapsed, start_all, stop_all);
		elapsed /= 1000.0f;
		time_all = elapsed;

		cout << "float32, size: "  << size << endl;
		cout <<"kernel: " << time_kernel << " seconds" << endl;
		cout <<"memory: " << time_memory << " seconds" << endl;
		cout <<"all: " << time_all << " seconds" << endl;
		cout << "GFlops:" << perf << endl << endl;		
	}
 
  	return 0;
}
