
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#define DIM 4
#define BLOCKDIM 4

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

void LUDecomposition(float A[DIM][DIM], float *L){
	float mik = 0;

	for (int i = 0; i < DIM - 1; i++){
		for (int j = i + 1; j < DIM; j++){
			mik = A[j][i] / A[i][i];

			L[i * DIM + j] = mik;

			for (int k = i + 1; k < DIM; k++){
				A[j][k] -= mik * A[i][k];
			}

			A[j][i] = 0;
		}
	}
}

float matrixDet(float m[DIM][DIM], int car) {
	float determinante = 0;
	//Cardinalità uno
	if (car == 1) determinante = m[0][0];
	//Cardinalità due
	if (car == 2)
		determinante = m[1][1] * m[0][0] - m[0][1] * m[1][0];
	//Cardinalità > 2
	else {
		for (int row = 0; row < car; row++) {
			float sub_m[DIM][DIM];
			//Sottomatrice di ordine car-1
			for (int i = 0; i < car - 1; i++) {
				for (int j = 0; j < car - 1; j++) {
					int sub_row = (i < row ? i : i + 1);
					int sub_col = j + 1;
					sub_m[i][j] = m[sub_row][sub_col];
				}
			}
			//Segno sottomatrice + per pari, - per dispari
			if (row % 2 == 0)
				determinante += m[row][0] * matrixDet(sub_m, car - 1);
			else
				determinante -= m[row][0] * matrixDet(sub_m, car - 1);
		}
	}
	return determinante;
}

float vectorProduct(float a[DIM], float b[DIM]){
	float result = 0;

	for (int i = 0; i < DIM; i++){
		result += a[i] * b[i];
	}

	return result;
}

void triangolarUpperMatrix(float M[DIM][DIM], float *U){
	int k = 0;

	for (int i = 0; i < DIM; i++){
		for (int j = k; j < DIM; j++){
			U[i * DIM + j] = M[i][j];
		}
		k++;
	}
}

void setupLowerMatrix(float *L){

	for (int i = 0; i < DIM; i++){
		for (int j = i + 1; j < DIM; j++){
			L[i * DIM + j] = 0;
		}
	}
}

void printMatrixCPU(float *M){
	for (int i = 0; i < DIM; i++){
		for (int j = 0; j < DIM; j++){
			printf("[%d][%d] = %8f ", i, j, M[i * DIM + j]);
		}
		printf("\n");
	}
}

__global__ void printMatrixGPU(float *M, int nRow, int nCol){
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	
	int temp = M[ix + iy * nCol];

	if (ix < nRow && iy < nCol){
		printf("dev_M[%d][%d] = %d\N", ix, iy, temp);
	}
}

/*Matrice esempio, su cui è verificata la correttezza dell'algoritmo
float host_A[DIM][DIM] = { { 1, 3, 4, 5 }, { 2, 1, 1, 0 }, { 2, 3, 1, -1 }, { 0, 4, 3, 2} };

Matrici risultato
L = {{1 0 0 0}, {2 1 0 0}, {2 0.6 1 0}, {0 -0.8 0.928571 1}}
U = {{1 3 4 5}, {0 -5 -7 -10}, {0 0 -2.8 -5}, {0 0 0 -1.357142}}*/

int main(){
	//DICHIARAZIONE VARIABILI

	//float *host_A[DIM][DIM];
	float *host_A, *host_L, *host_U;

	float *dev_A, *dev_L, *dev_U;
	int size = sizeof(float) * DIM * DIM;

	dim3 blockSize(BLOCKDIM, BLOCKDIM);

	//INIZIALIZZAZIONE VARIABILI CPU
	
	host_A = (float *)malloc(size);
	host_L = (float *)malloc(size);
	host_U = (float *)malloc(size);

	for (int i = 0; i < DIM; i++){
		for (int j = 0; j < DIM; j++){
			host_A[i * DIM + j] = rand() % 11;

			if (i == j)
				host_L[i * DIM + j] = 1;
			else
				host_L[i * DIM + j] = 0;

			host_U[i * DIM + j] = 0;
		}
	}

	//ALLOCAZIONE ED INIZIALIZZAZIONE VARIABILI GPU

	gpuErrorCheck(cudaMalloc((void **)&dev_A, size));
	gpuErrorCheck(cudaMalloc((void **)&dev_L, size));
	gpuErrorCheck(cudaMalloc((void **)&dev_U, size));

	gpuErrorCheck(cudaMemcpy(dev_A, host_A, size, cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(dev_L, host_L, size, cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(dev_U, host_U, size, cudaMemcpyHostToDevice));

	//STAMPA MATRICE DA DECOMPORRE

	printf("|____MATRICE A CPU____|\n");
	printMatrixCPU(host_A);
	printf("\n|____MATRICE A GPU____|\n");
	printMatrixGPU << <1, blockSize >> >(dev_A, DIM, DIM);
	
	gpuErrorCheck(cudaPeekAtLastError());
	gpuErrorCheck(cudaDeviceSynchronize());

	//CALCOLO LA DECOMPOSIZIONE

	//LUDecomposition(host_A, host_L);

	//STAMPO LE MATRICI L ED U

	printf("\n|____FATTORIZZAZIONE LU____|\n");

	setupLowerMatrix(host_L);
	printf("\n|____MATRICE L CPU____|\n");
	printMatrixCPU(host_L);

	/*printf("\n|____MATRICE L GPU____|\n");
	printMatrixGPU << <1, blockSize >> >(dev_L);

	gpuErrorCheck(cudaDeviceSynchronize());

	triangolarUpperMatrix(host_A, host_U);
	printf("\n|____MATRICE U____|\n");
	printMatrixCPU(host_U);*/
	
	cudaDeviceReset();
}
