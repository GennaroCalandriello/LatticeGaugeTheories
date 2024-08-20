#define BLOCK_SIZE 32
#define GRID_SIZE 32
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Kernel CUDA per l'algoritmo Ising
__global__ void cudaKernel(int n, double* gpuWeights, int* gpuG, int* gpuTempGrid, int *flag)
{
	// Coordinate del momento nel reticolo
	int momentCol = blockIdx.x * blockDim.x + threadIdx.x;
	int momentRow = blockIdx.y * blockDim.y + threadIdx.y;

	// Memoria condivisa allocata per i pesi
	__shared__ double sharedWeights[25];
	// Memoria condivisa allocata per un blocco di momenti
	int sharedSize = (BLOCK_SIZE+4);
	__shared__ int sharedG[(BLOCK_SIZE+4)*(BLOCK_SIZE+4)];

	// Coordinate del momento nella memoria condivisa
	int sharedRow = threadIdx.y + 2;
	int sharedCol = threadIdx.x + 2;

	int idxRow, idxCol;
	double weightFactor = 0.0;

	// Memorizza i pesi in memoria condivisa
	if(threadIdx.x < 5 && threadIdx.y < 5)
		sharedWeights[threadIdx.x * 5 + threadIdx.y] = gpuWeights[threadIdx.x * 5 + threadIdx.y];

	// Ciclo per trasferire i momenti e i vicini necessari dalla memoria globale a quella condivisa
	for(int i = momentRow; i < n + 2; i += blockDim.y * gridDim.y)
	{
		for(int j = momentCol; j < n + 2; j += blockDim.x * gridDim.x)
		{
			sharedG[sharedRow * sharedSize + sharedCol] = gpuG[((i + n) % n) * n + ((j + n) % n)];

			if(threadIdx.x < 2)
			{
				// Confini a sinistra
				idxRow = (i + n) % n;
				idxCol = (-2 + j + n) % n;
				sharedG[sharedRow * sharedSize + sharedCol - 2] = gpuG[n * idxRow + idxCol];

				// Confini a destra
				idxCol = (BLOCK_SIZE + j + n) % n;
				sharedG[sharedRow * sharedSize + sharedCol + BLOCK_SIZE] = gpuG[n * idxRow + idxCol];

				if(threadIdx.y < 2)
				{
					// Angoli
					idxRow = (-2 + i + n) % n;
					idxCol = (-2 + j + n) % n;
					sharedG[(sharedRow - 2) * sharedSize + sharedCol - 2] = gpuG[n * idxRow + idxCol];
					
					idxRow = (i + n + BLOCK_SIZE) % n;
					idxCol = (-2 + j + n) % n;
					sharedG[(sharedRow + BLOCK_SIZE) * sharedSize + sharedCol - 2] = gpuG[n * idxRow + idxCol];
					
					idxRow = (-2 + i + n) % n;
					idxCol = (j + n + BLOCK_SIZE) % n;
					sharedG[(sharedRow - 2) * sharedSize + sharedCol + BLOCK_SIZE] = gpuG[n * idxRow + idxCol];
					
					idxRow = (i + n + BLOCK_SIZE) % n;
					idxCol = (j + n + BLOCK_SIZE) % n;
					sharedG[(sharedRow + BLOCK_SIZE) * sharedSize + sharedCol + BLOCK_SIZE] = gpuG[n * idxRow + idxCol];
				}
			}

			if(threadIdx.y < 2)
			{
				// Confini superiori e inferiori
				idxRow = (-2 + i + n) % n;
				idxCol = (j + n) % n;
				sharedG[(sharedRow - 2) * sharedSize + sharedCol] = gpuG[n * idxRow + idxCol];

				idxRow = (i + n + BLOCK_SIZE) % n;
				sharedG[(sharedRow + BLOCK_SIZE) * sharedSize + sharedCol] = gpuG[n * idxRow + idxCol];
			}

			__syncthreads();

			if(i < n && j < n)
			{
				weightFactor = 0.0;
				for(int row = 0; row < 5; row++)
				{
					for(int col = 0; col < 5; col++)
					{
						if(col == 2 && row == 2)
							continue;

						weightFactor += sharedG[(sharedRow - 2 + row) * sharedSize + sharedCol - 2 + col] * sharedWeights[row * 5 + col];
					}
				}

				if(weightFactor < 0.0001 && weightFactor > -0.0001)
				{
					gpuTempGrid[n * i + j] = sharedG[sharedRow * sharedSize + sharedCol];
				}
				else if(weightFactor > 0.00001)
				{
					gpuTempGrid[n * i + j] = 1;
					if (gpuG[n * i + j] == -1)
					{
						*flag = 1;	
					}	
				}
				else
				{
					gpuTempGrid[n * i + j] = -1;
					if (gpuG[n * i + j] == -1)
					{
						*flag = 1;	
					}
				}
			}

			__syncthreads();
		}
	}
}

// Funzione per scrivere il reticolo su file
void write_lattice(int* lattice, int n, const char* filename)
{
	FILE* file = fopen(filename, "w");
	if (file == NULL)
	{
		printf("Errore nell'apertura del file %s\n", filename);
		return;
	}

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			fprintf(file, "%d ", lattice[i * n + j]);
		}
		fprintf(file, "\n");
	}

	fclose(file);
	printf("Lattice scritto su %s\n", filename);
}

int main()
{
	// Parametri di input
	int n = 1024; // Dimensione del reticolo
	int k = 1000; // Numero di iterazioni
	double weights[25] = { 0.004, 0.016, 0.026, 0.016, 0.004,
	                       0.016, 0.071, 0.117, 0.071, 0.016,
	                       0.026, 0.117, 0.0, 0.117, 0.026,
	                       0.016, 0.071, 0.117, 0.071, 0.016,
	                       0.004, 0.016, 0.026, 0.016, 0.004 };
	int *G = (int*)malloc(n * n * sizeof(int));

	// Inizializzazione del reticolo con valori casuali -1 o 1
	for(int i = 0; i < n * n; i++)
	{
		G[i] = (rand() % 2) * 2 - 1;
	}

	// Array per memorizzare i pesi nella memoria GPU
	double *gpuWeights;
	// Array per memorizzare il reticolo nella memoria GPU
	int *gpuG;
	// Array per l'aggiornamento del reticolo nella memoria GPU
	int *gpuTempGrid;
	// Variabile per fermare l'aggiornamento se il reticolo rimane invariato
	int flag;
	int *gpuFlag;

	// Allocazione della memoria sulla GPU
	cudaMalloc(&gpuFlag, sizeof(int));
	cudaMalloc(&gpuWeights, 25 * sizeof(double));
	cudaMemcpy(gpuWeights, weights, 25 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc(&gpuG, n * n * sizeof(int));
	cudaMemcpy(gpuG, G, n * n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc(&gpuTempGrid, n * n * sizeof(int));

	// Configurazione delle dimensioni di Grid e Block
	dim3 dimGrid(GRID_SIZE, GRID_SIZE);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	// Loop di aggiornamento
	for(int i = 0; i < k; i++)
	{
		flag = 0;
		cudaMemcpy(gpuFlag, &flag, sizeof(int), cudaMemcpyHostToDevice);
		cudaKernel<<<dimGrid, dimBlock>>>(n, gpuWeights, gpuG, gpuTempGrid, gpuFlag);
		cudaDeviceSynchronize();

		// Scambio dei puntatori per evitare il trasferimento di dati ad ogni iterazione
		int *gpuTempPtr = gpuG;
		gpuG = gpuTempGrid;
		gpuTempGrid = gpuTempPtr;

		// Controllo se il reticolo è rimasto invariato
		cudaMemcpy(&flag, gpuFlag, sizeof(int), cudaMemcpyDeviceToHost);
		if(flag == 0)
			break;
	}

	// Copia finale del reticolo dalla GPU alla CPU
	cudaMemcpy(G, gpuG, n * n * sizeof(int), cudaMemcpyDeviceToHost);

	// Scrittura del reticolo su file
	write_lattice(G, n, "final.txt");

	// Liberazione della memoria
	cudaFree(gpuG);
	cudaFree(gpuTempGrid);
	cudaFree(gpuWeights);
	cudaFree(gpuFlag);
	free(G);

	return 0;
}
