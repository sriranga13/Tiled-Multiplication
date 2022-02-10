/* ACADEMIC INTEGRITY PLEDGE                                              */
/*                                                                        */
/* - I have not used source code obtained from another student nor        */
/*   any other unauthorized source, either modified or unmodified.        */
/*                                                                        */
/* - All source code and documentation used in my program is either       */
/*   my original work or was derived by me from the source code           */
/*   published in the textbook for this course or presented in            */
/*   class.                                                               */
/*                                                                        */
/* - I have not discussed coding details about this project with          */
/*   anyone other than my instructor. I understand that I may discuss     */
/*   the concepts of this program with other students and that another    */
/*   student may help me debug my program so long as neither of us        */
/*   writes anything during the discussion or modifies any computer       */
/*   file during the discussion.                                          */
/*                                                                        */
/* - I have violated neither the spirit nor letter of these restrictions. */
/*                                                                        */
/*                                                                        */
/*                                                                        */
/* Signed:Sriranga   Date: 3/8/2021          */
/*                                                                        */
/*                                                                        */
/* 3460:677 CUDA Tiled Matrix Multiplication lab, V. 1.01, Fall 2016.     */

#include <stdio.h>
#include <stdlib.h>
#include "helper_timer.h"
#include "exception.h"
#define width 16
#define BLOCK_SIZE 16

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  __shared__ float A_tile[width][width];
  __shared__ float B_tile[width][width];
  
  int t1 = threadIdx.x;
  int t2 = threadIdx.y;
  int b1 = blockIdx.x ;
  int b2 = blockIdx.y;
  int row = b2 * blockDim.y + t2 ;
  int col = b1 * blockDim.x + t1 ; 
 
  float CValue = 0;
  for(int p = 0 ; p < (numARows -1)/width + 1; ++p) {
     A_tile[t2][t1] = A[row*numARows + p*width + t1];
     B_tile[t2][t1] = B[(p*width+t2)*numARows + col];
    __syncthreads();

     for(int i = 0 ; i < width ; ++i){
       CValue += A_tile[t2][i] * B_tile[i][t1];
     }
  __syncthreads();
  }
  C[row*numARows+col] = CValue;
}

int main(int argc, char **argv) {
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *expectedOutput;
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)
  int numEORows, numEOColumns;
				   
  FILE *infile1, *infile2, *outfile;
  StopWatchLinux stw;
  unsigned int  blog = 1;

  // Import host input data
  stw.start();
  if ((infile1 = fopen("input0.raw", "r")) == NULL)
  { printf("Cannot open input0.raw.\n"); exit(EXIT_FAILURE); }
  if ((infile2 = fopen("input1.raw", "r")) == NULL)
  { printf("Cannot open input1.raw.\n"); exit(EXIT_FAILURE); }
  fscanf(infile1, "%i", &numARows);
  fscanf(infile1, "%i", &numAColumns);
  hostA = (float *)malloc(sizeof(float) * numARows * numAColumns);  
  for (int i = 0; i < numARows; i++)
    for (int j = 0; j < numAColumns; j++)
      fscanf(infile1, "%f", &hostA[i * numAColumns + j]);	
  fscanf(infile2, "%i", &numBRows);
  fscanf(infile2, "%i", &numBColumns);
  hostB = (float *)malloc(sizeof(float) * numBRows * numBColumns);  
  for (int i = 0; i < numBRows; i++)
    for (int j = 0; j < numBColumns; j++)
      fscanf(infile2, "%f", &hostB[i * numBColumns + j]);	
  fclose(infile1);
  fclose(infile2);

  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;  
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(sizeof(float) * numCRows * numCColumns);
  
  stw.stop();
  printf("Importing data and creating memory on host: %f ms\n", stw.getTime());

  if (blog) printf("*** The dimensions of A are %i x %i\n", numARows, numAColumns);
  if (blog) printf("*** The dimensions of B are %i x %i\n", numBRows, numBColumns);
  if (blog) printf("*** The dimensions of C are %i x %i\n", numCRows, numCColumns);

  stw.reset();
  stw.start();

  //@@ Allocate GPU memory here
  cudaMalloc((void **) &deviceA, sizeof(float) * numARows * numAColumns);
  cudaMalloc((void **) &deviceB, sizeof(float) * numBRows * numBColumns);
  cudaMalloc((void **) &deviceC, sizeof(float) * numCRows * numCColumns);
  stw.stop();
  printf("Allocating GPU memory: %f ms\n", stw.getTime());

  stw.reset();
  stw.start();

  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA,hostA,sizeof(float) * numARows * numAColumns, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB,hostB,sizeof(float) * numBRows * numBColumns, cudaMemcpyHostToDevice);
  stw.stop();
  printf("Copying input memory to the GPU: %f ms\n", stw.getTime());

  //@@ Initialize the grid and block dimensions here
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 dimGrid((numAColumns - 1) / BLOCK_SIZE + 1, (numAColumns - 1) / BLOCK_SIZE + 1 , 1);  

  
  if (blog) printf("*** The block dimensions are %i x %i\n",dimBlock.x, dimBlock.y);
  if (blog) printf("*** The grid dimensions are %i x %i\n", dimGrid.x, dimGrid.y);

  stw.reset();
  stw.start();

  //@@ Launch the GPU Kernel here
  matrixMultiply<<<dimGrid,dimBlock>>>(deviceA,deviceB,deviceC,numARows,numAColumns,numBRows,numBColumns,numCRows,numCColumns);
  cudaDeviceSynchronize();
  stw.stop();
  printf("Performing CUDA computation: %f ms\n", stw.getTime());

  stw.reset();
  stw.start();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC,deviceC,sizeof(float) * numCRows * numCColumns,cudaMemcpyDeviceToHost);
  stw.stop();
  printf("Copying output memory to the CPU: %f ms\n", stw.getTime());

  stw.reset();
  stw.start();

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  stw.stop();
  printf("Freeing GPU Memory: %f ms\n", stw.getTime());

  if ((outfile = fopen("output.raw", "r")) == NULL)
  { printf("Cannot open output.raw.\n"); exit(EXIT_FAILURE); }
  fscanf(outfile, "%i", &numEORows);
  fscanf(outfile, "%i", &numEOColumns);
  expectedOutput = (float *)malloc(sizeof(float) * numEORows * numEOColumns);  
  for (int i = 0; i < numEORows; i++)
    for (int j = 0; j < numEOColumns; j++)
      fscanf(outfile, "%f", &expectedOutput[i * numEOColumns + j]);	
  fclose(outfile);
  int test = 1;
  for (int i = 0; i < numEORows; i++)
    for (int j = 0; j < numEOColumns; j++) {
      test = test && (abs(expectedOutput[i * numEOColumns + j] - hostC[i * numCColumns + j]) < 0.005);
  }
  if (test) printf("Results correct.\n");
  else printf("Results incorrect.\n");

  free(hostA);
  free(hostB);
  free(hostC);
  free(expectedOutput);

  return 0;
}
