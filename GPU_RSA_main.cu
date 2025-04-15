#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "support.h"
#include "kernel.cu"

#define THREADS_PER_BLOCK 32 

// data is LP problem in standard form
// min z = cx
// s.t. Ax = b
//      x >= 0

void getMatrixSize(const char* filename, int* rows, int* cols) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file %s\n", filename);
        exit(1);
    }

    char line[1000000];
    int r = 0, c = 0;
    while (fgets(line, sizeof(line), file)) {
        int count = 0;
        char* token = strtok(line, " \t\n");
        while (token != NULL) {
            count++;
            token = strtok(NULL, " \t\n");
        }
        if (count > c) c = count;
        r++;
    }

    fclose(file);
    *rows = r;
    *cols = c;
}

// memory allocation
float** allocateMatrix(int rows, int cols) {
    float** matrix = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++)
        matrix[i] = (float*)malloc(cols * sizeof(float));
    return matrix;
}

void readMatrixA(const char* filename, float** A, int rows, int cols) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fscanf(file, "%f", &A[i][j]) != 1) {
                printf("Error reading A[%d][%d] from file!\n", i, j);
                exit(1);
            }
        }
    }
    fclose(file);
}
// Read file and transpose
void readMatrixT(const char* filename, float* A_T, int rows, int cols) {
    FILE *file = fopen(filename, "r");
    if (!file) {
         printf("Error opening file %s\n", filename);
         exit(1);
    }
    float* A = (float *)malloc(sizeof(float)*rows*cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
             if (fscanf(file, "%f", &A[i*cols+j]) != 1) {
                 printf("Error reading A[%d][%d] from file!\n", i, j);
                 exit(1);
             }
        }
    }
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
             A_T[j*rows+i] = A[i*cols+j];
        }
    }
    fclose(file);
    free(A);
}
// Read file
void readVector(const char* filename, float* vec, int size) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file %s\n", filename);
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%f", &vec[i]) != 1) {
            printf("Error reading vector[%d] from file!\n", i);
            exit(1);
        }
    }
    fclose(file);
}

// find initial basis
int is_identity_column(float** A, int col, int num_row, int* pivot_row) {
    int one_count = 0, zero_count = 0, row = -1;
    for (int i = 0; i < num_row; i++) {
        if (fabs(A[i][col] - 1.0) < 1e-5) {
            one_count++;
            row = i;
        } else if (fabs(A[i][col]) < 1e-5) {
            zero_count++;
        }
    }
    if (one_count == 1 && zero_count == num_row - 1) {
        *pivot_row = row;
        return 1;
    }
    return 0;
}

int assign_variable(int* basic_index, int* non_basic_index, int entering_var, int leaving_var, int N, int M) {
    for (int i = 0; i < N; i++) {
        if (basic_index[i] == leaving_var) {
            basic_index[i] = entering_var;
            break;
        }
    }
    for (int j = 0; j < M; j++) {
        if (non_basic_index[j] == entering_var) {
            non_basic_index[j] = leaving_var;
            break;    
        }
    }
    return 0;
}

int main(){
    Timer timer;
    float transfer_time = 0.0;

    int num_row, num_col;
    // get matrix size
    getMatrixSize("A_matrix.txt", &num_row, &num_col);
    printf("num_row:%d,num_col:%d,", num_row, num_col);

    // Allocate memory 
    float** A = allocateMatrix(num_row, num_col);
    float * A_1d = (float *)malloc(num_row*num_col * sizeof(float));;
    float* b = (float *)malloc(num_row * sizeof(float));
    float* c = (float *)malloc(num_col * sizeof(float));
    float* I = (float *) malloc(num_row*num_row * sizeof(float));

    // Read files & initialize data
    readMatrixA("A_matrix.txt", A, num_row, num_col);
    readMatrixT("A_matrix.txt", A_1d, num_row, num_col);
    readVector("b_vector.txt", b, num_row);
    readVector("c_vector.txt", c, num_col);
    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_row; j++) {
            I[i*num_row+j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // initial basis
    int N = num_row;  // number of basic variables
    int M = num_col - N;  // number of non-basic variables
    int* basic_index = (int*)malloc(num_row * sizeof(int));
    int* non_basic_index = (int*)malloc(num_col * sizeof(int));
    int* row_covered = (int*)calloc(num_row, sizeof(int));

    startTime(&timer);
    int basic_count = 0, nonbasic_count = 0;
    for (int j = 0; j < num_col; j++) {
        int pivot_row;
        if (is_identity_column(A, j, num_row, &pivot_row) && !row_covered[pivot_row]) {
            basic_index[basic_count++] = j;
            row_covered[pivot_row] = 1;
        } else {
            non_basic_index[nonbasic_count++] = j;
        }
    }

    // <device> Allocate device variables
    float* A_T__dev;              cudaMalloc((void**) &A_T__dev, sizeof(float)*num_row*num_col);
    float* b__dev;                cudaMalloc((void**) &b__dev, sizeof(float)*num_row);
    float* c__dev;                cudaMalloc((void**) &c__dev, sizeof(float)*num_col);
    float* B_inverse__dev;        cudaMalloc((void**) &B_inverse__dev, sizeof(float)*num_row*num_row);
    float* Xb__dev;               cudaMalloc((void**) &Xb__dev, sizeof(float)*num_row);
    float* A_bar__dev;            cudaMalloc((void**) &A_bar__dev, sizeof(float)*num_row);
    float* cN_bar__dev;           cudaMalloc((void**) &cN_bar__dev, sizeof(float)*M);
    float* p__dev;                cudaMalloc((void**) &p__dev, sizeof(float)*num_row);
    int*   basic_index__dev;      cudaMalloc((void**) &basic_index__dev, sizeof(int)*num_row);
    int*   non_basic_index__dev;  cudaMalloc((void**) &non_basic_index__dev, sizeof(int)*num_col);
    float* ratio__dev;            cudaMalloc((void**) &ratio__dev, sizeof(float)*N);
    float* E_col__dev;            cudaMalloc((void**) &E_col__dev, sizeof(float)*num_row);
    float* temp_vec_N__dev;       cudaMalloc((void**) &temp_vec_N__dev, sizeof(float)*N);
    float* temp_vec_M__dev;       cudaMalloc((void**) &temp_vec_M__dev, sizeof(float)*M);
    cudaDeviceSynchronize();

    // <device> Copying data to device
    startTime(&timer);
    cudaMemcpy(A_T__dev, A_1d, sizeof(float)*num_row*num_col, cudaMemcpyHostToDevice);
    cudaMemcpy(b__dev, b, sizeof(float)*num_row, cudaMemcpyHostToDevice);
    cudaMemcpy(c__dev, c, sizeof(float)*num_col, cudaMemcpyHostToDevice);
    cudaMemcpy(B_inverse__dev, I, sizeof(float)*num_row*num_row, cudaMemcpyHostToDevice);
    cudaMemcpy(basic_index__dev, basic_index, sizeof(int)*num_row, cudaMemcpyHostToDevice);
    cudaMemcpy(non_basic_index__dev, non_basic_index, sizeof(int)*num_col, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    float* cN_bar = (float *)malloc(M * sizeof(float));
    float* Xb = (float *)malloc(N * sizeof(float));
    float* A_bar = (float *)malloc(num_row * sizeof(float));
    float* ratio = (float *)malloc(N * sizeof(float));

    unsigned int numBlocks = (N - 1)/THREADS_PER_BLOCK + 1;
    dim3 gridDim(numBlocks, 1, 1), blockDim(THREADS_PER_BLOCK, 1, 1);
    unsigned int numBlocks_2 = (M - 1)/THREADS_PER_BLOCK + 1;
    dim3 gridDim_M(numBlocks_2, 1, 1);
    printf("numBlocks:%u,numBlocks_2:%u,", numBlocks, numBlocks_2);

    int iter_num = 0; 
    while (iter_num <= 10000) {
        iter_num++;

        cudaMemcpy(basic_index__dev, basic_index, sizeof(int)*num_row, cudaMemcpyHostToDevice);
        cudaMemcpy(non_basic_index__dev, non_basic_index, sizeof(int)*num_row, cudaMemcpyHostToDevice);
        vecMatrixMulti_cB<<<gridDim, blockDim>>>(basic_index__dev, c__dev, B_inverse__dev, p__dev, N);
        cudaDeviceSynchronize();
        vecMatrixMulti<<<gridDim, blockDim>>>(p__dev, A_T__dev, non_basic_index__dev, temp_vec_M__dev, M, N);
        cudaDeviceSynchronize();
        vecDiff_cN_bar<<<gridDim_M, blockDim>>>(non_basic_index__dev, c__dev, temp_vec_M__dev, cN_bar__dev, M);
        cudaDeviceSynchronize();
        cudaMemcpy(cN_bar, cN_bar__dev, sizeof(float)*M, cudaMemcpyDeviceToHost);

        int entering_index = -1;
        float min_value = -1e-10;
        for (int i = 0; i < M; i++) {
            if (cN_bar[i] < min_value) {
                min_value = cN_bar[i];
                entering_index = i;
            }
        }

        if (entering_index == -1) {
            matrixVecMulti<<<gridDim, blockDim>>>(B_inverse__dev, b__dev, Xb__dev, N);
            cudaDeviceSynchronize();
            cudaMemcpy(Xb, Xb__dev, sizeof(float)*num_row, cudaMemcpyDeviceToHost);

            float z = 0;
            for (int i = 0; i < N; i++)
                z += c[basic_index[i]] * Xb[i];
            printf("iterations:%d,", iter_num);
            printf("optimal_value:%f,", z);
            break;
        }

        int entering_var = non_basic_index[entering_index];
        matrixVecMulti<<<gridDim, blockDim>>>(B_inverse__dev, &(A_T__dev[num_row*entering_var]), A_bar__dev, N);
        matrixVecMulti<<<gridDim, blockDim>>>(B_inverse__dev, b__dev, Xb__dev, N);
        cudaDeviceSynchronize();
        vecDiv<<<gridDim, blockDim>>>(Xb__dev, A_bar__dev, ratio__dev, N);
        cudaMemcpy(A_bar, A_bar__dev, sizeof(float)*num_row, cudaMemcpyDeviceToHost);
        cudaMemcpy(Xb, Xb__dev, sizeof(float)*num_row, cudaMemcpyDeviceToHost);
        cudaMemcpy(ratio, ratio__dev, sizeof(float)*num_row, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        int leaving_index = -1;
        float min_val = INFINITY;
        for (int i = 0; i < num_row; i++) {
            if ((A_bar[i] > 1e-8) && (ratio[i] < min_val)) {
                min_val = ratio[i];
                leaving_index = i;
            }
        }
        if (leaving_index == -1) {
            printf("Unbounded LP.\n");
            return 1;
        }
        int leaving_var = basic_index[leaving_index];
        assign_variable(basic_index, non_basic_index, entering_var, leaving_var, N, M);

        cudaMemcpy(E_col__dev, A_bar__dev, sizeof(float)*num_row, cudaMemcpyDeviceToDevice);
        vecMultiConst<<<gridDim, blockDim>>>(E_col__dev, -1.0f/A_bar[leaving_index], N);
        B_inverse_col<<<gridDim, blockDim>>>(temp_vec_N__dev, B_inverse__dev, leaving_index, N);
        cudaDeviceSynchronize();
        float E_col_leaving = 1.0f / A_bar[leaving_index] - 1.0f;
        cudaMemcpy(&(E_col__dev[leaving_index]), &E_col_leaving, sizeof(float), cudaMemcpyHostToDevice);
        dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
        dim3 blocks(numBlocks, numBlocks);
        updateB_inverse<<<blocks, threads>>>(B_inverse__dev, E_col__dev, temp_vec_N__dev, N);
        cudaDeviceSynchronize();
    }
    stopTime(&timer);
    printf("total_time:%fs\n", elapsedTime(timer));

    free(cN_bar);
    free(Xb);
    free(A_bar);

    for (int i = 0; i < num_row; i++) {
        free(A[i]);
    }
    free(A);
    free(b);
    free(c);
    free(non_basic_index);
    free(basic_index);
    return 0;
}
