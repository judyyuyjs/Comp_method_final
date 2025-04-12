#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "matrix_multi.h"
#include "support.h"
#include "kernel.cu"

#define DEBUG 0 
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
    for (int i = 0; i < rows; i++) {
        matrix[i] = (float*)malloc(cols * sizeof(float));
    }
    return matrix;
}

// Read file
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

// if LU solver fails, use this code to check the matrix
void dumpMatrix(const char* filename, float** A, int size) {
    FILE* f = fopen(filename, "w");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fprintf(f, "%.12f ", A[i][j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

// Phase 1 
void addArtificialVariables(float*** A_ptr, float** c_ptr, int* num_col_ptr, int num_row, int* basic_index, int* artificial_start_col) {
    float** A = *A_ptr;
    float* c = *c_ptr;
    int original_cols = *num_col_ptr;
    int new_cols = original_cols + num_row;

    // Reallocate each row to add new columns for artificial variables
    for (int i = 0; i < num_row; i++) {
        float* new_row = (float *) realloc(A[i], new_cols * sizeof(float));
        if (!new_row) {
            printf("Failed to realloc A[%d]\n", i);
            exit(1);
        }
        A[i] = new_row;

        // Initialize new columns to 0
        for (int j = original_cols; j < new_cols; j++) {
            A[i][j] = 0.0;
        }

        // Set identity matrix column for artificial variable
        A[i][original_cols + i] = 1.0;
    }

    // Update matrix pointer
    *A_ptr = A;

    // Reallocate cost vector c
    float* new_c = (float *) realloc(c, new_cols * sizeof(float));
    if (!new_c) {
        printf("Failed to realloc c vector\n");
        exit(1);
    }
    *c_ptr = new_c;

    // Set original part of c to 0, artificial variables to 1
    for (int j = 0; j < original_cols; j++) {
        new_c[j] = 0.0;
    }
    for (int j = original_cols; j < new_cols; j++) {
        new_c[j] = 1.0;
    }

    // Set basic indices to artificial variable indices
    for (int i = 0; i < num_row; i++) {
        basic_index[i] = original_cols + i;
    }

    *artificial_start_col = original_cols;
    *num_col_ptr = new_cols;
    printf("[DEBUG] Successfully added artificial variables. A now has %d columns.\n", *num_col_ptr);
    fflush(stdout);
}


// romove artificial variable after phase 1
void removeArtificialVariables(float*** A_ptr, float** c_ptr, int* num_col_ptr, int artificial_start_col, int num_row) {
    float** A = *A_ptr;
    float* c = *c_ptr;
    int new_cols = artificial_start_col;

    for (int i = 0; i < num_row; i++) {
        A[i] = (float *) realloc(A[i], new_cols * sizeof(float));
    }
    *A_ptr = (float **) realloc(A, num_row * sizeof(float*));
    *c_ptr = (float *) realloc(c, new_cols * sizeof(float));
    *num_col_ptr = new_cols;
}

// Use basic index to find Ab, An
void find_An(float** A, int basis[], float **Aj, int N, int M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            Aj[i][j] = A[i][basis[j]];
        }
    }
}
void find_AJ(float** A, int basis[], float **Aj, int N) {
    find_An(A, basis, Aj, N, N);
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

int count_unique(int* arr, int size) {
    int count = 0;
    for (int i = 0; i < size; i++) {
        int duplicate = 0;
        for (int j = 0; j < i; j++) {
            if (arr[i] == arr[j]) {
                duplicate = 1;
                break;
            }
        }
        if (!duplicate) count++;
    }
    return count;
}

// Try to replace a problematic basic column (row_idx) with a suitable non-basic one
int replace_bad_basic_column(float** A, int* basic_index, int* non_basic_index, int N, int M, int row_idx, int num_row, int num_col) {
    for (int j = 0; j < M; j++) {
        int col = non_basic_index[j];

        int one_count = 0, zero_count = 0;
        for (int i = 0; i < num_row; i++) {
            if (fabs(A[i][col] - 1.0) < 1e-6) one_count++;
            else if (fabs(A[i][col]) < 1e-6) zero_count++;
        }

        if (one_count == 1 && zero_count == num_row - 1 && fabs(A[row_idx][col] - 1.0) < 1e-6) {
            int old_basic = basic_index[row_idx];
            basic_index[row_idx] = col;
            non_basic_index[j] = old_basic;
            printf("[INFO] Replaced basic col %d at row %d with col %d\n", old_basic, row_idx, col);
            return 1; // success
        }
    }
    return 0; // failed
}

int main(){
    Timer timer, timer2;
    float temp_time = 0.0;
    cudaError_t cuda_ret;

    int num_row, num_col;
    // get matrix size
    getMatrixSize("A_matrix.txt", &num_row, &num_col);

    // Allocate memory 
    float** A = allocateMatrix(num_row, num_col);
    float* b = (float *)malloc(num_row * sizeof(float));
    float* c = (float *)malloc(num_col * sizeof(float));
    float* original_c = (float *)malloc(num_col * sizeof(float));
    float* I = (float *) malloc(num_row*num_row * sizeof(float));

    // Read files & initialize data
    readMatrixA("A_matrix.txt", A, num_row, num_col);
    readVector("b_vector.txt", b, num_row);
    readVector("c_vector.txt", c, num_col);
    memcpy(original_c, c, num_col * sizeof(float));
    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_row; j++) {
	    I[i*num_row+j] = (i == j) ? 1.0f : 0.0f;
	}
    }

    float * A_1d = (float *)malloc(num_row*num_col * sizeof(float));;
    readMatrixT("A_matrix.txt", A_1d, num_row, num_col);
#if DEBUG
    for (int i = 0; i < num_col; i++) {
        for (int j = 0; j < num_row; j++) {
	    printf("%g ", A_1d[i*num_row+j]);
	}
	printf("\n");
    }
#endif

    printf("num_row: %d\nnum_col: %d\n", num_row, num_col);
    // <device> Allocate device variables
    startTime(&timer);
    float* A_T__dev;
    cuda_ret = cudaMalloc((void**) &A_T__dev, sizeof(float)*num_row*num_col);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory (A_T__dev)");
    float* b__dev;
    cuda_ret = cudaMalloc((void**) &b__dev, sizeof(float)*num_row);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory (b__dev)");
    float* c__dev;
    cuda_ret = cudaMalloc((void**) &c__dev, sizeof(float)*num_col);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory (c__dev)");
    float* B_inverse__dev;
    cuda_ret = cudaMalloc((void**) &B_inverse__dev, sizeof(float)*num_row*num_row);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory (B_inverse__dev)");
    float* Xb__dev;
    cuda_ret = cudaMalloc((void**) &Xb__dev, sizeof(float)*num_row);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory (Xb__dev)");
    float* A_bar__dev;
    cuda_ret = cudaMalloc((void**) &A_bar__dev, sizeof(float)*num_row);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory (A_bar__dev)");
    float* p__dev;
    cuda_ret = cudaMalloc((void**) &p__dev, sizeof(float)*num_row);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory (p__dev)");
    int* basic_index__dev;
    cuda_ret = cudaMalloc((void**) &basic_index__dev, sizeof(int)*num_row);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory (basic_index__dev)");
    int* non_basic_index__dev;
    cuda_ret = cudaMalloc((void**) &non_basic_index__dev, sizeof(int)*num_col);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory (non_basic_index__dev)");
    float* E__dev;
    cuda_ret = cudaMalloc((void**) &E__dev, sizeof(float)*num_row*num_row);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory (E__dev)");
    float* B_inverse_new__dev;
    cuda_ret = cudaMalloc((void**) &B_inverse_new__dev, sizeof(float)*num_row*num_row);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory (B_inverse_new__dev)");
    cudaDeviceSynchronize();
    stopTime(&timer);
    printf("Allocate device variables: %f s\n", elapsedTime(timer));

    // <device> Copying data to device
    startTime(&timer);
    cuda_ret = cudaMemcpy(A_T__dev, A_1d, sizeof(float)*num_row*num_col, cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device (A)");
    cuda_ret = cudaMemcpy(b__dev, b, sizeof(float)*num_row, cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device (b)");
    cuda_ret = cudaMemcpy(c__dev, c, sizeof(float)*num_col, cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device (c)");
    cuda_ret = cudaMemcpy(B_inverse__dev, I, sizeof(float)*num_row*num_row, cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device (B_inverse)");
    cudaDeviceSynchronize();
    stopTime(&timer);
    printf("Copying data to device: %f s\n", elapsedTime(timer));

    startTime(&timer);
    // initial basis
    int N = num_row;  // number of basic variables
    int M = num_col - N;  // number of non-basic variables

    int* basic_index = (int*)malloc(num_row * sizeof(int));
    int* non_basic_index = (int*)malloc(num_col * sizeof(int));
    int* row_covered = (int*)calloc(num_row, sizeof(int));

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

    int artificial_start_col = -1;
    if (basic_count < num_row) {
        printf("[INFO] Adding artificial variables (Phase I) since initial basis is not full rank...\n");
        addArtificialVariables(&A, &c, &num_col, num_row, basic_index, &artificial_start_col);

        M = num_col - N;
        int* new_non_basic_index = (int*)malloc(M * sizeof(int));
        if (!new_non_basic_index) {
            printf("Failed to allocate new_non_basic_index\n");
            exit(1);
        }

        nonbasic_count = 0;
        for (int i = 0; i < num_col; i++) {
            int found = 0;
            for (int j = 0; j < N; j++) {
                if (basic_index[j] == i) {
                    found = 1;
                    break;
                }
            }
            if (!found) new_non_basic_index[nonbasic_count++] = i;
        }

        free(non_basic_index); 
        non_basic_index = new_non_basic_index;
    }

    // <device>
    cuda_ret = cudaMemcpy(basic_index__dev, basic_index, sizeof(int)*num_row, cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device (basic_index)");
    cuda_ret = cudaMemcpy(non_basic_index__dev, non_basic_index, sizeof(int)*num_col, cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device (non_basic_index)");
    cudaDeviceSynchronize();

    int phase = 1;
    float* p = (float *)malloc(N * sizeof(float));
    float* cN_bar = (float *)malloc(M * sizeof(float));
    float* Xb = (float *)malloc(N * sizeof(float));
    float* A_bar = (float *)malloc(num_row * sizeof(float));
    float *E = (float *) malloc(sizeof(float)*N*N);

    unsigned int numBlocks = (N - 1)/THREADS_PER_BLOCK + 1;
    dim3 gridDim(numBlocks, 1, 1), blockDim(THREADS_PER_BLOCK, 1, 1);
    printf("numBlocks: %u\n", numBlocks);
    fflush(stdout);
    while (phase <= 2){
#if DEBUG
        printf("[DEBUG] Top of phase %d loop\n", phase);
#endif
        int iter_num = 0; 
        while (iter_num <= 10000) {
#if DEBUG
            printf("%d. ", iter_num);
#endif
            iter_num++;

            float** Ab = allocateMatrix(N, N);
            float** An = allocateMatrix(N, M);
            find_AJ(A, basic_index, Ab, N);
            find_An(A, non_basic_index, An, N, M);

            cuda_ret = cudaMemcpy(basic_index__dev, basic_index, sizeof(int)*num_row, cudaMemcpyHostToDevice);
	    vecMatrixMulti_cB<<<gridDim, blockDim>>>(basic_index__dev, c__dev, B_inverse__dev, p__dev, N);
	    cudaDeviceSynchronize();
	    cuda_ret = cudaMemcpy(p, p__dev, sizeof(float)*num_row, cudaMemcpyDeviceToHost);

            for (int i = 0; i < M; i++) {
                float sum = 0;
                for (int j = 0; j < N; j++) {
                    sum += p[j] * An[j][i];
                }
                cN_bar[i] = c[non_basic_index[i]] - sum;
            }
#if DEBUG
            printf("Reduced costs (cN_bar): ");
            for (int i = 0; i < 10; i++) {
                printf("%.6f ", cN_bar[i]);
            }
            printf("\n");
#endif
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
                cuda_ret = cudaMemcpy(Xb, Xb__dev, sizeof(float)*num_row, cudaMemcpyDeviceToHost);

                if (phase == 1 && artificial_start_col != -1) {
                    float phase1_obj = 0;
                    for (int i = 0; i < N; i++) {
                        if (basic_index[i] >= artificial_start_col) {
                            phase1_obj += b[i];
                        }
                    }

                    printf("[DEBUG] Phase 1 Objective: %.6f\n", phase1_obj);

                    if (fabs(phase1_obj) > 1e-6) {
                        printf("Infeasible LP.\n");
                        return 1;
                    }

                    // Phase 2 cleanup and reinitialization
                    removeArtificialVariables(&A, &c, &num_col, artificial_start_col, num_row);

                    // Restore original objective coefficients
                    memcpy(c, original_c, num_col * sizeof(float));

                    // === Clean basic_index: remove any out-of-bounds entries ===
                    int* new_basic_index = (int*)malloc(N * sizeof(int));
                    int valid_basic_count = 0;
                    for (int i = 0; i < N; i++) {
                        if (basic_index[i] >= 0 && basic_index[i] < num_col) {
                            new_basic_index[valid_basic_count++] = basic_index[i];
                        } else {
                            printf("[WARNING] Removed invalid basic_index[%d] = %d (num_col = %d)\n", i, basic_index[i], num_col);
                        }
                    }
                    free(basic_index);
                    basic_index = new_basic_index;
                    N = valid_basic_count;

                    // Update number of non-basic variables
                    int unique_N = count_unique(basic_index, N);
                    M = num_col - unique_N;

                    // Rebuild non-basic index array
                    int* new_non_basic_index = (int*)malloc(M * sizeof(int));
                    if (!new_non_basic_index) {
                        printf("Failed to allocate new_non_basic_index for Phase 2\n");
                        exit(1);
                    }

                    int count = 0;
                    for (int i = 0; i < num_col; i++) {
                        int is_basic = 0;
                        for (int j = 0; j < N; j++) {
                            if (basic_index[j] == i) {
                                is_basic = 1;
                                break;
                            }
                        }
                        if (!is_basic) {
                            new_non_basic_index[count++] = i;
                        }
                    }        

                    if (count != M) {
                        printf("[ERROR] non_basic_index mismatch in Phase 2: count=%d, M=%d\n", count, M);
                        printf("[DEBUG] Phase 2 Setup Completed. N = %d, M = %d, num_col = %d\n", N, M, num_col);
                        exit(1);
                    }

                    free(non_basic_index);
                    non_basic_index = new_non_basic_index;

                    // DEBUG INFO
                    printf("[DEBUG] Phase 2 Setup Completed. N = %d, M = %d, num_col = %d\n", N, M, num_col);
                    fflush(stdout);

                    // Ensure all Phase-2 allocations happen AFTER this block

                    printf("[INFO] Transitioning to Phase 2...\n");
                    printf("[DEBUG] Entering Phase 2...\n");
                    fflush(stdout);
                    phase = 2;
                    printf("[DEBUG] Entering Phase 2 iteration loop\n");
                    printf("[DEBUG] basic_index: ");
                    for (int i = 0; i < N; i++) printf("%d ", basic_index[i]);
                    printf("\n[DEBUG] non_basic_index: ");
                    for (int i = 0; i < M; i++) printf("%d ", non_basic_index[i]);
                    printf("\n");
                    break;

                } else {
#if DEBUG
                    printf("\nbasic variables\n");
                    for (int j = 0; j < N; j++) {
                        printf("%d%s", basic_index[j], j == N - 1 ? "" : " ");
                    }
                    printf("\nXb\n");
                    for (int j = 0; j < N; j++) {
                        printf("%.3f%s", Xb[j], j == N - 1 ? "" : " ");
                    }
#endif
		    printf("iterations: %d\n", iter_num);
                    float z = 0;
		    for (int i = 0; i < N; i++) {
			z += c[basic_index[i]] * Xb[i];
		    }
		    printf("optimal value: %f\n", z);

                    phase = 3;
                    break;
                }
            }

            int entering_var = non_basic_index[entering_index];
	    matrixVecMulti<<<gridDim, blockDim>>>(B_inverse__dev, &(A_T__dev[num_row*entering_var]), A_bar__dev, N);
	    matrixVecMulti<<<gridDim, blockDim>>>(B_inverse__dev, b__dev, Xb__dev, N);
	    cuda_ret = cudaMemcpy(A_bar, A_bar__dev, sizeof(float)*num_row, cudaMemcpyDeviceToHost);
            cuda_ret = cudaMemcpy(Xb, Xb__dev, sizeof(float)*num_row, cudaMemcpyDeviceToHost);
	    cudaDeviceSynchronize();

	    int leaving_index = -1;
            float min_val = INFINITY;
            for (int i = 0; i < num_row; i++) {
                if (A_bar[i] > 1e-6) {
                    float ratio = Xb[i] / A_bar[i];
                    if (ratio < min_val) {
                        min_val = ratio;
                        leaving_index = i;
                    }
                }
            }
            if (leaving_index == -1) {
                printf("Unbounded LP.\n");
                return 1;
            }
            int leaving_var = basic_index[leaving_index];
            assign_variable(basic_index, non_basic_index, entering_var, leaving_var, N, M);

	    startTime(&timer2);
	    memcpy(E, I, sizeof(float)*num_row*num_row);
	    for(int i = 0; i < N; i++) {
		E[i*N+leaving_index] = ((i == leaving_index) ? 1.0f : -A_bar[i]) / A_bar[leaving_index];
	    }
	    cuda_ret = cudaMemcpy(E__dev, E, sizeof(float)*num_row*num_row, cudaMemcpyHostToDevice);
	    int BLOCKS = (N-1) / THREADS_PER_BLOCK + 1;
            dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
	    dim3 blocks(BLOCKS, BLOCKS);
	    matrixMulti<<<blocks, threads>>>(E__dev, B_inverse__dev, B_inverse_new__dev, N);
	    cudaDeviceSynchronize();
	    cuda_ret = cudaMemcpy(B_inverse__dev, B_inverse_new__dev, sizeof(float)*num_row*num_row, cudaMemcpyDeviceToDevice);

	    stopTime(&timer2);
	    temp_time += elapsedTime(timer2);

            for (int i = 0; i < N; i++) {
                free(Ab[i]);
                free(An[i]);
            }
            free(Ab);
	    free(An);
        }
    }
    stopTime(&timer);
    printf("RSA: %f s\n", elapsedTime(timer));
    printf("temp_time: %f s\n", temp_time);
    printf("\n");

    free(E);
    free(p);
    free(cN_bar);
    free(Xb);
    free(A_bar);

    for (int i = 0; i < num_row; i++) {
        free(A[i]);
    }
    free(A);
    free(b);
    free(c);
    free(original_c);
    free(non_basic_index);
    free(basic_index);
    return 0;
}
