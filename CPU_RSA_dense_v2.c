#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "Eta.h"

// data is LP problem in standard form
// min z = cx
// s.t. Ax = b
//      x >= 0

// decide number of row and col
void getMatrixSize(const char* filename, int* rows, int* cols) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("did not find file\n");
        exit(1);
    }

    char line[86384];
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
double** allocateMatrix(int rows, int cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double*)malloc(cols * sizeof(double));
    }
    return matrix;
}

double* allocateVector(int size) {
    return (double*)malloc(size * sizeof(double));
}

// read file (A)
void readMatrixA(const char* filename, double** A, int rows, int cols) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("did not find file\n");
        exit(1);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fscanf(file, "%lf", &A[i][j]) != 1) {
                printf("error reading A[%d][%d]\n", i, j);
                exit(1);
            }
        }
    }
    fclose(file);
}

// read file (c and b)
void readVector(const char* filename, double* vec, int size) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("did not find file\n");
        exit(1);
    }

    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%lf", &vec[i]) != 1) {
            printf("error reading vector[%d]\n", i);
            exit(1);
        }
    }
    fclose(file);
}

// find the identity column in A to determine initail basis
int is_identity_column(double** A, int col, int num_row, int* pivot_row) {
    int one_count = 0, zero_count = 0, row = -1;
    for (int i = 0; i < num_row; i++) {
        if (fabs(A[i][col] - 1.0) < 1e-5) { // A[i][col] == 1
            one_count++;
            row = i;
        } else if (fabs(A[i][col]) < 1e-5) { // A[i][col] == 0
            zero_count++;
        }
    }
    if (one_count == 1 && zero_count == num_row - 1) { // identity col
        *pivot_row = row;
        return 1;
    }
    return 0;
}

// use basic index to find An
void find_An(double** A, int basis[], double **Aj, int N, int M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            Aj[i][j] = A[i][basis[j]];
        }
    }
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



int main(int argc, char *argv[]) {
    const char* A_file = argv[1];
    const char* b_file = argv[2];
    const char* c_file = argv[3];

    int num_row, num_col;
    // get matrix size
    getMatrixSize(A_file, &num_row, &num_col);

    // allocate memory 
    double** A = allocateMatrix(num_row, num_col);
    double* b = allocateVector(num_row);
    double* c = allocateVector(num_col);
    double* original_c = allocateVector(num_col);

    // read files
    readMatrixA(A_file, A, num_row, num_col);
    readVector(b_file, b, num_row);
    readVector(c_file, c, num_col);

    // initial basis
    int N = num_row;  // number of basic variables
    int M = num_col - N;  // number of non-basic variables

    int* basic_index = (int*)malloc(num_row * sizeof(int));
    int* non_basic_index = (int*)malloc(num_col * sizeof(int));
    int* row_covered = (int*)calloc(num_row, sizeof(int));

    clock_t start = clock();

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

    int phase = 2;

    while (phase <= 2){
        int iter_num = 0;
        ///***
        double** Ab_inverse = allocateMatrix(N, N);
        initialize_Ab_inverse(Ab_inverse, N);
        ///***

        
        while (iter_num <= 5000) {
            
            // printf("%d\n", iter_num);

            iter_num++;

            double** An = allocateMatrix(N, M);
            find_An(A, non_basic_index, An, N, M);


            double* cB = allocateVector(N);
            for (int i = 0; i < N; i++) {
                cB[i] = c[basic_index[i]];
            }

            //***
            // p calculation
            double* p = allocateVector(N);
            for (int j = 0; j < N; j++) {
                p[j] = 0.0;
                for (int i = 0; i < N; i++) {
                    p[j] += cB[i] * Ab_inverse[i][j];
                }
            }

            // reduced cost calculation
            double* cN_bar = allocateVector(M);
            for (int i = 0; i < M; i++) {
                double sum = 0;
                for (int j = 0; j < N; j++) {
                    sum += p[j] * An[j][i];
                }
                cN_bar[i] = c[non_basic_index[i]] - sum;
            }
            

            // find entering variable for cN_bar < 0
            int entering_index = -1;
            double min_value = -1e-10;
            for (int i = 0; i < M; i++) {
                if (cN_bar[i] < min_value) {
                    min_value = cN_bar[i];
                    entering_index = i;
                }
            }

            if (entering_index == -1) { // all cN_bar >= 0
                ///***
                // Xb calculation
                double* Xb = allocateVector(N);
                for (int i = 0; i < N; i++) {
                    Xb[i] = 0.0;
                    for (int j = 0; j < N; j++) {
                        Xb[i] += Ab_inverse[i][j] * b[j];
                    }
                }
                ///***


                
                // Phase 2 ends
                //printf("basic variables\n");
                //for (int j = 0; j < N; j++) {
                //    printf("%d%s", basic_index[j], j == N - 1 ? "" : " ");
                //}

                //printf("\nXb\n");
                //for (int j = 0; j < N; j++) {
                //    printf("%.3f%s", Xb[j], j == N - 1 ? "" : " ");
                //}

                clock_t end = clock();
                double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
                printf("Time elapsed: %.6f seconds\n", cpu_time_used);
                
                printf("\noptimal value\n");
                double z = 0;
                for (int i = 0; i < N; i++) z += c[basic_index[i]] * Xb[i];
                printf("%f\n", z);

                printf("iteration: %d\n", iter_num);

                phase = 3;
                break;
                }

            int entering_var = non_basic_index[entering_index];
            double* colA = allocateVector(N);
            for (int i = 0; i < num_row; i++) {
                colA[i] = A[i][entering_var];
            }
            
            ///***
            double* A_bar = allocateVector(N);
            for (int i = 0; i < N; i++) {
                A_bar[i] = 0.0;
                for (int j = 0; j < N; j++) {
                    A_bar[i] += Ab_inverse[i][j] * colA[j];
                }
            }
            
            
            double* Xb = allocateVector(N);
            for (int i = 0; i < N; i++) {
                Xb[i] = 0.0;
                for (int j = 0; j < N; j++) {
                    Xb[i] += Ab_inverse[i][j] * b[j];
                }
            }
            ///***

            // min ratio test
            double* min_ratio = (double*)calloc(num_row, sizeof(double));
            int leaving_index = -1;
            double min_val = INFINITY;
            for (int i = 0; i < num_row; i++) {
                if (A_bar[i] > 1e-8) {
                    double ratio = Xb[i] / A_bar[i];
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

            // update basis and non baisc index
            int leaving_var = basic_index[leaving_index];
            assign_variable(basic_index, non_basic_index, entering_var, leaving_var, N, M);

            ///***
            // update eta and Ab_inverse
            // check if pivot close to 0
            double pivot = A_bar[leaving_index];
            if (fabs(pivot) < 1e-8) {
                printf("Pivot too small: %.10f\n", pivot);
                exit(1);
            }

            EtaMatrix* eta = (EtaMatrix*)malloc(sizeof(EtaMatrix));
            eta->pivot = leaving_index;
            eta->eta_col = allocateVector(N);

            for (int i = 0; i < N; i++) {
                if (i == leaving_index)
                    eta->eta_col[i] = 1.0 / pivot;
                else
                    eta->eta_col[i] = -A_bar[i] / pivot;
            }            


            update_Ab_inverse_with_eta(Ab_inverse, eta, N);
            free(eta->eta_col);
            free(eta);
            ///***


            free(cB); free(p); free(cN_bar);
            free(Xb); free(min_ratio); free(colA); free(A_bar);
            for (int i = 0; i < N; i++) {
                free(An[i]);
            }
            
            free(An);

        }

    }

    for (int i = 0; i < num_row; i++) {
        free(A[i]);
    }
    free(A); free(b); free(c); free(original_c);
    free(non_basic_index); free(basic_index);
    
    return 0;
}
