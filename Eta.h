#include <stdio.h>
#include <stdlib.h>

double** allocateMatrixE(int rows, int cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double*)malloc(cols * sizeof(double));
    }
    return matrix;
}

typedef struct {
    int pivot;
    double* eta_col;
} EtaMatrix;

void initialize_Ab_inverse(double** Ab_inverse, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Ab_inverse[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

void multiply_matrix(double** result, double** A, double** B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void update_Ab_inverse_with_eta(double** Ab_inverse, EtaMatrix* eta, int N) {
    int r = eta->pivot;

    // Step 1: u = eta - e_r
    double* u = (double*)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        u[i] = eta->eta_col[i];
    }
    u[r] -= 1.0;

    // Step 2: Copy original row r
    double* row_r_copy = (double*)malloc(N * sizeof(double));
    for (int j = 0; j < N; j++) {
        row_r_copy[j] = Ab_inverse[r][j];
    }

    // Step 3: Perform the rank-1 update safely
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            Ab_inverse[i][j] += u[i] * row_r_copy[j];
        }
    }

    free(u);
    free(row_r_copy);
}

