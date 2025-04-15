#include <stdio.h>
#include <stdlib.h>

/// Eta matrix struvture
typedef struct {
    int pivot;
    double* eta_col;
} EtaMatrix;

// initialize basis inverse
void initialize_Ab_inverse(double** Ab_inverse, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Ab_inverse[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

// update basis inverse using Eta
void update_Ab_inverse_with_eta(double** Ab_inverse, EtaMatrix* eta, int N) {
    int r = eta->pivot;

    double* u = (double*)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        u[i] = eta->eta_col[i];
    }
    u[r] -= 1.0;

    double* row_r_copy = (double*)malloc(N * sizeof(double));
    for (int j = 0; j < N; j++) {
        row_r_copy[j] = Ab_inverse[r][j];
    }

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            Ab_inverse[i][j] += u[i] * row_r_copy[j];
        }
    }

    free(u);
    free(row_r_copy);
}
