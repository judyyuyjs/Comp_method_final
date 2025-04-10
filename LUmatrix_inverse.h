#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// metrix allocation
double **LUallocate(int size) {
    double **matrix = (double **)malloc(size * sizeof(double *));
    for (int i = 0; i < size; i++) {
        matrix[i] = (double *)calloc(size, sizeof(double));
    }
    return matrix;
}

// clean memory 
void freeMatrix(double **matrix, int size) {
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// find L and U with partial pivoting (PA=LU) dividing by small diagonal elements can cause to large numeriacal errors
int LU_decomposition(double **A, double **L, double **U, int *P, int size) {
    for (int i = 0; i < size; i++) {
        P[i] = i; // initialization, assume pivot is original diagnal element
        for (int j = 0; j < size; j++) {
            U[i][j] = A[i][j];
            L[i][j] = (i == j) ? 1.0 : 0.0; // L has 1 on diagonal
        }
    }

    for (int k = 0; k < size - 1; k++) {
        // find the pivot row (largest absolute value in column k)
        int maxRow = k;
        double maxVal = fabs(U[k][k]);
        for (int i = k + 1; i < size; i++) {
            if (fabs(U[i][k]) > maxVal) {
                maxVal = fabs(U[i][k]);
                maxRow = i;
            }
        }

        // change rows in U and update pivot array
        if (maxRow != k) {
            // swap rows in U
            double *tempU = U[k];
            U[k] = U[maxRow];
            U[maxRow] = tempU;
        
            // swap the corresponding rows in L
            for (int j = 0; j < k; j++) {
                double tempL = L[k][j];
                L[k][j] = L[maxRow][j];
                L[maxRow][j] = tempL;
            }
        
            // swap pivot record
            int tempP = P[k];
            P[k] = P[maxRow];
            P[maxRow] = tempP;
        }

        // *** check if any diagonal element is nearly zero
        if (fabs(U[k][k]) < 1e-15) {
            printf("small pivot detected at row %d, value = %.12e\n", k, U[k][k]);
            return 0;
        }        

        // eliminate below diagonal
        for (int i = k + 1; i < size; i++) {
            if (U[i][k] != 0) {
                L[i][k] = U[i][k] / U[k][k];
                for (int j = k; j < size; j++) {
                    U[i][j] -= L[i][k] * U[k][j];
                }
            }
        }
    }

    return 1;
}

// forward substitution, solve Ly = Pb (y = Ux)
void forwardSubstitution(double **L, int *P, double *b, double *y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = b[P[i]];
        for (int j = 0; j < i; j++) {
            y[i] -= L[i][j] * y[j];
        }
    }
}

// backward substitution, solve Ux = y
void backwardSubstitution(double **U, double *y, double *x, int size) {
    for (int i = size - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < size; j++) {
            x[i] -= U[i][j] * x[j];
        }
        x[i] /= U[i][i];
    }
}

// Solve Ax = b using LU decomposition
int solve_LU(double **A, double *b, double *x, int size) {
    double **L = LUallocate(size);
    double **U = LUallocate(size);
    int *P = (int *)malloc(size * sizeof(int));
    double *y = (double *)malloc(size * sizeof(double));

    if (!LU_decomposition(A, L, U, P, size)) {
        freeMatrix(L, size);
        freeMatrix(U, size);
        free(P);
        free(y);
        return 0; // Matrix is singular
    }

    forwardSubstitution(L, P, b, y, size);
    backwardSubstitution(U, y, x, size);

    freeMatrix(L, size);
    freeMatrix(U, size);
    free(P);
    free(y);
    return 1;
}
