#include <stdio.h>
#include <stdlib.h>

// allocate a matrix
double **allocateMatrix2(int size) {
    double **matrix = (double **)malloc(size * sizeof(double *));
    for (int i = 0; i < size; i++) {
        matrix[i] = (double *)malloc(size * sizeof(double));
    }
    return matrix;
}

// free matrix memory
void freeMatrix(double **matrix, int size) {
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Gauss-Jordan elimination
int inverseMatrix(double **matrix, double **inverse, int size) {
    int i, j, k;
    double ratio;

    double **augmented = allocateMatrix2(size * 2);

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            augmented[i][j] = matrix[i][j];
        }
        for (j = size; j < 2 * size; j++) {
            augmented[i][j] = (i == (j - size)) ? 1 : 0;
        }
    }

    // row operations
    for (i = 0; i < size; i++) {
        if (augmented[i][i] == 0) {
            freeMatrix(augmented, size);
            return 0; // Singular matrix
        }

        double diag = augmented[i][i];
        for (j = 0; j < 2 * size; j++) {
            augmented[i][j] /= diag;
        }

        for (k = 0; k < size; k++) {
            if (k != i) {
                ratio = augmented[k][i];
                for (j = 0; j < 2 * size; j++) {
                    augmented[k][j] -= ratio * augmented[i][j];
                }
            }
        }
    }

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            inverse[i][j] = augmented[i][j + size];
        }
    }

    freeMatrix(augmented, size);  // free matrix memory
    return 1;
}