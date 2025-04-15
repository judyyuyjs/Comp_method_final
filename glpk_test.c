#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <glpk.h> 

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

// read file (b and c)
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

    // read files
    readMatrixA(A_file, A, num_row, num_col);
    readVector(b_file, b, num_row);
    readVector(c_file, c, num_col);
    
    glp_prob *lp;
    lp = glp_create_prob();
    glp_set_prob_name(lp, "standard_form_lp");
    glp_set_obj_dir(lp, GLP_MIN);  // minimization

    // load data into GLPK
    glp_add_rows(lp, num_row);
    for (int i = 0; i < num_row; i++) {
        glp_set_row_bnds(lp, i + 1, GLP_FX, b[i], b[i]); // Ax = b
    }

    glp_add_cols(lp, num_col);
    for (int j = 0; j < num_col; j++) {
        glp_set_col_bnds(lp, j + 1, GLP_LO, 0.0, 0.0);   // x ≥ 0
        glp_set_obj_coef(lp, j + 1, c[j]);               // cost coefficients
    }

    // load A into GLPK
    // GLPK stores A in CSR format
    int size = num_row * num_col;
    int *ia = (int *)malloc((size + 1) * sizeof(int));
    int *ja = (int *)malloc((size + 1) * sizeof(int));
    double *ar = (double *)malloc((size + 1) * sizeof(double));

    int idx = 1;
    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
            ia[idx] = i + 1;
            ja[idx] = j + 1;
            ar[idx] = A[i][j];
            idx++;
        }
    }

    glp_load_matrix(lp, size, ia, ja, ar);

    // solve
    clock_t start = clock();

    glp_smcp param;
    glp_init_smcp(&param);
    param.msg_lev = GLP_MSG_ALL;  // turn off output
    glp_simplex(lp, &param);

    clock_t end = clock();
    double solve_time = (double)(end - start) / CLOCKS_PER_SEC;

    int status = glp_get_status(lp);
    int iter_count = glp_get_it_cnt(lp);
    if (status == GLP_OPT) {
        printf("optimal value = %.6f\n", glp_get_obj_val(lp));
        printf("time (seconds) = %.6f\n", solve_time);
        printf("Iteration count: %d\n", iter_count);
        

    } else {
        printf("no optimal solution found (status = %d)\n", status);
    }

    glp_delete_prob(lp);
    free(ia);
    free(ja);
    free(ar);
    }