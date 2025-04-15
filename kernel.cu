__global__ void vecDiff_cN_bar(const int* v_index, const float* v, const float* w, float* r, int n) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < n) {
        r[i] = v[v_index[i]] - w[i];
    }
}

__global__ void vecDiv(const float* A, const float* B, float* C, int n) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] / B[i];
    }
}

__global__ void vecMultiConst(float* A, const float scalar, int n) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < n) {
        A[i] = A[i] * scalar;
    }
}
// matrix: n*n
// vec: n*1
__global__ void matrixVecMulti(const float *matrix, const float *vec, float *c, int n) {
    int r = blockDim.x*blockIdx.x + threadIdx.x;
    if(r < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += matrix[r*n+i] * vec[i];
        }
        c[r] = sum;
    }
}
__global__ void vecMatrixMulti(const float *vec, const float *matrix, const int *matrix_index, float *r, int m, int n) {
    int c = blockDim.x*blockIdx.x + threadIdx.x;
    if(c < m) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += vec[j] * matrix[matrix_index[c]*n+j];
        }
        r[c] = sum;
    }
}
// vec: 1*n
// matrix: n*n
__global__ void vecMatrixMulti_cB(const int *orig_vec_index, const float *orig_vec, const float *matrix, float *r, int n) {
    int c = blockDim.x*blockIdx.x + threadIdx.x;
    if(c < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += orig_vec[orig_vec_index[i]] * matrix[i*n+c];
        }
        r[c] = sum;
    }
}

__global__ void B_inverse_col(float *col, const float *B_inverse, int col_index, int n) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < n) {
        col[i] = B_inverse[col_index*n+i];
    }
}

__global__ void updateB_inverse(float *B_inverse, const float *eta_col, const float *row_r_copy, int n) {
   int row = blockIdx.x * blockDim.x + threadIdx.x;
   int col = blockIdx.y * blockDim.y + threadIdx.y;
   if ((row < n) && (col < n)) {
       B_inverse[row*n+col] += eta_col[row] * row_r_copy[col];
   }
}
