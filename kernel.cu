__global__ void vecDiff(const float* A, const float* B, float* C, int n) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] - B[i];
    }
}
__global__ void vecDiv(const float* A, const float* B, float* C, int n) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] / B[i];
    }
}
__global__ void vecDot(const float* A, const float* B, float* C, int n) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] * B[i];
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
// vec: 1*n
// matrix: n*n
__global__ void vecMatrixMulti(const float *vec, const float *matrix, float *r, int n) {
    int c = blockDim.x*blockIdx.x + threadIdx.x;
    if(c < n) {
	float sum = 0.0f;
	for (int i = 0; i < n; i++) {
            sum += vec[i] * matrix[i*n+c];
	}
	r[c] = sum;
    }
}
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
/*
__iglobal__ void updateE(float *E, const float A_bar, int leaving index, int n) {
   int i = blockDim.x*blockIdx.x + threadIdx.x;
   E[i] = 0.0f;
   if(i < n) {
       
   }
}
*/
__global__ void matrixMulti(const float *a, const float *b, float *c, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < n) && (col < n)) {
	float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += a[row*n+i] * b[i*n+col];
        }
	c[row*n+col] = sum;
    }
}

__global__ void SpMV_ELL(int num_rows, float *data, unsigned int *col_index, int num_elem, float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
	float dot = 0;
	for (int i = 0; i < num_elem; i++) {
            dot += data[row+i*num_rows]*x[col_index[row+i*num_rows]];
        }
        y[row] = dot;
    }
}

__global__ void SpMV_CSR(int num_rows, float *data, int *col_index, int *row_ptr, float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
	float dot = 0;
	int row_start = row_ptr[row];
	int row_end = row_ptr[row+1];
	for (int elem = row_start; elem < row_end; elem++) {
            dot += data[elem] * x[col_index[elem]];
	}
	y[row] = dot;
    }
}

