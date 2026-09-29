/* dots.c - the matrix products of residual_attr.py in three summation orders.

   Y[i][j] = sum_k X[i][k] W[k][j] in binary32, X of T x n and W of n x m, both
   row-major:

     dots_seq   a left fold from zero, each product rounded before it is added
     dots_fma   the same fold with each product fused into its sum, rounding once
     dots_tree  the rounded products summed as a balanced binary tree over k,
                pairing (0,1), (2,3), ... and carrying an odd last term up

   Each (i, j) keeps its own order; only the independent columns run in
   parallel. -ffp-contract=off keeps dots_seq from being contracted into fused
   multiply-adds, and -mfma makes fmaf one instruction.

     gcc -O2 -mfma -ffp-contract=off -fopenmp -shared -o dots.dll dots.c */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define JB 64

void dots_seq(const float *X, const float *W, float *Y, int T, int n, int m) {
#pragma omp parallel for collapse(2) schedule(static)
  for (int i = 0; i < T; i++)
    for (int j0 = 0; j0 < m; j0 += JB) {
      int jb = m - j0 < JB ? m - j0 : JB;
      float acc[JB];
      for (int jj = 0; jj < jb; jj++) acc[jj] = 0.0f;
      for (int k = 0; k < n; k++) {
        const float x = X[(size_t)i * n + k];
        const float *w = W + (size_t)k * m + j0;
        for (int jj = 0; jj < jb; jj++) {
          float p = x * w[jj];
          acc[jj] = acc[jj] + p;
        }
      }
      memcpy(Y + (size_t)i * m + j0, acc, sizeof(float) * jb);
    }
}

void dots_fma(const float *X, const float *W, float *Y, int T, int n, int m) {
#pragma omp parallel for collapse(2) schedule(static)
  for (int i = 0; i < T; i++)
    for (int j0 = 0; j0 < m; j0 += JB) {
      int jb = m - j0 < JB ? m - j0 : JB;
      float acc[JB];
      for (int jj = 0; jj < jb; jj++) acc[jj] = 0.0f;
      for (int k = 0; k < n; k++) {
        const float x = X[(size_t)i * n + k];
        const float *w = W + (size_t)k * m + j0;
        for (int jj = 0; jj < jb; jj++) acc[jj] = fmaf(x, w[jj], acc[jj]);
      }
      memcpy(Y + (size_t)i * m + j0, acc, sizeof(float) * jb);
    }
}

void dots_tree(const float *X, const float *W, float *Y, int T, int n, int m) {
#pragma omp parallel
  {
    float *buf = (float *)malloc(sizeof(float) * (size_t)n * JB);
#pragma omp for collapse(2) schedule(static)
    for (int i = 0; i < T; i++)
      for (int j0 = 0; j0 < m; j0 += JB) {
        int jb = m - j0 < JB ? m - j0 : JB;
        for (int k = 0; k < n; k++) {
          const float x = X[(size_t)i * n + k];
          const float *w = W + (size_t)k * m + j0;
          for (int jj = 0; jj < jb; jj++) buf[k * JB + jj] = x * w[jj];
        }
        int len = n;
        while (len > 1) {
          int half = len / 2;
          for (int p = 0; p < half; p++)
            for (int jj = 0; jj < jb; jj++)
              buf[p * JB + jj] = buf[2 * p * JB + jj] + buf[(2 * p + 1) * JB + jj];
          if (len & 1) {
            memmove(buf + half * JB, buf + (len - 1) * JB, sizeof(float) * jb);
            len = half + 1;
          } else {
            len = half;
          }
        }
        memcpy(Y + (size_t)i * m + j0, buf, sizeof(float) * jb);
      }
    free(buf);
  }
}
