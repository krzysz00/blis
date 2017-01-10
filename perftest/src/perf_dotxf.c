#include "general.h"
#include <stdlib.h>

void ddotxf_kernel_halfopt
     (
       conj_t    conjat,
       conj_t    conjx,
       dim_t     m,
       dim_t     b_n,
       double*   alpha,
       double*   a, inc_t inca, inc_t lda,
       double*   x, inc_t incx,
       double*   beta,
       double*   y, inc_t incy,
       cntx_t*   cntx
      ) {
    bool_t use_ref = FALSE;

    if (bli_zero_dim1(b_n)) return;

    if (bli_zero_dim1(m)) {
        bli_dscalv(BLIS_NO_CONJUGATE, b_n,
                      beta, y, incy,
                      cntx);
        return;
    }
    // If there is anything that would interfere with our use of aligned
    // vector loads/stores, call the reference implementation.
    if (b_n < BLIS_DEFAULT_DF_D) {
        use_ref = TRUE;
    }
    else if (bli_has_nonunit_inc2(inca, incx)) {
        use_ref = TRUE;
    }
    else if (bli_is_unaligned_to( a,     BLIS_SIMD_ALIGN_SIZE) ||
             bli_is_unaligned_to( a+lda, BLIS_SIMD_ALIGN_SIZE) ||
             bli_is_unaligned_to( x,     BLIS_SIMD_ALIGN_SIZE)) {
        use_ref = TRUE;

        // If a, the second column of a, and x are unaligned by the same
        // offset, then we can still use an implementation that depends on
        // alignment for most of the operation.
        off_t off_a  = bli_offset_past_alignment(a, BLIS_SIMD_ALIGN_SIZE);
        off_t off_a2 = bli_offset_past_alignment(a+lda, BLIS_SIMD_ALIGN_SIZE);
        off_t off_x = bli_offset_past_alignment(x, BLIS_SIMD_ALIGN_SIZE);

        if (off_a == off_a2 && off_a == off_x) {
            use_ref = FALSE;
        }
    }

    // Call the reference implementation if needed.
    if (use_ref == TRUE) {
        BLIS_DDOTXF_KERNEL_REF
            (
             conjat,
             conjx,
             m,
             b_n,
             alpha,
             a, inca, lda,
             x, incx,
             beta,
             y, incy,
             cntx
             );
        return;
    }

    double beta_val = *beta;
    double alpha_val = *alpha;
    for (int i = 0; i < BLIS_DEFAULT_DF_D; i++) {
        double out = 0.0;
        BLIS_DDOTV_KERNEL(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE,\
                          m, a + i * lda, 1, x, 1, &out, cntx);
        y[i] = beta_val * y[i] + alpha_val * out;
    }
    return;
}

PERF_FN(dotxf, double flops_me, flops_them, flops_semi;\
        int lda = i + ((i % 4 != 0) ? (4 - (i % 4)) : 0);
        double* a1 = (double*)memsimd(lda * BLIS_DEFAULT_DF_D * sizeof(double));\
        double* x1 = (double*)memsimd(i * sizeof(double));\
        double* y1 = (double*)memsimd(BLIS_DEFAULT_DF_D * sizeof(double));\
        double* a2 = (double*)memsimd(lda * BLIS_DEFAULT_DF_D * sizeof(double));\
        double* x2 = (double*)memsimd(i * sizeof(double));\
        double* y2 = (double*)memsimd(BLIS_DEFAULT_DF_D * sizeof(double));\
        double* a3 = (double*)memsimd(lda * BLIS_DEFAULT_DF_D * sizeof(double)); \
        double* x3 = (double*)memsimd(i * sizeof(double));\
        double* y3 = (double*)memsimd(BLIS_DEFAULT_DF_D * sizeof(double));\
        double alpha = -2;
        double beta = 1.2;
        bli_drandm(0, BLIS_DENSE, i, BLIS_DEFAULT_DF_D, a1, 1, lda, NULL);\
        memcpy(a2, a1, lda * BLIS_DEFAULT_DF_D * sizeof(double));
        bli_drandv(i, x1, 1, NULL);\
        bli_drandv(BLIS_DEFAULT_DF_D, y1, 1, NULL);\
        memcpy(x2, x1, i * sizeof(double));\
        memcpy(y2, y1, BLIS_DEFAULT_DF_D * sizeof(double));\
        memcpy(x3, x2, i * sizeof(double));                 \
        memcpy(y3, y2, BLIS_DEFAULT_DF_D * sizeof(double));\
        FLOPS_INTO(BLIS_DEFAULT_DF_D, flops_me,\
                   BLIS_DDOTXF_KERNEL(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, \
                                      i, BLIS_DEFAULT_DF_D, &alpha,     \
                                      a1, 1, lda, x1, 1, &beta, y1, 1, &cntx);); \
        FLOPS_INTO(BLIS_DEFAULT_DF_D, flops_them,\
                   BLIS_DDOTXF_KERNEL_REF(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, \
                                          i, BLIS_DEFAULT_DF_D, &alpha, \
                                          a1, 1, lda, x1, 1, &beta, y1, 1, &cntx););\
        FLOPS_INTO(BLIS_DEFAULT_DF_D, flops_semi,\
                   ddotxf_kernel_halfopt(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE,\
                                         i, BLIS_DEFAULT_DF_D, &alpha,\
                                         a1, 1, lda, x1, 1, &beta, y1, 1, &cntx););\

        REPORT3(flops_me, flops_them, flops_semi);\
        free(a1);\
        free(x1);\
        free(y1);\
        free(a2);\
        free(x2);\
        free(y2);\
        free(a3);\
        free(x3);\
        free(y3);

)
