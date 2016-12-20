#include "general.h"
#include <stdlib.h>

PERF_FN(dotv, double rho1 = -1;\
        double rho2 = -1;\
        double flops_me, flops_them;\
        double* x1 = (double*)malloc(i * sizeof(double));\
        double* y1 = (double*)malloc(i * sizeof(double));\
        double* x2 = (double*)malloc(i * sizeof(double));\
        double* y2 = (double*)malloc(i * sizeof(double));\
        bli_drandv(i, x1, 1, NULL);\
        memcpy(y1, x1, i * sizeof(double));\
        memcpy(x2, x1, i * sizeof(double));\
        memcpy(y2, y1, i * sizeof(double));\
        FLOPS_INTO(flops_me, BLIS_DDOTV_KERNEL(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE,\
                                              i, x1, 1, y1, 1, &rho1, NULL););\
        FLOPS_INTO(flops_them, BLIS_DDOTV_KERNEL_REF(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE,\
                                                     i, x2, 1, y2, 1, &rho2, NULL);); \
        if (fabs(rho1 - rho2) > 1e-8) retval = false;\
        else REPORT(flops_me, flops_them);\
        free(x1);\
        free(y1);\
        free(x2);\
        free(y2);
)
