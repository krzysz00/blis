#include "general.h"
#include <stdlib.h>

PERF_FN(addv, double flops_me, flops_them;\
        double* x1 = (double*)malloc(i * sizeof(double));\
        double* y1 = (double*)malloc(i * sizeof(double));\
        double* x2 = (double*)malloc(i * sizeof(double));\
        double* y2 = (double*)malloc(i * sizeof(double));\
        bli_drandv(i, x1, 1, NULL);\
        memcpy(y1, x1, i * sizeof(double));\
        memcpy(x2, x1, i * sizeof(double));\
        memcpy(y2, y1, i * sizeof(double));\
        FLOPS_INTO(flops_me, BLIS_DADDV_KERNEL(BLIS_NO_CONJUGATE,\
                                              i, x1, 1, y1, 1, NULL););\
        FLOPS_INTO(flops_them, BLIS_DADDV_KERNEL_REF(BLIS_NO_CONJUGATE,\
                                                     i, x2, 1, y2, 1, NULL);); \
        REPORT(flops_me, flops_them);
        free(x1);\
        free(y1);\
        free(x2);\
        free(y2);
)
