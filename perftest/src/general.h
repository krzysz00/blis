#ifndef BLIS_PERF_GENERAL_H
#define BLIS_PERF_GENERAL_H

#include "blis.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define OUT_DIR_STR "out/"
#define OUT_FILE_EXT ".dat"

#define DEFAULT_MIN_N 4
#define DEFAULT_MAX_N 2000
#define DEFAULT_INC_N 4

#define FLOPS_PER_UNIT_PERF 1e9

extern void* memsimd(size_t size);

typedef struct perf_params_t {
    dim_t min_n;
    dim_t max_n;
    inc_t inc_n;
} perf_params_t;

#define PERF_FN(op_name, ...)\
    bool perf_##op_name (const perf_params_t *params)\
    {\
        printf("Benchmarking " #op_name "\n");\
        dim_t min_sz = params->min_n;\
        dim_t max_sz = params->max_n;\
        dim_t inc_sz = params->inc_n;\
        \
        FILE* data_out = fopen(OUT_DIR_STR #op_name OUT_FILE_EXT, "w");\
        if (data_out == NULL)\
        {\
            return false;\
        }\
        bool retval = true;\
        for (dim_t i = min_sz; i < max_sz; i += inc_sz)\
        {\
            __VA_ARGS__\
        }\
        fclose(data_out);\
        return retval;\
    }

#define FLOPS_INTO(perfvar, ...) do\
        {\
            perfvar = DBL_MAX;\
            double _time = bli_clock();\
            __VA_ARGS__;\
            perfvar = bli_clock_min_diff(perfvar, _time);\
            perfvar = ( 2.0 * i ) / perfvar / FLOPS_PER_UNIT_PERF;\
        } while(0)

#define REPORT(my_flops, their_flops) fprintf(data_out, "%ld,%.8e,%.8e\n", i, (my_flops), (their_flops))

#define PERF_FN_DECL(op_name) bool perf_##op_name (const perf_params_t *params);

#endif // BLIS_PERF_GENERAL_H
