#include "general.h"
#include <getopt.h>

#define TRY(op)\
    do {\
        bool res = perf_##op(&params);\
        if (!res) {\
            fprintf(stderr, #op " failed to complete successfully\n");\
        }\
    } while(0)

PERF_FN_DECL(dotv)

static const struct option long_options[] = {
    {.name = "minimum", .has_arg = required_argument, .flag = NULL, .val = 'm'},
    {.name = "min", .has_arg = required_argument, .flag = NULL, .val = 'm'},
    {.name = "maximum", .has_arg = required_argument, .flag = NULL, .val = 'M'},
    {.name = "max", .has_arg = required_argument, .flag = NULL, .val = 'M'},
    {.name = "increment", .has_arg = required_argument, .flag = NULL, .val = 'i'},
    {.name = "inc", .has_arg = required_argument, .flag = NULL, .val = 'i'},
    {.name = "help", .has_arg = required_argument, .flag = NULL, .val = 'h'},
    {.name = "version", .has_arg = required_argument, .flag = NULL, .val = 'v'},
};

void usage(const char* argv0, int code) {
    fprintf(stderr, "Usage: %s [-h|--help] [-v|--version] [-m|--min|--minimum N] [-M|--max|--maximum N] [-i|--inc|--increment N]\n", argv0);
    exit(code);
}

int64_t int_arg_or_fail(const char* argv0) {
    if (!optarg) {
        usage(argv0, 1);
    }
    char *num_end;
    int64_t ret = strtoll(optarg, &num_end, 10);
    if (*num_end != '\0') {
        usage(argv0, 1);
    }
    return ret;
}

int main(int argc, char **argv) {
    dim_t min_n = DEFAULT_MIN_N;
    dim_t max_n = DEFAULT_MAX_N;
    inc_t inc_n = DEFAULT_INC_N;

    int c, index = 0;
    while ((c = getopt_long(argc, argv, "m:M:i:vh", long_options, &index)) != -1) {
        switch (c) {
        case 'm':
            min_n = int_arg_or_fail(argv[0]);
            break;
        case 'M':
            max_n = int_arg_or_fail(argv[0]);
            break;
        case 'i':
            inc_n = int_arg_or_fail(argv[0]);
            break;
        case 'v':
            printf("0.0.1\n");
            return 0;
        case 'h':
            usage(argv[0], 0);
            return -1;
        case '?':
        default:
            usage(argv[0], 1);
            return -1;
        }
    }

    const perf_params_t params = {.min_n = min_n, .max_n = max_n, .inc_n = inc_n};

    TRY(dotv);
    return 0;
}
