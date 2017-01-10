#include "general.h"
#include <stdlib.h>
#include <errno.h>

void* memsimd(size_t size) {
    void* ret;
    if (posix_memalign(&ret, BLIS_SIMD_ALIGN_SIZE, size) != 0) {
        perror("memsimd allocation: ");
        abort();
    }
    return ret;
}
