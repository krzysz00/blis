/*
   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"
#include "bli_asm_macros.h"

#define INCMEM(n, delta)\
    ADD(delta, VAR(A##n))

#define INCMEMS(delta)\
    INCMEM(0, delta)\
    INCMEM(1, delta)\
    INCMEM(2, delta)\
    INCMEM(3, delta)\
    ADD(delta, VAR(y))\

#define INCMEMS_CONST(delta)\
    INCMEMS(IMM(delta * ELEM_SIZE))

#define FETCH_CONST(n)\
    VBROADCASTS(DEREF_OFF(VAR(x), (n * ELEM_SIZE)), YMM(n))\
    VMULP(YMM(n), YMM(15), YMM(n))

#define FETCH_CONSTS()\
    VBROADCASTS(DEREF(VAR(alpha)), YMM(15))\
    FETCH_CONST(0)\
    FETCH_CONST(1)\
    FETCH_CONST(2)\
    FETCH_CONST(3)

#define SCAL_AX(m, n, reg)\
    VMULS(DEREF_OFF(VAR(A##n), (-m * ELEM_SIZE)), XMM(n), XMM(reg))

#define SCAL_ELEM(m, typ)\
    ULABEL(typ##m)\
    VMOVS(DEREF_OFF(VAR(y), (-m * ELEM_SIZE)), XMM(15))\
    SCAL_AX(m, 0, 4)\
    SCAL_AX(m, 1, 5)\
    SCAL_AX(m, 2, 6)\
    SCAL_AX(m, 3, 7)\
    R_VADDS(XMM(4), XMM(5), XMM(5))\
    R_VADDS(XMM(6), XMM(7), XMM(7))\
    R_VADDS(XMM(5), XMM(7), XMM(5))\
    R_VADDS(XMM(5), XMM(15), XMM(15))\
    VMOVS(XMM(15), DEREF_OFF(VAR(y), (-m * ELEM_SIZE)))

#define SCAL_SEGMENT(typ)\
    ZEROUPPER()\
    SHL(IMM(3), VAR(typ))\
    INCMEMS(VAR(typ))\
    SHR(IMM(3), VAR(typ))\
    DUFFJMP_GEN(typ##axpy, VAR(typ), VAR(jump_tmp), VAR(typ))\
    SCAL_ELEM(3, typ##axpy)\
    SCAL_ELEM(2, typ##axpy)\
    SCAL_ELEM(1, typ##axpy)\
    ULABEL(typ##axpy0)

#define VEC_AX(m, n, reg, delta)\
    VMULP(DEREF_OFF(VAR(A##n), ((m - 1) * MAX_TAIL_SIZE * ELEM_SIZE + delta)), YMM(n), YMM(reg))

#define VEC_AX2(m, n, r1, r2)\
    VEC_AX(m, n, r1, 0)\
    VEC_AX(m, n, r2, (SSE_ALIGN_SIZE * ELEM_SIZE))

#define AXPY_ELEM(m, dprefetch)\
    ULABEL(axpyloop##m)\
    VMOVAP(DEREF_OFF(VAR(y), ((m - 1) * MAX_TAIL_SIZE * ELEM_SIZE)), YMM(14))\
    VMOVAP(DEREF_OFF(VAR(y), (((m - 1) * MAX_TAIL_SIZE + SSE_ALIGN_SIZE) * ELEM_SIZE)), YMM(15))\
    PREFETCH(1, DEREF_OFF(VAR(y), (dprefetch * MAX_TAIL_SIZE * ELEM_SIZE)))\
    VEC_AX2(m, 0, 4, 8)\
    VEC_AX2(m, 1, 5, 9)\
    VEC_AX2(m, 2, 6, 10)\
    VEC_AX2(m, 3, 7, 11)\
    R_VADDP(YMM(4), YMM(5), YMM(5))\
    R_VADDP(YMM(8), YMM(9), YMM(9))\
    R_VADDP(YMM(6), YMM(7), YMM(7))\
    R_VADDP(YMM(10), YMM(11), YMM(11))\
    R_VADDP(YMM(5), YMM(7), YMM(5))\
    R_VADDP(YMM(9), YMM(11), YMM(9))\
    R_VADDP(YMM(5), YMM(14), YMM(14))\
    R_VADDP(YMM(9), YMM(15), YMM(15))\
    VMOVAP(YMM(14), DEREF_OFF(VAR(y), ((m - 1) * MAX_TAIL_SIZE * ELEM_SIZE))) \
    VMOVAP(YMM(15), DEREF_OFF(VAR(y), (((m - 1) * MAX_TAIL_SIZE + SSE_ALIGN_SIZE) * ELEM_SIZE)))


#define MAIN_LOOP_BODY()\
    AXPY_ELEM(8, 6)\
    AXPY_ELEM(7, 5)\
    AXPY_ELEM(6, 4)\
    AXPY_ELEM(5, 3)\
    AXPY_ELEM(4, 2)\
    AXPY_ELEM(3, 1)\
    AXPY_ELEM(2, 0)\
    AXPY_ELEM(1, 8)\
    INCMEMS_CONST((UNROLL_SIZE * MAX_TAIL_SIZE))\
    SUB(IMM(UNROLL_SIZE), VAR(iter))
// there's no 0 because that would index too far back

#define SINGLE_ALIGNED_SEGMENT()\
    VMOVAP(DEREF(VAR(y)), YMM(14))\
    VEC_AX(1, 0, 4, 0)\
    VEC_AX(1, 1, 5, 0)\
    VEC_AX(1, 2, 6, 0)\
    VEC_AX(1, 3, 7, 0)\
    R_VADDP(YMM(4), YMM(5), YMM(5))\
    R_VADDP(YMM(6), YMM(7), YMM(7))\
    R_VADDP(YMM(5), YMM(7), YMM(5))\
    R_VADDP(YMM(5), YMM(14), YMM(14))\
    VMOVAP(YMM(14), DEREF(VAR(y)))\
    INCMEMS_CONST(SSE_ALIGN_SIZE)\
    SUB(IMM(4), VAR(post))

#define VERT_SIZE 4
#define UNROLL_SIZE 8
#define ELEM_SIZE 8

#define SSE(instr) instr##d
#define MAX_TAIL_SIZE 8
#define SSE_ALIGN_SIZE 4

void bli_daxpyf_opt_4x8
     (
       conj_t    conjat,
       conj_t    conjx,
       dim_t     m,
       dim_t     b_n,
       double*   alpha,
       double*   a, inc_t inca, inc_t lda,
       double*   x, inc_t incx,
       double*   y, inc_t incy,
       cntx_t*   cntx
     )
{
    bool_t use_ref = FALSE;

    dim_t register m_pre = 0;

    if (bli_zero_dim1(b_n)) return;

    if (bli_zero_dim1(m)) return;
    // If there is anything that would interfere with our use of aligned
    // vector loads/stores, call the reference implementation.
    if (b_n < VERT_SIZE) {
        use_ref = TRUE;
    }
    else if (bli_has_nonunit_inc2(inca, incy)) {
        use_ref = TRUE;
    }
    else if (bli_is_unaligned_to( a,     BLIS_SIMD_ALIGN_SIZE) ||
             bli_is_unaligned_to( a+lda, BLIS_SIMD_ALIGN_SIZE) ||
             bli_is_unaligned_to( y,     BLIS_SIMD_ALIGN_SIZE)) {
        use_ref = TRUE;

        // If a, the second column of a, and x are unaligned by the same
        // offset, then we can still use an implementation that depends on
        // alignment for most of the operation.
        off_t off_a  = bli_offset_past_alignment(a, BLIS_SIMD_ALIGN_SIZE);
        off_t off_a2 = bli_offset_past_alignment(a+lda, BLIS_SIMD_ALIGN_SIZE);
        off_t off_y = bli_offset_past_alignment(y, BLIS_SIMD_ALIGN_SIZE);

        if (off_a == off_a2 && off_a == off_y) {
            use_ref = FALSE;
            m_pre   = (BLIS_SIMD_ALIGN_SIZE - off_y) / ELEM_SIZE;
        }
    }

    // Call the reference implementation if needed.
    if (use_ref == TRUE) {
        BLIS_DAXPYF_KERNEL_REF
            (
             conjat,
             conjx,
             m,
             b_n,
             alpha,
             a, inca, lda,
             x, incx,
             y, incy,
             cntx
             );
        return;
    }

    double stor[4] __attribute__((aligned (32)));

    if (incx != 1) {
        double *dx = x;
        double *dstor = stor;
        for (int i = 0; i < VERT_SIZE; i++) {
            *dstor = *dx;
            dstor++;
            dx += incx;
        }
    }

    dim_t register m_post = (m - m_pre) % MAX_TAIL_SIZE;
    dim_t register m_iter = (m - m_pre - m_post) / MAX_TAIL_SIZE;
    dim_t register jump_tmp = 0;
    double register *a1 = a + 1 * lda;
    double register *a2 = a + 2 * lda;
    double register *a3 = a + 3 * lda;
    //    printf("n: %ld, iter: %ld, pre: %ld, post: %ld, addr: %llx, rho: %llx\n", n, n_iter, n_pre, n_post, (uint64_t)x, (uint64_t)rho);
    // registers are dispersed in pairs [x_tmp, accum]
    // current loop has four pairs, repeated once
    __asm__ volatile
    (
     ASM(vzeroall)
     FETCH_CONSTS()
     TEST(VAR(pre), VAR(pre))
     JZ(UNIQ(axpyloop_start))
     SCAL_SEGMENT(pre)
     "\n# Deal with cleared off upper part from the scalar segment\n\t"
     FETCH_CONSTS()
     ALIGN16
     ULABEL(axpyloop_start)
     TEST(VAR(iter), VAR(iter))
     JLE(UNIQ(axpyloop0))
     CMP(IMM(UNROLL_SIZE), VAR(iter))
     JL(UNIQ(axpyloop_tail))
     MAIN_LOOP_BODY()
     JMP(UNIQ(axpyloop_start))
     ULABEL(axpyloop_tail)
     DUFFJMP_GEN(axpyloop, VAR(iter), VAR(jump_tmp), VAR(pre))
     ULABEL(axpyloop0)
     TEST(VAR(post), VAR(post))
     JZ(UNIQ(postaxpy0))
     "\n# memory overshoot (in negative) * 2^(lg(sizeof double) + lg(max_tail_size))\n\t"
     SAL(IMM(3 + 3), VAR(iter))
     INCMEMS(VAR(iter))
     CMP(IMM(4), VAR(post))
     JL(UNIQ(small_tail_part))
     SINGLE_ALIGNED_SEGMENT()
     ULABEL(small_tail_part)
     SCAL_SEGMENT(post)
     JUMPTABLES()
     JUMPTABLE4(preaxpy)
     JUMPTABLE8(axpyloop)
     JUMPTABLE4(postaxpy)
     END_JUMPTABLES()
     : [pre] "+r" (m_pre),
       [iter] "+r" (m_iter),
       [post] "+r" (m_post),
       [y] "+r" (y),
       [A0] "+r" (a),
       [A1] "+r" (a1),
       [A2] "+r" (a2),
       [A3] "+r" (a3),
       [jump_tmp] "=r" (jump_tmp)
     : [x] "r" (incx == 1 ? x : stor),
       [alpha] "r" (alpha)
     : "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8",
       "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15", "memory"
     );
    return;
}
