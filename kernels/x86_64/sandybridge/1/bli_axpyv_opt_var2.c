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

// *alpha is in YMM15
#define GET_ALPHA(reg)\
    MOVQ(VAR(alpha), VAR(jump_tmp1))\
    VBROADCASTS(DEREF(VAR(jump_tmp1)), YMM(15))

#define SCAAXPY(n, nam, mx, my, rt, r)\
    ULABEL(nam##n)\
    VMOVS(my, r)\
    VMULS(mx, XMM(15), rt)\
    R_VADDS(rt, r, r)\
    VMOVS(r, my)\


// Compute a dot product in < 4 elements
#define TAIL(name, xmr, ymr, n1t, n1, n2t, n2, n3t, n3, countr)\
    DUFFJMP(name, countr)\
    SCAAXPY(3, name, DEREF_OFF(xmr, 16), DEREF_OFF(ymr, 16), XMM(n3t), XMM(n3))\
    SCAAXPY(2, name, DEREF_OFF(xmr, 8), DEREF_OFF(ymr, 8), XMM(n3t), XMM(n2))\
    SCAAXPY(1, name, DEREF(xmr), DEREF(ymr), XMM(n1t), XMM(n1))\
    SHL(IMM(3), countr)\
    ADD(countr, xmr)\
    ADD(countr, ymr)\
    ULABEL(name##0)

#define INDIRECT(name) name
#define MACROCAT(a, b) INDIRECT(a##b)

#define VECAXPY_NOLABEL(n, xreg)\
    VMOVAP(DEREF_OFF(VAR(y), (n * MAX_TAIL_SIZE * ELEM_SIZE)), YMM(n))\
    VMULP(DEREF_OFF(VAR(x), (n * MAX_TAIL_SIZE * ELEM_SIZE)), YMM(15), YMM(xreg))\
    R_VADDP(YMM(xreg), YMM(n), YMM(n))\
    VMOVAP(YMM(n), DEREF_OFF(VAR(y), (n * MAX_TAIL_SIZE * ELEM_SIZE)))

#define VECAXPY(lab, n, xreg)\
    ULABEL(axpyloop##lab)\
    VECAXPY_NOLABEL(n, xreg)

#define LOOP_NOLABEL()\
    VECAXPY_NOLABEL(5, 13)\
    VECAXPY_NOLABEL(4, 12)\
    VECAXPY_NOLABEL(3, 11)\
    VECAXPY_NOLABEL(2, 10)\
    VECAXPY_NOLABEL(1, 9)\
    VECAXPY_NOLABEL(0, 8)\
    ADD(IMM((UNROLL_SIZE * MAX_TAIL_SIZE * ELEM_SIZE)), VAR(x))\
    ADD(IMM((UNROLL_SIZE * MAX_TAIL_SIZE * ELEM_SIZE)), VAR(y))\
    SUB(IMM(UNROLL_SIZE), VAR(iter))

#define LOOP()\
    VECAXPY(6, 5, 13)\
    VECAXPY(5, 4, 12)\
    VECAXPY(4, 3, 11)\
    VECAXPY(3, 2, 10)\
    VECAXPY(2, 1, 9)\
    VECAXPY(1, 0, 8)\
    ADD(IMM((UNROLL_SIZE * MAX_TAIL_SIZE * ELEM_SIZE)), VAR(x))\
    ADD(IMM((UNROLL_SIZE * MAX_TAIL_SIZE * ELEM_SIZE)), VAR(y))\
    SUB(IMM(UNROLL_SIZE), VAR(iter))

#define UNROLL_SIZE 6
#define ELEM_SIZE 8

#define SSE(instr) instr##d
#define MAX_TAIL_SIZE 4

void bli_daxpyv_opt_var2
     (
       conj_t conjx,
       dim_t n,
       double* restrict alpha,
       double* restrict x, inc_t incx,
       double* restrict y, inc_t incy,
       cntx_t* cntx
     )
{
    bool_t use_ref = FALSE;

    dim_t register n_pre = 0;
    dim_t register n_iter = 0;
    dim_t register n_post = 0;
    uint64_t register jump_tmp1;// __asm__ ("r14");
    uint64_t register jump_tmp2;// __asm__ ("r15");

    // If the vector lengths are zero, set rho to zero and return.
    if (n == 0)
    {
        return;
    }

    // If there is anything that would interfere with our use of aligned
    // vector loads/stores, call the reference implementation.
    if ( bli_has_nonunit_inc2( incx, incy ) )
    {
        use_ref = TRUE;
    }
    else if ( bli_is_unaligned_to( x, BLIS_SIMD_ALIGN_SIZE ) ||
              bli_is_unaligned_to( y, BLIS_SIMD_ALIGN_SIZE ) )
    {
        use_ref = TRUE;

        // If a, the second column of a, and y are unaligned by the same
        // offset, then we can still use an implementation that depends on
        // alignment for most of the operation.
        dim_t off_x  = BLIS_SIMD_ALIGN_SIZE - bli_offset_past_alignment(x, BLIS_SIMD_ALIGN_SIZE);
        dim_t off_y  = BLIS_SIMD_ALIGN_SIZE - bli_offset_past_alignment(y, BLIS_SIMD_ALIGN_SIZE);

        if ( off_x == off_y )
        {
            use_ref = FALSE;
            n_pre = off_x / sizeof(double);
        }
    }

    // Call the reference implementation if needed.
    if ( use_ref == TRUE )
    {
        BLIS_DAXPYV_KERNEL_REF
        (
          conjx,
          n, alpha,
          x, incx,
          y, incy,
          cntx
        );
        return;
    }

    // Compute front edge cases if x and y were unaligned.
    n_post = (n - n_pre) % MAX_TAIL_SIZE;
    n_iter = (n - n_pre - n_post) / MAX_TAIL_SIZE;

    //    printf("n: %ld, iter: %ld, pre: %ld, post: %ld, addr: %llx, rho: %llx\n", n, n_iter, n_pre, n_post, (uint64_t)x, (uint64_t)rho);
    // registers are dispersed in pairs [x_tmp, accum]
    // current loop has four pairs, repeated once
    __asm__ volatile
    (
     GET_ALPHA()
     TEST(VAR(pre), VAR(pre))
     JZ(UNIQ(preaxpy0))
     TAIL(preaxpy, VAR(x), VAR(y), 10, 2, 9, 1, 8, 0, VAR(pre))
     ASM(vzeroall)
     GET_ALPHA()
     ALIGN16
     ULABEL(axpyloop_start)
     TEST(VAR(iter), VAR(iter))
     JLE(UNIQ(axpyloop0))
     CMP(IMM(UNROLL_SIZE), VAR(iter))
     JL(UNIQ(axpyloop_tail))
     LOOP()
     TEST(VAR(iter), VAR(iter))
     JLE(UNIQ(axpyloop0))
     CMP(IMM(UNROLL_SIZE), VAR(iter))
     JL(UNIQ(axpyloop_tail))
     LOOP_NOLABEL()
     JMP(UNIQ(axpyloop_start))
     ULABEL(axpyloop_tail)
     DUFFJMP(axpyloop, VAR(iter))
     ULABEL(axpyloop0)
     TEST(VAR(post), VAR(post))
     JZ(UNIQ(postaxpy0))
     "\n# memory overshoot (in negative) * 2^(lg(sizeof double) + lg(max_tail_size))\n\t"
     SAL(IMM(3 + 2), VAR(iter))
     ADD(VAR(iter), VAR(x))
     ADD(VAR(iter), VAR(y))
     ZEROUPPER()
     TAIL(postaxpy, VAR(x), VAR(y), 8, 0, 9, 1, 10, 2, VAR(post))
     JUMPTABLES()
     JUMPTABLE4(preaxpy)
     JUMPTABLE6(axpyloop)
     JUMPTABLE4(postaxpy)
     END_JUMPTABLES()
     : [pre] "+r" (n_pre),
       [iter] "+r" (n_iter),
       [post] "+r" (n_post),
       [x] "+r" (x),
       [y] "+r" (y),
       [jump_tmp1] "=r" (jump_tmp1),
       [jump_tmp2] "=r" (jump_tmp2)
     : [alpha] "m" (alpha)
     : "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8",
       "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15", "memory"
     );
    return;
}
#undef TYPE_LETTER
