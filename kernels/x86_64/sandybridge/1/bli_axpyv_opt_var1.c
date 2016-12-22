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

#define LOAD_BODY(prevn)\
    VMOVAP(DEREF_OFF(VAR(y), (prevn * MAX_TAIL_SIZE * ELEM_SIZE)), YMM(prevn))

#define LOAD0
#define LOAD1 LOAD0 LOAD_BODY(0)
#define LOAD2 LOAD1 LOAD_BODY(1)
#define LOAD3 LOAD2 LOAD_BODY(2)
#define LOAD4 LOAD3 LOAD_BODY(3)
#define LOAD5 LOAD4 LOAD_BODY(4)
#define LOAD6 LOAD5 LOAD_BODY(5)
#define LOAD7 LOAD6 LOAD_BODY(6)
#define LOAD8 LOAD7 LOAD_BODY(7)

#define AXPY_BODY(prevn, xreg)\
    VMULP(DEREF_OFF(VAR(x), (prevn * MAX_TAIL_SIZE * ELEM_SIZE)), YMM(15), YMM(xreg))\
    R_VADDP(YMM(xreg), YMM(prevn), YMM(prevn))\

#define AXPY0
#define AXPY1 AXPY0 AXPY_BODY(0, 8)
#define AXPY2 AXPY1 AXPY_BODY(1, 9)
#define AXPY3 AXPY2 AXPY_BODY(2, 10)
#define AXPY4 AXPY3 AXPY_BODY(3, 11)
#define AXPY5 AXPY4 AXPY_BODY(4, 12)
#define AXPY6 AXPY5 AXPY_BODY(5, 13)
#define AXPY7 AXPY6 AXPY_BODY(6, 14)
#define AXPY8 AXPY7 AXPY_BODY(7, 15)

#define STORE_BODY(prevn)\
    VMOVAP(YMM(prevn), DEREF_OFF(VAR(y), (prevn * MAX_TAIL_SIZE * ELEM_SIZE)))

#define STORE0
#define STORE1 STORE0 STORE_BODY(0)
#define STORE2 STORE1 STORE_BODY(1)
#define STORE3 STORE2 STORE_BODY(2)
#define STORE4 STORE3 STORE_BODY(3)
#define STORE5 STORE4 STORE_BODY(4)
#define STORE6 STORE5 STORE_BODY(5)
#define STORE7 STORE6 STORE_BODY(6)
#define STORE8 STORE7 STORE_BODY(7)

#define LOOP_NOLABEL(n)\
    MACROCAT(LOAD, n)\
    MACROCAT(AXPY, n)\
    MACROCAT(STORE, n)\
    ADD(IMM((n * MAX_TAIL_SIZE * ELEM_SIZE)), VAR(x))\
    ADD(IMM((n * MAX_TAIL_SIZE * ELEM_SIZE)), VAR(y))\
    SUB(IMM(n), VAR(iter))

#define LOOP(n)\
    ULABEL(axpyloop##n)\
    LOOP_NOLABEL(n)

#define SSETAIL(n)\
    LOOP(n)\
    JMP(UNIQ(axpyloop0))\

#define UNROLL_SIZE 7
#define ELEM_SIZE 8

#define SSE(instr) instr##d
#define MAX_TAIL_SIZE 4

void bli_daxpyv_opt_var1
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
     LOOP(7)
     CMP(IMM(UNROLL_SIZE), VAR(iter))
     JL(UNIQ(axpyloop_tail))
     LOOP_NOLABEL(7)
     JMP(UNIQ(axpyloop_start))
     ULABEL(axpyloop_tail)
     DUFFJMP(axpyloop, VAR(iter))
     ULABEL(axpyloop0)
     TEST(VAR(post), VAR(post))
     JZ(UNIQ(axpy_end))
     ZEROUPPER()
     TAIL(postaxpy, VAR(x), VAR(y), 8, 0, 9, 1, 10, 2, VAR(post))
     JMP(UNIQ(axpy_end))
     SSETAIL(6)
     SSETAIL(5)
     SSETAIL(4)
     SSETAIL(3)
     SSETAIL(2)
     SSETAIL(1)
     ULABEL(axpy_end)
     JUMPTABLES()
     JUMPTABLE4(preaxpy)
     JUMPTABLE8(axpyloop)
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


