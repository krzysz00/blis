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

#define SCAADD(n, nam, mx, my, r)\
    ULABEL(nam##n)\
    VMOVS(my, r)\
    M_VADDS(mx, r)\
    VMOVS(r, my)\


// Compute a dot product in < 4 elements
#define TAIL(name, xmr, ymr, n1, n2, n3, countr)\
    DUFFJMP(name, countr)\
    SCAADD(3, name, DEREF_OFF(xmr, 16), DEREF_OFF(ymr, 16), XMM(n3))\
    SCAADD(2, name, DEREF_OFF(xmr, 8), DEREF_OFF(ymr, 8), XMM(n2))\
    SCAADD(1, name, DEREF(xmr), DEREF(ymr), XMM(n1))\
    SHL(IMM(3), countr)\
    ADD(countr, xmr)\
    ADD(countr, ymr)\
    ULABEL(name##0)

#define LOAD1 VMOVAP(DEREF_OFF(VAR(y), (0 * MAX_TAIL_SIZE * ELEM_SIZE)), YMM(0))
#define LOAD2 LOAD1 VMOVAP(DEREF_OFF(VAR(y), (1 * MAX_TAIL_SIZE * ELEM_SIZE)), YMM(1))
#define LOAD3 LOAD2 VMOVAP(DEREF_OFF(VAR(y), (2 * MAX_TAIL_SIZE * ELEM_SIZE)), YMM(2))
#define LOAD4 LOAD3 VMOVAP(DEREF_OFF(VAR(y), (3 * MAX_TAIL_SIZE * ELEM_SIZE)), YMM(3))
#define LOAD5 LOAD4 VMOVAP(DEREF_OFF(VAR(y), (4 * MAX_TAIL_SIZE * ELEM_SIZE)), YMM(4))
#define LOAD6 LOAD5 VMOVAP(DEREF_OFF(VAR(y), (5 * MAX_TAIL_SIZE * ELEM_SIZE)), YMM(5))
#define LOAD7 LOAD6 VMOVAP(DEREF_OFF(VAR(y), (6 * MAX_TAIL_SIZE * ELEM_SIZE)), YMM(6))
#define LOAD8 LOAD7 VMOVAP(DEREF_OFF(VAR(y), (7 * MAX_TAIL_SIZE * ELEM_SIZE)), YMM(7))

#define ADD1 R_VADDP(DEREF_OFF(VAR(x), (0 * MAX_TAIL_SIZE * ELEM_SIZE)), YMM(0), YMM(0))
#define ADD2 ADD1 R_VADDP(DEREF_OFF(VAR(x), (1 * MAX_TAIL_SIZE * ELEM_SIZE)), YMM(1), YMM(1))
#define ADD3 ADD2 R_VADDP(DEREF_OFF(VAR(x), (2 * MAX_TAIL_SIZE * ELEM_SIZE)), YMM(2), YMM(2))
#define ADD4 ADD3 R_VADDP(DEREF_OFF(VAR(x), (3 * MAX_TAIL_SIZE * ELEM_SIZE)), YMM(3), YMM(3))
#define ADD5 ADD4 R_VADDP(DEREF_OFF(VAR(x), (4 * MAX_TAIL_SIZE * ELEM_SIZE)), YMM(4), YMM(4))
#define ADD6 ADD5 R_VADDP(DEREF_OFF(VAR(x), (5 * MAX_TAIL_SIZE * ELEM_SIZE)), YMM(5), YMM(5))
#define ADD7 ADD6 R_VADDP(DEREF_OFF(VAR(x), (6 * MAX_TAIL_SIZE * ELEM_SIZE)), YMM(6), YMM(6))
#define ADD8 ADD7 R_VADDP(DEREF_OFF(VAR(x), (7 * MAX_TAIL_SIZE * ELEM_SIZE)), YMM(7), YMM(7))

#define STORE1 VMOVAP(YMM(0), DEREF_OFF(VAR(y), (0 * MAX_TAIL_SIZE * ELEM_SIZE)))
#define STORE2 STORE1 VMOVAP(YMM(1), DEREF_OFF(VAR(y), (1 * MAX_TAIL_SIZE * ELEM_SIZE)))
#define STORE3 STORE2 VMOVAP(YMM(2), DEREF_OFF(VAR(y), (2 * MAX_TAIL_SIZE * ELEM_SIZE)))
#define STORE4 STORE3 VMOVAP(YMM(3), DEREF_OFF(VAR(y), (3 * MAX_TAIL_SIZE * ELEM_SIZE)))
#define STORE5 STORE4 VMOVAP(YMM(4), DEREF_OFF(VAR(y), (4 * MAX_TAIL_SIZE * ELEM_SIZE)))
#define STORE6 STORE5 VMOVAP(YMM(5), DEREF_OFF(VAR(y), (5 * MAX_TAIL_SIZE * ELEM_SIZE)))
#define STORE7 STORE6 VMOVAP(YMM(6), DEREF_OFF(VAR(y), (6 * MAX_TAIL_SIZE * ELEM_SIZE)))
#define STORE8 STORE7 VMOVAP(YMM(7), DEREF_OFF(VAR(y), (7 * MAX_TAIL_SIZE * ELEM_SIZE)))

#define INDIRECT(name) name
#define MACROCAT(a, b) INDIRECT(a##b)

#define LOOP(n)\
    ULABEL(addloop##n)\
    MACROCAT(LOAD, n)\
    MACROCAT(ADD, n)\
    MACROCAT(STORE, n)\
    ADD(IMM((n * MAX_TAIL_SIZE * ELEM_SIZE)), VAR(x))\
    ADD(IMM((n * MAX_TAIL_SIZE * ELEM_SIZE)), VAR(y))\
    SUB(IMM(n), VAR(iter))

#define SSETAIL(n)\
    LOOP(n)\
    JMP(UNIQ(addloop0))\

#define UNROLL_SIZE 8
#define ELEM_SIZE 8

#define SSE(instr) instr##d
#define MAX_TAIL_SIZE 4

void bli_daddv_opt_var1
     (
       conj_t conjx,
       dim_t n,
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
        BLIS_DADDV_KERNEL_REF
        (
          conjx,
          n,
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
     TEST(VAR(pre), VAR(pre))
     JZ(UNIQ(preadd0))
     TAIL(preadd, VAR(x), VAR(y), 2, 1, 0, VAR(pre))
     ASM(vzeroall)
     ALIGN16
     ULABEL(addloop_start)
     TEST(VAR(iter), VAR(iter))
     JLE(UNIQ(addloop0))
     CMP(IMM(UNROLL_SIZE), VAR(iter))
     JL(UNIQ(addloop_tail))
     LOOP(8)
     JMP(UNIQ(addloop_start))
     ULABEL(addloop_tail)
     DUFFJMP(addloop, VAR(iter))
     ULABEL(addloop0)
     TEST(VAR(post), VAR(post))
     JZ(UNIQ(add_end))
     TAIL(postadd, VAR(x), VAR(y), 0, 1, 2, VAR(post))
     JMP(UNIQ(add_end))
     SSETAIL(7)
     SSETAIL(6)
     SSETAIL(5)
     SSETAIL(4)
     SSETAIL(3)
     SSETAIL(2)
     SSETAIL(1)
     ULABEL(add_end)
     JUMPTABLES()
     JUMPTABLE4(preadd)
     JUMPTABLE8(addloop)
     JUMPTABLE4(postadd)
     END_JUMPTABLES()
     : [pre] "+r" (n_pre),
       [iter] "+r" (n_iter),
       [post] "+r" (n_post),
       [x] "+r" (x),
       [y] "+r" (y),
       [jump_tmp1] "=r" (jump_tmp1),
       [jump_tmp2] "=r" (jump_tmp2)
     :
     : "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8",
       "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15", "memory"
     );
    return;
}
#undef TYPE_LETTER


