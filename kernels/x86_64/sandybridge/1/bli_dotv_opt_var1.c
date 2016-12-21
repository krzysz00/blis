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

#define SCADOT(n, nam, rx, mx, my, rd)\
    ULABEL(nam##n)\
    VMOVS(mx, rx)\
    VMULADDS(my, rx, rd, rx)


// Compute a dot product in < 4 elements
#define TAIL(name, xmr, ymr, n1t, n1, n2t, n2, n3t, n3, countr)\
    ZERO(XMM(n1))\
    ZERO(XMM(n2))\
    ZERO(XMM(n3))\
    DUFFJMP(name, countr)\
    SCADOT(3, name, XMM(n3t), DEREF_OFF(xmr, 16), DEREF_OFF(ymr, 16), XMM(n3))\
    SCADOT(2, name, XMM(n2t), DEREF_OFF(xmr, 8), DEREF_OFF(ymr, 8), XMM(n2))\
    SCADOT(1, name, XMM(n1t), DEREF(xmr), DEREF(ymr), XMM(n1))\
    SHL(IMM(3), countr)\
    ADD(countr, xmr)\
    ADD(countr, ymr)\
    M_VADDS(DEREF(VAR(rho)), XMM(n1))\
    R_VADDS(XMM(n2), XMM(n3), XMM(n1t))\
    R_VADDS(XMM(n1), XMM(n1t), XMM(n1))\
    VMOVS(XMM(n1), DEREF(VAR(rho)))\
    ULABEL(name##0)

// x[delta] * y[delta] -> r via rt in a 4x1 way
#define LOOP_ELEM(n, name, xmr, ymr, delta, rt, r)\
    ULABEL(name##n)\
    VMOVAP(DEREF_OFF(xmr, delta), rt)\
    VMULADDP(DEREF_OFF(ymr, delta), rt , r, rt)\

#define UNROLL_SIZE 8
#define ELEM_SIZE 8
#define MAIN_LOOP_BODY()\
    LOOP_ELEM(8, dotloop, VAR(x), VAR(y), (7 * MAX_TAIL_SIZE * ELEM_SIZE), YMM(1), YMM(0)) \
    LOOP_ELEM(7, dotloop, VAR(x), VAR(y), (6 * MAX_TAIL_SIZE * ELEM_SIZE), YMM(3), YMM(2)) \
    LOOP_ELEM(6, dotloop, VAR(x), VAR(y), (5 * MAX_TAIL_SIZE * ELEM_SIZE), YMM(5), YMM(4)) \
    LOOP_ELEM(5, dotloop, VAR(x), VAR(y), (4 * MAX_TAIL_SIZE * ELEM_SIZE), YMM(7), YMM(6)) \
    LOOP_ELEM(4, dotloop, VAR(x), VAR(y), (3 * MAX_TAIL_SIZE * ELEM_SIZE), YMM(9), YMM(8)) \
    LOOP_ELEM(3, dotloop, VAR(x), VAR(y), (2 * MAX_TAIL_SIZE * ELEM_SIZE), YMM(11), YMM(10)) \
    LOOP_ELEM(2, dotloop, VAR(x), VAR(y), (1 * MAX_TAIL_SIZE * ELEM_SIZE), YMM(13), YMM(12)) \
    LOOP_ELEM(1, dotloop, VAR(x), VAR(y), (0 * MAX_TAIL_SIZE * ELEM_SIZE), YMM(15), YMM(14)) \
    ADD(IMM((UNROLL_SIZE * MAX_TAIL_SIZE * ELEM_SIZE)), VAR(x))\
    ADD(IMM((UNROLL_SIZE * MAX_TAIL_SIZE * ELEM_SIZE)), VAR(y))\
    SUB(IMM(UNROLL_SIZE), VAR(iter))

#define COLLAPSE_PARTIALS()\
    R_VADDP(YMM(0), YMM(2), YMM(1))\
    R_VADDP(YMM(4), YMM(6), YMM(3))\
    R_VADDP(YMM(8), YMM(10), YMM(5))\
    R_VADDP(YMM(12), YMM(14), YMM(7))\
    R_VADDP(YMM(1), YMM(3), YMM(0))\
    R_VADDP(YMM(5), YMM(7), YMM(2))\
    VHADDP(YMM(0), YMM(2), YMM(8))\
    VEXTRACTHIGH(YMM(8), XMM(9))\
    ASM(vzeroupper)\
    R_VADDP(XMM(8), XMM(9), XMM(10))\
    ASM(movhlps XMM(10), XMM(11))\
    R_VADDS(XMM(10), XMM(11), XMM(12))\
    M_VADDS(DEREF(VAR(rho)), XMM(12))\
    VMOVS(XMM(12), DEREF(VAR(rho)))
/*
  Template dotv kernel implementation

  This function contains a template implementation for a double-precision
  complex kernel, coded in C, which can serve as the starting point for one
  to write an optimized kernel on an arbitrary architecture. (We show a
  template implementation for only double-precision complex because the
  templates for the other three floating-point types would be similar, with
  the real instantiations being noticeably simpler due to the disappearance
  of conjugation in the real domain.)

  This kernel performs an inner (dot) product operation:

    rho := conjx( x^T ) * conjy( y )

  where x and y are vectors of length n and rho is a scalar.

  Parameters:

  - conjx:  Compute with conjugated values of x?
  - conjy:  Compute with conjugated values of y?
  - n:      The number of elements in vectors x and y.
  - x:      The address of vector x.
  - incx:   The vector increment of x. incx should be unit unless the
            implementation makes special accomodation for non-unit values.
  - y:      The address of vector y.
  - incy:   The vector increment of y. incy should be unit unless the
            implementation makes special accomodation for non-unit values.
  - rho:    The address of the output scalar.

  This template code calls the reference implementation if any of the
  following conditions are true:

  - Either of the strides incx or incy is non-unit.
  - Vectors x and y are unaligned with different offsets.

  If the vectors are aligned, or unaligned by the same offset, then optimized
  code can be used for the bulk of the computation. This template shows how
  the front-edge case can be handled so that the remaining computation is
  aligned. (This template guarantees alignment to be BLIS_SIMD_ALIGN_SIZE,
  which is defined in bli_config.h.)

  Additional things to consider:

  - While four combinations of possible values of conjx and conjy exist, we
    implement only conjugation on x explicitly; we induce the other two cases
    by toggling the effective conjugation on x and then conjugating the dot
    product result.
  - Because conjugation disappears in the real domain, real instances of
    this kernel can safely ignore the values of any conjugation parameters,
    thereby simplifying the implementation.

  For more info, please refer to the BLIS website and/or contact the
  blis-devel mailing list.

  -FGVZ
*/

void bli_sdotv_opt_var1
     (
       conj_t    conjx,
       conj_t    conjy,
       dim_t     n,
       float*    x, inc_t incx,
       float*    y, inc_t incy,
       float*    rho,
       cntx_t*   cntx
     )
{
    /* Just call the reference implementation. */
    BLIS_SDOTV_KERNEL_REF
    (
      conjx,
      conjy,
      n,
      x, incx,
      y, incy,
      rho,
      cntx
    );
}

#define SSE(instr) instr##d
#define MAX_TAIL_SIZE 4

void bli_ddotv_opt_var1
     (
       conj_t conjx,
       conj_t conjy,
       dim_t n,
       double* restrict x, inc_t incx,
       double* restrict y, inc_t incy,
       double* restrict rho,
       cntx_t* cntx
     )
{
    bool_t use_ref = FALSE;

    dim_t register n_pre = 0;
    dim_t register n_iter = 0;
    dim_t register n_post = 0;
    uint64_t register jump_tmp1;// __asm__ ("r14");
    uint64_t register jump_tmp2;// __asm__ ("r15");

    *rho = 0;
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
        BLIS_DDOTV_KERNEL_REF
        (
          conjx,
          conjy,
          n,
          x, incx,
          y, incy,
          rho,
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
     JZ(UNIQ(predot0))
     TAIL(predot, VAR(x), VAR(y), 3, 1, 7, 5, 11, 9, VAR(pre))
     ASM(vzeroall)
     ALIGN16
     ULABEL(dotloop_start)
     TEST(VAR(iter), VAR(iter))
     JLE(UNIQ(dotloop0))
     CMP(IMM(UNROLL_SIZE), VAR(iter))
     JL(UNIQ(dotloop_tail))
     MAIN_LOOP_BODY()
     JMP(UNIQ(dotloop_start))
     ULABEL(dotloop_tail)
     DUFFJMP(dotloop, VAR(iter))
     ULABEL(dotloop0)
     COLLAPSE_PARTIALS()
     TEST(VAR(post), VAR(post))
     JZ(UNIQ(postdot0))
     "\n# memory overshoot (in negative) * 2^(lg(sizeof double) + lg(max_tail_size))\n\t"
     SAL(IMM(3 + 2), VAR(iter))
     ADD(VAR(iter), VAR(x))
     ADD(VAR(iter), VAR(y))
     TAIL(postdot, VAR(x), VAR(y), 3, 1, 7, 5, 11, 9, VAR(post))
     JUMPTABLES()
     JUMPTABLE4(predot)
     JUMPTABLE8(dotloop)
     JUMPTABLE4(postdot)
     END_JUMPTABLES()
     : [pre] "+r" (n_pre),
       [iter] "+r" (n_iter),
       [post] "+r" (n_post),
       [x] "+r" (x),
       [y] "+r" (y),
       [jump_tmp1] "=r" (jump_tmp1),
       [jump_tmp2] "=r" (jump_tmp2),
       [rho] "=r" (rho)
     :
     : "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8",
       "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15", "memory"
     );
    return;
}
#undef TYPE_LETTER


void bli_cdotv_opt_var1
     (
       conj_t    conjx,
       conj_t    conjy,
       dim_t     n,
       scomplex* x, inc_t incx,
       scomplex* y, inc_t incy,
       scomplex* rho,
       cntx_t*   cntx
     )
{
    /* Just call the reference implementation. */
    BLIS_CDOTV_KERNEL_REF
    (
      conjx,
      conjy,
      n,
      x, incx,
      y, incy,
      rho,
      cntx
    );
}



void bli_zdotv_opt_var1
     (
       conj_t    conjx,
       conj_t    conjy,
       dim_t     n,
       dcomplex* x, inc_t incx,
       dcomplex* y, inc_t incy,
       dcomplex* rho,
       cntx_t*   cntx
     )
{
    BLIS_ZDOTV_KERNEL_REF
    (
      conjx,
      conjy,
      n,
      x, incx,
      y, incy,
      rho,
      cntx
    );
}

