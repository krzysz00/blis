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
    INCMEM(4, delta)\
    INCMEM(5, delta)\
    ADD(delta, VAR(x))\

#define INCMEMS_CONST(delta)\
    INCMEMS(IMM(delta * ELEM_SIZE))

#define SCADOT(m, n, reg, tmp)\
    VMULADDS(DEREF_OFF(VAR(A##n), (-m * ELEM_SIZE)), XMM(15), XMM(reg), XMM(tmp))

#define ZERO_SCAL()\
    ZERO(YMM(8))\
    ZERO(YMM(9))\
    ZERO(YMM(10))\
    ZERO(YMM(11))\
    ZERO(YMM(12))\
    ZERO(YMM(13))

// safe since the registers are [0, 0, 0, res] by construction
// (the adds in the pseudo-FMA will copy in the [127:64] word from rN, not tmp, so it'll be zero)
#define MERGE_SCALAR_PART()\
    R_VADDP(YMM(0), YMM(8), YMM(0))\
    R_VADDP(YMM(1), YMM(9), YMM(1))\
    R_VADDP(YMM(2), YMM(10), YMM(2))\
    R_VADDP(YMM(3), YMM(11), YMM(3))\
    R_VADDP(YMM(4), YMM(12), YMM(4))\
    R_VADDP(YMM(5), YMM(13), YMM(5))

#define SCAL_ELEM(m, typ)\
    ULABEL(typ##m)\
    VMOVS(DEREF_OFF(VAR(x), (-m * ELEM_SIZE)), XMM(15))\
    SCADOT(m, 0, 8, 14)\
    SCADOT(m, 1, 9, 14)\
    SCADOT(m, 2, 10, 14)\
    SCADOT(m, 3, 11, 14)\
    SCADOT(m, 4, 12, 14)\
    SCADOT(m, 5, 13, 14)\

#define SCAL_SEGMENT(typ)\
    SHL(IMM(3), VAR(typ))\
    INCMEMS(VAR(typ))\
    SHR(IMM(3), VAR(typ))\
    ZERO_SCAL()\
    DUFFJMP_GEN(typ##dot, VAR(typ), VAR(jump_tmp), VAR(typ))\
    SCAL_ELEM(3, typ##dot)\
    SCAL_ELEM(2, typ##dot)\
    SCAL_ELEM(1, typ##dot)\
    MERGE_SCALAR_PART()\
    ULABEL(typ##dot0)

#define VECDOT(m, n, tmp)\
    VMULADDP(DEREF_OFF(VAR(A##n), ((m - 1) * MAX_TAIL_SIZE * ELEM_SIZE)), YMM(15), YMM(n), YMM(tmp))

#define DOT_ELEM(m, dprefetch)\
    ULABEL(dotloop##m)\
    VMOVAP(DEREF_OFF(VAR(x), ((m - 1) * MAX_TAIL_SIZE * ELEM_SIZE)), YMM(15))\
    PREFETCH(1, DEREF_OFF(VAR(x), (dprefetch * MAX_TAIL_SIZE * ELEM_SIZE)))\
    VECDOT(m, 0, 8)\
    VECDOT(m, 1, 9)\
    VECDOT(m, 2, 10)\
    VECDOT(m, 3, 11)\
    VECDOT(m, 4, 12)\
    VECDOT(m, 5, 13)

#define MAIN_LOOP_BODY()\
    DOT_ELEM(8, 6)\
    DOT_ELEM(7, 5)\
    DOT_ELEM(6, 4)\
    DOT_ELEM(5, 3)\
    DOT_ELEM(4, 2)\
    DOT_ELEM(3, 1)\
    DOT_ELEM(2, 0)\
    DOT_ELEM(1, 8)\
    INCMEMS_CONST((UNROLL_SIZE * MAX_TAIL_SIZE))\
    SUB(IMM(UNROLL_SIZE), VAR(iter))
// there's no 0 because that would index too far back

#define COLLAPSE_TRANSPOSE()\
    R_VADDP(YMM(0), YMM(1), YMM(0))\
    R_VADDP(YMM(2), YMM(3), YMM(2))\
    R_VADDP(YMM(4), YMM(5), YMM(4))\
    R_VADDP(YMM(12), YMM(13), YMM(12))\
    R_VADDP(YMM(0), YMM(2), YMM(0))\
    R_VADDP(YMM(4), YMM(12), YMM(4))

#define VERT_SIZE 6
#define UNROLL_SIZE 8
#define ELEM_SIZE 8
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

#define SSE(instr) instr##d
#define MAX_TAIL_SIZE 4

/*
  Template dotxf kernel implementation

  This function contains a template implementation for a double-precision
  complex kernel, coded in C, which can serve as the starting point for one
  to write an optimized kernel on an arbitrary architecture. (We show a
  template implementation for only double-precision complex because the
  templates for the other three floating-point types would be similar, with
  the real instantiations being noticeably simpler due to the disappearance
  of conjugation in the real domain.)

  This kernel performs the following gemv-like operation:

    y := beta * y + alpha * conjat( A^T ) * conjx( x )

  where A is an m x b_n matrix, x is a vector of length m, y is a vector
  of length b_n, and alpha and beta are scalars. The operation is performed
  as a series of fused dotxv operations, and therefore A should be column-
  stored.

  Parameters:

  - conjat: Compute with conjugated values of A^T?
  - conjx:  Compute with conjugated values of x?
  - m:      The number of rows in matrix A.
  - b_n:    The number of columns in matrix A. Must be equal to or less than
            the fusing factor.
  - alpha:  The address of the scalar to be applied to A*x.
  - a:      The address of matrix A.
  - inca:   The row stride of A. inca should be unit unless the
            implementation makes special accomodation for non-unit values.
  - lda:    The column stride of A.
  - x:      The address of vector x.
  - incx:   The vector increment of x. incx should be unit unless the
            implementation makes special accomodation for non-unit values.
  - beta:   The address of the scalar to be applied to y.
  - y:      The address of vector y.
  - incy:   The vector increment of y.

  This template code calls the reference implementation if any of the
  following conditions are true:

  - Either of the strides inca or incx is non-unit.
  - The address of A, the second column of A, and x are unaligned with
    different offsets.

  If the first/second columns of A and address of x are aligned, or unaligned
  by the same offset, then optimized code can be used for the bulk of the
  computation. This template shows how the front-edge case can be handled so
  that the remaining computation is aligned. (This template guarantees
  alignment in the main loops to be BLIS_SIMD_ALIGN_SIZE, which is defined
  in bli_config.h.)

  Additional things to consider:

  - When optimizing, you should fully unroll the loops over b_n. This is the
    dimension across which we are fusing dotxv operations.
  - This template code chooses to call the reference implementation whenever
    b_n is less than the fusing factor, so as to avoid having to handle edge
    cases. One may choose to optimize this edge case, if desired.
  - Because conjugation disappears in the real domain, real instances of
    this kernel can safely ignore the values of any conjugation parameters,
    thereby simplifying the implementation.

  For more info, please refer to the BLIS website and/or contact the
  blis-devel mailing list.

  -FGVZ
*/

void bli_ddotxf_opt_6x4
     (
       conj_t    conjat,
       conj_t    conjx,
       dim_t     m,
       dim_t     b_n,
       double*   alpha,
       double*   a, inc_t inca, inc_t lda,
       double*   x, inc_t incx,
       double*   beta,
       double*   y, inc_t incy,
       cntx_t*   cntx
     )
{
    bool_t use_ref = FALSE;

    dim_t register m_pre = 0;

    if (bli_zero_dim1(b_n)) return;

    if (bli_zero_dim1(m)) {
        bli_dscalv(BLIS_NO_CONJUGATE, b_n,
                      beta, y, incy,
                      cntx);
        return;
    }
    // If there is anything that would interfere with our use of aligned
    // vector loads/stores, call the reference implementation.
    if (b_n < VERT_SIZE) {
        use_ref = TRUE;
    }
    else if (bli_has_nonunit_inc2(inca, incx)) {
        use_ref = TRUE;
    }
    else if (bli_is_unaligned_to( a,     BLIS_SIMD_ALIGN_SIZE) ||
             bli_is_unaligned_to( a+lda, BLIS_SIMD_ALIGN_SIZE) ||
             bli_is_unaligned_to( x,     BLIS_SIMD_ALIGN_SIZE)) {
        use_ref = TRUE;

        // If a, the second column of a, and x are unaligned by the same
        // offset, then we can still use an implementation that depends on
        // alignment for most of the operation.
        off_t off_a  = bli_offset_past_alignment(a, BLIS_SIMD_ALIGN_SIZE);
        off_t off_a2 = bli_offset_past_alignment(a+lda, BLIS_SIMD_ALIGN_SIZE);
        off_t off_x = bli_offset_past_alignment(x, BLIS_SIMD_ALIGN_SIZE);

        if (off_a == off_a2 && off_a == off_x) {
            use_ref = FALSE;
            m_pre   = (BLIS_SIMD_ALIGN_SIZE - off_x) / ELEM_SIZE;
        }
    }

    // Call the reference implementation if needed.
    if (use_ref == TRUE) {
        BLIS_DDOTXF_KERNEL_REF
            (
             conjat,
             conjx,
             m,
             b_n,
             alpha,
             a, inca, lda,
             x, incx,
             beta,
             y, incy,
             cntx
             );
        return;
    }

    double stor[8] __attribute__((aligned (32)));

    if (incy == 1) {
        memcpy(stor, y, VERT_SIZE * sizeof(double));
    }
    else {
        double *dy = y;
        double *dstor = stor;
        for (int i = 0; i < VERT_SIZE; i++) {
            *dstor = *dy;
            dstor++;
            dy += incy;
        }
    }
    stor[6] = 0.0;
    stor[7] = 0.0;

    dim_t register m_post = (m - m_pre) % MAX_TAIL_SIZE;
    dim_t register m_iter = (m - m_pre - m_post) / MAX_TAIL_SIZE;
    dim_t register jump_tmp = 0;
    double register *a1 = a + 1 * lda;
    double register *a2 = a + 2 * lda;
    double register *a3 = a + 3 * lda;
    double register *a4 = a + 4 * lda;
    double register *a5 = a + 5 * lda;
    //    printf("n: %ld, iter: %ld, pre: %ld, post: %ld, addr: %llx, rho: %llx\n", n, n_iter, n_pre, n_post, (uint64_t)x, (uint64_t)rho);
    // registers are dispersed in pairs [x_tmp, accum]
    // current loop has four pairs, repeated once
    __asm__ volatile
    (
     ASM(vzeroall)
     TEST(VAR(pre), VAR(pre))
     JZ(UNIQ(predot0))
     SCAL_SEGMENT(pre)
     ALIGN16
     ULABEL(dotloop_start)
     TEST(VAR(iter), VAR(iter))
     JLE(UNIQ(dotloop0))
     CMP(IMM(UNROLL_SIZE), VAR(iter))
     JL(UNIQ(dotloop_tail))
     MAIN_LOOP_BODY()
     JMP(UNIQ(dotloop_start))
     ULABEL(dotloop_tail)
     DUFFJMP_GEN(dotloop, VAR(iter), VAR(jump_tmp), VAR(pre))
     ULABEL(dotloop0)
     TEST(VAR(post), VAR(post))
     JZ(UNIQ(postdot0))
     "\n# memory overshoot (in negative) * 2^(lg(sizeof double) + lg(max_tail_size))\n\t"
     SAL(IMM(3 + 2), VAR(iter))
     INCMEMS(VAR(iter))
     SCAL_SEGMENT(post)
     VTRANSPOSED(YMM(0), YMM(1), YMM(2), YMM(3), YMM(8), YMM(9), YMM(10), YMM(11))
     ZERO(YMM(12))
     ZERO(YMM(13))
     VTRANSPOSED(YMM(4), YMM(5), YMM(12), YMM(13), YMM(8), YMM(9), YMM(10), YMM(11))
     COLLAPSE_TRANSPOSE()
     MOVQ(VAR(alpha), VAR(jump_tmp))
     VBROADCASTS(DEREF(VAR(jump_tmp)), YMM(14))
     MOVQ(VAR(beta), VAR(x))
     VBROADCASTS(DEREF(VAR(x)), YMM(15))
     VMOVAP(DEREF(VAR(y)), YMM(1))
     VMOVAP(DEREF_OFF(VAR(y), (MAX_TAIL_SIZE * ELEM_SIZE)), YMM(5))
     VMULP(YMM(0), YMM(14), YMM(0))
     VMULP(YMM(1), YMM(15), YMM(1))
     R_VADDP(YMM(0), YMM(1), YMM(1))
     VMULP(YMM(4), YMM(14), YMM(4))
     VMULP(YMM(5), YMM(15), YMM(5))
     R_VADDP(YMM(4), YMM(5), YMM(5))
     VMOVAP(YMM(1), DEREF(VAR(y)))
     VMOVAP(YMM(5), DEREF_OFF(VAR(y), (MAX_TAIL_SIZE * ELEM_SIZE)))
     JUMPTABLES()
     JUMPTABLE4(predot)
     JUMPTABLE8(dotloop)
     JUMPTABLE4(postdot)
     END_JUMPTABLES()
     : [pre] "+r" (m_pre),
       [iter] "+r" (m_iter),
       [post] "+r" (m_post),
       [x] "+r" (x),
       [A0] "+r" (a),
       [A1] "+r" (a1),
       [A2] "+r" (a2),
       [A3] "+r" (a3),
       [A4] "+r" (a4),
       [A5] "+r" (a5),
       [jump_tmp] "=r" (jump_tmp)
     : [y] "r" (stor),
       [alpha] "rm" (alpha),
       [beta] "rm" (beta)
     : "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8",
       "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15", "memory"
     );
    if (incy == 1) {
        memcpy(y, stor, VERT_SIZE * sizeof(double));
    }
    else {
        double *dy = y;
        double *dstor = stor;
        for (int i = 0; i < VERT_SIZE; i++) {
            *dy = *dstor;
            dstor++;
            dy += incy;
        }
    }
    return;
}
