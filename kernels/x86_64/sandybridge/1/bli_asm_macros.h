#ifndef BLIS_AVX512_MACROS_H
#define BLIS_AVX512_MACROS_H

//
// Assembly macros to make assembly in general less painful
//

#define COMMENT_BEGIN "#"
#define COMMENT_END

#define STRINGIFY(...) #__VA_ARGS__
#define ASM(...) STRINGIFY(__VA_ARGS__) "\n\t"
#define LABEL(label) STRINGIFY(label) ":\n\t"
#define UNIQ(blah) blah##_%=
#define ULABEL(label) STRINGIFY(label) "_%=:\n\t"

#define XMM(x) %%xmm##x
#define YMM(x) %%ymm##x
#define EAX %%eax
#define EBX %%ebx
#define ECX %%ecx
#define EDX %%edx
#define EBP %%ebp
#define EDI %%edi
#define ESI %%esi
#define RAX %%rax
#define RBX %%rbx
#define RCX %%rcx
#define RDX %%rdx
#define RBP %%rbp
#define RDI %%rdi
#define RSI %%rsi
#define R(x) %%r##x
#define R8 %%r8
#define R9 %%r9
#define R10 %%r10
#define R11 %%r11
#define R12 %%r12
#define R13 %%r13
#define R14 %%r14
#define R15 %%r15
#define RD(x) %% r##x##d
#define R8D %%r8d
#define R9D %%r9d
#define R10D %%r10d
#define R11D %%r11d
#define R12D %%r12d
#define R13D %%r13d
#define R14D %%r14d
#define R15D %%r15d
#define RIP %%rip
#define IMM(x) $(x)
#define VAR(x) %[x]

#define DEREF_4(reg,off,scale,disp) disp(reg,off,scale)
#define DEREF_ARR(reg,off,scale) (reg,off,scale)
#define DEREF_OFF(reg,disp) disp(reg)
#define DEREF(reg) (reg)

#define ALIGN16 ASM(.p2align 4)
#define ALIGN32 ASM(.p2align 5)
#define RDTSC ASM(rdstc)
#define MOV(_0, _1) ASM(mov _0, _1)
#define MOVD(_0, _1) ASM(movd _0, _1)
#define MOVL(_0, _1) ASM(movl _0, _1)
#define MOVQ(_0, _1) ASM(movq _0, _1)
#define VMOVD(_0, _1) ASM(vmovd _0, _1)
#define VMOVQ(_0, _1) ASM(vmovq _0, _1)
#define VMOVS(_0, _1) ASM(SSE(vmovs) _0, _1)
#define CMP(_0, _1) ASM(cmp _0, _1)
#define AND(_0, _1) ASM(and _0, _1)
#define ADD(_0, _1) ASM(add _0, _1)
#define SUB(_0, _1) ASM(sub _0, _1)
#define SAL(_0, _1) ASM(sal _0, _1)
#define SHLX(_0, _1, _2) ASM(shlx _0, _1, _2)
#define SAR(_0, _1) ASM(sar _0, _1)
#define SAL1(_0) ASM(sal _0)
#define SAR1(_0) ASM(sar _0)
#define SHL(_0, _1) ASM(shl _0, _1)
#define LEA(_0, _1) ASM(lea _0, _1)
#define LEAQ(_0, _1) ASM(leaq _0, _1)
#define TEST(_0, _1) ASM(test _0, _1)
//#define DEC(_0) ASM(dec _0)
#define DEC(_0) SUB(_0, IMM(1))
#define JLE(_0) ASM(jle _0)
#define JL(_0) ASM(jl _0)
#define JNZ(_0) ASM(jnz _0)
#define JZ(_0) ASM(jz _0)
#define JNE(_0) ASM(jne _0)
#define JE(_0) ASM(je _0)
#define JNC(_0) ASM(jnc _0)
#define JC(_0) ASM(jc _0)
#define JMP(_0) ASM(jmp _0)
#define R_VADDS(_0, _1, _2) ASM(SSE(vadds) _0, _1, _2)
#define M_VADDS(_0, _1) ASM(SSE(adds) _0, _1)
#define R_VADDP(_0, _1, _2) ASM(SSE(vaddp) _0, _1, _2)
#define M_VADDP(_0, _1) ASM(SSE(addp) _0, _1)

#define VGATHERDP(_0, _1) ASM(SSE(vgatherdp) _0, _1)
#define VSCATTERDPS(_0, _1) ASM(SSE(vscatterdp) _0, _1)
#define VMULP(_0, _1, _2) ASM(SSE(vmulp) _0, _1, _2)
#define VPMULL(_0, _1, _2) ASM(SSE(vpmull) _0, _1, _2)
#define VPADDD(_0, _1, _2) ASM(SSE(vpadd) _0, _1, _2)
#define VPSLLD(_0, _1, _2) ASM(vpslld _0, _1, _2)
#define VPXORD(_0, _1, _2) ASM(vpxord _0, _1, _2)
#define VMULADDP(_0, _1, _2, _tmp) ASM(SSE(vmulp) _0, _1, _tmp)\
    ASM(SSE(vaddp) _2, _tmp, _2)
#define VMULADDS(_0, _1, _2, _tmp) ASM(SSE(vmuls) _0, _1, _tmp) \
    ASM(SSE(vadds) _2, _tmp, _2)
#define VMOVAP(_0, _1) ASM(SSE(vmovap) _0, _1)
#define VMOVUP(_0, _1) ASM(SSE(vmovup) _0, _1)
#define VBROADCASTS(_0, _1) ASM(SSE(vbroadcastss) _0, _1)
#define VPBROADCASTD(_0, _1) ASM(vpbroadcastd _0, _1)
#define VHADDP(_0, _1, _2) ASM(SSE(vhaddp) _0, _1, _2)
#define VEXTRACTHIGH(_0, _1) ASM(vextractf128 $1, _0, _1)

#define PREFETCH(LEVEL,ADDRESS) ASM(prefetcht##LEVEL ADDRESS)
#define PREFETCHW0(ADDRESS) ASM(prefetchw ADDRESS)
#define PREFETCHW1(ADDRESS) ASM(prefetchwt1 ADDRESS)
#define VGATHERPFDPS(LEVEL,ADDRESS) ASM(vgatherpf##LEVEL##dps ADDRESS)
#define VSCATTERPFDPS(LEVEL,ADDRESS) ASM(vscatterpf##LEVEL##dps ADDRESS)
#define VGATHERPFDPD(LEVEL,ADDRESS) ASM(vgatherpf##LEVEL##dpd ADDRESS)
#define VSCATTERPFDPD(LEVEL,ADDRESS) ASM(vscatterpf##LEVEL##dpd ADDRESS)

#define ZERO(r) ASM(vxorpd r, r, r)

#define DUFFJMP(nam, reg)\
    LEAQ(DEREF_OFF(RIP, UNIQ(nam##jumptable)), VAR(jump_tmp1))\
    ASM(movslq DEREF_ARR(VAR(jump_tmp1), reg, 4), VAR(jump_tmp2))\
    LEAQ((VAR(jump_tmp2), VAR(jump_tmp1)), VAR(jump_tmp1))\
    ASM(jmp *VAR(jump_tmp1))\

#define JUMPTABLES()\
    ".section .rodata.jumptables%=\n\t"

#define END_JUMPTABLES()\
    ".text\n\t"

#define JUMPTABLE4(nam)\
    ULABEL(nam##jumptable)\
    ASM(.long UNIQ(nam##0) - UNIQ(nam##jumptable), UNIQ(nam##1) - UNIQ(nam##jumptable),\
        UNIQ(nam##2) - UNIQ(nam##jumptable), UNIQ(nam##3) - UNIQ(nam##jumptable))

#define JUMPTABLE8(nam)\
    ULABEL(nam##jumptable)\
    ASM(.long UNIQ(nam##0) - UNIQ(nam##jumptable), UNIQ(nam##1) - UNIQ(nam##jumptable),\
        UNIQ(nam##2) - UNIQ(nam##jumptable), UNIQ(nam##3) - UNIQ(nam##jumptable),\
        UNIQ(nam##4) - UNIQ(nam##jumptable), UNIQ(nam##5) - UNIQ(nam##jumptable),\
        UNIQ(nam##6) - UNIQ(nam##jumptable), UNIQ(nam##7) - UNIQ(nam##jumptable))

#endif
