; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 5
; RUN: llc < %s -mtriple=riscv64 -mattr=+v,+zvl16384b | FileCheck %s

define void @a(ptr %0, ptr %1) {
; CHECK-LABEL: a:
; CHECK:       # %bb.0:
; CHECK-NEXT:    li a2, 1024
; CHECK-NEXT:    vsetvli zero, a2, e8, mf2, ta, ma
; CHECK-NEXT:    vle8.v v8, (a1)
; CHECK-NEXT:    vse8.v v8, (a0)
; CHECK-NEXT:    addi a1, a1, 1024
; CHECK-NEXT:    vle8.v v8, (a1)
; CHECK-NEXT:    addi a0, a0, 1024
; CHECK-NEXT:    vse8.v v8, (a0)
; CHECK-NEXT:    ret
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %0, ptr align 4 %1, i64 2048, i1 false)
  ret void
}
