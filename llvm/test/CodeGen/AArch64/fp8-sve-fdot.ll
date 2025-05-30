; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 5
; RUN: llc -mattr=+sve2,+fp8dot2,+fp8dot4  < %s | FileCheck %s
; RUN: llc -mattr=+sme,+ssve-fp8dot2,+ssve-fp8dot4 --force-streaming < %s | FileCheck %s

target triple = "aarch64-linux"

define <vscale x 4 x float> @fdot_4way(<vscale x 4 x float> %a, <vscale x 16 x i8> %s1, <vscale x 16 x i8> %s2) {
; CHECK-LABEL: fdot_4way:
; CHECK:       // %bb.0:
; CHECK-NEXT:    fdot z0.s, z1.b, z2.b
; CHECK-NEXT:    ret
    %r = call <vscale x 4 x float> @llvm.aarch64.sve.fp8.fdot.nxv4f32(<vscale x 4 x float> %a, <vscale x 16 x i8> %s1, <vscale x 16 x i8> %s2)
    ret <vscale x 4 x float> %r
}

define <vscale x 8 x half> @fdot_2way(<vscale x 8 x half> %a, <vscale x 16 x i8> %s1, <vscale x 16 x i8> %s2) {
; CHECK-LABEL: fdot_2way:
; CHECK:       // %bb.0:
; CHECK-NEXT:    fdot z0.h, z1.b, z2.b
; CHECK-NEXT:    ret
    %r = call <vscale x 8 x half> @llvm.aarch64.sve.fp8.fdot.nxv8f16(<vscale x 8 x half> %a, <vscale x 16 x i8> %s1, <vscale x 16 x i8> %s2)
    ret <vscale x 8 x half> %r
}

define <vscale x 4 x float> @fdot_4way_lane(<vscale x 4 x float> %a, <vscale x 16 x i8> %s1, <vscale x 16 x i8> %s2) {
; CHECK-LABEL: fdot_4way_lane:
; CHECK:       // %bb.0:
; CHECK-NEXT:    fdot z0.s, z1.b, z2.b[3]
; CHECK-NEXT:    ret
    %r = call <vscale x 4 x float> @llvm.aarch64.sve.fp8.fdot.lane.nxv4f32(<vscale x 4 x float> %a, <vscale x 16 x i8> %s1, <vscale x 16 x i8> %s2, i32 3)
    ret <vscale x 4 x float> %r
}

define <vscale x 8 x half> @fdot_2way_lane(<vscale x 8 x half> %a, <vscale x 16 x i8> %s1, <vscale x 16 x i8> %s2) {
; CHECK-LABEL: fdot_2way_lane:
; CHECK:       // %bb.0:
; CHECK-NEXT:    fdot z0.h, z1.b, z2.b[5]
; CHECK-NEXT:    ret
    %r = call <vscale x 8 x half> @llvm.aarch64.sve.fp8.fdot.lane.nxv8f16(<vscale x 8 x half> %a, <vscale x 16 x i8> %s1, <vscale x 16 x i8> %s2, i32 5)
    ret <vscale x 8 x half> %r
}
