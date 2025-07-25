; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -passes=loop-vectorize \
; RUN: -force-tail-folding-style=data-with-evl \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -mtriple=riscv64 -mattr=+v -S < %s | FileCheck --check-prefix=IF-EVL %s

; RUN: opt -passes=loop-vectorize \
; RUN: -force-tail-folding-style=none \
; RUN: -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue \
; RUN: -mtriple=riscv64 -mattr=+v -S < %s | FileCheck --check-prefix=NO-VP %s

; FIXME: interleaved accesses are not supported yet with predicated vectorization.
define void @interleave(ptr noalias %a, ptr noalias %b, i64 %N) {
; IF-EVL-LABEL: @interleave(
; IF-EVL-NEXT:  entry:
; IF-EVL-NEXT:    br i1 false, label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; IF-EVL:       vector.ph:
; IF-EVL-NEXT:    [[TMP4:%.*]] = call i64 @llvm.vscale.i64()
; IF-EVL-NEXT:    [[TMP5:%.*]] = mul nuw i64 [[TMP4]], 4
; IF-EVL-NEXT:    [[TMP6:%.*]] = sub i64 [[TMP5]], 1
; IF-EVL-NEXT:    [[N_RND_UP:%.*]] = add i64 [[N:%.*]], [[TMP6]]
; IF-EVL-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N_RND_UP]], [[TMP5]]
; IF-EVL-NEXT:    [[N_VEC:%.*]] = sub i64 [[N_RND_UP]], [[N_MOD_VF]]
; IF-EVL-NEXT:    [[TRIP_COUNT_MINUS_1:%.*]] = sub i64 [[N]], 1
; IF-EVL-NEXT:    [[TMP7:%.*]] = call i64 @llvm.vscale.i64()
; IF-EVL-NEXT:    [[TMP8:%.*]] = mul nuw i64 [[TMP7]], 4
; IF-EVL-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[TRIP_COUNT_MINUS_1]], i64 0
; IF-EVL-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 4 x i64> [[BROADCAST_SPLATINSERT]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; IF-EVL-NEXT:    br label [[VECTOR_BODY:%.*]]
; IF-EVL:       vector.body:
; IF-EVL-NEXT:    [[EVL_BASED_IV:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_EVL_NEXT:%.*]], [[VECTOR_BODY]] ]
; IF-EVL-NEXT:    [[BROADCAST_SPLATINSERT1:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[EVL_BASED_IV]], i64 0
; IF-EVL-NEXT:    [[BROADCAST_SPLAT2:%.*]] = shufflevector <vscale x 4 x i64> [[BROADCAST_SPLATINSERT1]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; IF-EVL-NEXT:    [[TMP9:%.*]] = call <vscale x 4 x i64> @llvm.stepvector.nxv4i64()
; IF-EVL-NEXT:    [[TMP10:%.*]] = add <vscale x 4 x i64> zeroinitializer, [[TMP9]]
; IF-EVL-NEXT:    [[VEC_IV:%.*]] = add <vscale x 4 x i64> [[BROADCAST_SPLAT2]], [[TMP10]]
; IF-EVL-NEXT:    [[TMP11:%.*]] = icmp ule <vscale x 4 x i64> [[VEC_IV]], [[BROADCAST_SPLAT]]
; IF-EVL-NEXT:    [[TMP12:%.*]] = getelementptr inbounds [2 x i32], ptr [[B:%.*]], i64 [[EVL_BASED_IV]], i32 0
; IF-EVL-NEXT:    [[INTERLEAVED_MASK:%.*]] = call <vscale x 8 x i1> @llvm.vector.interleave2.nxv8i1(<vscale x 4 x i1> [[TMP11]], <vscale x 4 x i1> [[TMP11]])
; IF-EVL-NEXT:    [[WIDE_MASKED_VEC:%.*]] = call <vscale x 8 x i32> @llvm.masked.load.nxv8i32.p0(ptr [[TMP12]], i32 4, <vscale x 8 x i1> [[INTERLEAVED_MASK]], <vscale x 8 x i32> poison)
; IF-EVL-NEXT:    [[STRIDED_VEC:%.*]] = call { <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.vector.deinterleave2.nxv8i32(<vscale x 8 x i32> [[WIDE_MASKED_VEC]])
; IF-EVL-NEXT:    [[WIDE_MASKED_GATHER3:%.*]] = extractvalue { <vscale x 4 x i32>, <vscale x 4 x i32> } [[STRIDED_VEC]], 0
; IF-EVL-NEXT:    [[WIDE_MASKED_GATHER5:%.*]] = extractvalue { <vscale x 4 x i32>, <vscale x 4 x i32> } [[STRIDED_VEC]], 1
; IF-EVL-NEXT:    [[TMP26:%.*]] = add nsw <vscale x 4 x i32> [[WIDE_MASKED_GATHER5]], [[WIDE_MASKED_GATHER3]]
; IF-EVL-NEXT:    [[TMP27:%.*]] = getelementptr inbounds i32, ptr [[A:%.*]], i64 [[EVL_BASED_IV]]
; IF-EVL-NEXT:    [[TMP29:%.*]] = getelementptr inbounds i32, ptr [[TMP27]], i32 0
; IF-EVL-NEXT:    call void @llvm.masked.store.nxv4i32.p0(<vscale x 4 x i32> [[TMP26]], ptr [[TMP29]], i32 4, <vscale x 4 x i1> [[TMP11]])
; IF-EVL-NEXT:    [[INDEX_EVL_NEXT]] = add i64 [[EVL_BASED_IV]], [[TMP8]]
; IF-EVL-NEXT:    [[TMP14:%.*]] = icmp eq i64 [[INDEX_EVL_NEXT]], [[N_VEC]]
; IF-EVL-NEXT:    br i1 [[TMP14]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; IF-EVL:       middle.block:
; IF-EVL-NEXT:    br label [[FOR_COND_CLEANUP:%.*]]
; IF-EVL:       scalar.ph:
; IF-EVL-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ]
; IF-EVL-NEXT:    br label [[FOR_BODY:%.*]]
; IF-EVL:       for.body:
; IF-EVL-NEXT:    [[IV:%.*]] = phi i64 [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ], [ [[IV_NEXT:%.*]], [[FOR_BODY]] ]
; IF-EVL-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x i32], ptr [[B]], i64 [[IV]], i32 0
; IF-EVL-NEXT:    [[TMP34:%.*]] = load i32, ptr [[ARRAYIDX]], align 4
; IF-EVL-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x i32], ptr [[B]], i64 [[IV]], i32 1
; IF-EVL-NEXT:    [[TMP35:%.*]] = load i32, ptr [[ARRAYIDX2]], align 4
; IF-EVL-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP35]], [[TMP34]]
; IF-EVL-NEXT:    [[ARRAYIDX4:%.*]] = getelementptr inbounds i32, ptr [[A]], i64 [[IV]]
; IF-EVL-NEXT:    store i32 [[ADD]], ptr [[ARRAYIDX4]], align 4
; IF-EVL-NEXT:    [[IV_NEXT]] = add nuw nsw i64 [[IV]], 1
; IF-EVL-NEXT:    [[EXITCOND_NOT:%.*]] = icmp eq i64 [[IV_NEXT]], [[N]]
; IF-EVL-NEXT:    br i1 [[EXITCOND_NOT]], label [[FOR_COND_CLEANUP]], label [[FOR_BODY]], !llvm.loop [[LOOP3:![0-9]+]]
; IF-EVL:       for.cond.cleanup:
; IF-EVL-NEXT:    ret void
;
; NO-VP-LABEL: @interleave(
; NO-VP-NEXT:  entry:
; NO-VP-NEXT:    [[TMP0:%.*]] = call i64 @llvm.vscale.i64()
; NO-VP-NEXT:    [[TMP1:%.*]] = mul nuw i64 [[TMP0]], 4
; NO-VP-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[N:%.*]], [[TMP1]]
; NO-VP-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; NO-VP:       vector.ph:
; NO-VP-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vscale.i64()
; NO-VP-NEXT:    [[TMP3:%.*]] = mul nuw i64 [[TMP2]], 4
; NO-VP-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N]], [[TMP3]]
; NO-VP-NEXT:    [[N_VEC:%.*]] = sub i64 [[N]], [[N_MOD_VF]]
; NO-VP-NEXT:    [[TMP4:%.*]] = call i64 @llvm.vscale.i64()
; NO-VP-NEXT:    [[TMP8:%.*]] = mul nuw i64 [[TMP4]], 4
; NO-VP-NEXT:    br label [[VECTOR_BODY:%.*]]
; NO-VP:       vector.body:
; NO-VP-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; NO-VP-NEXT:    [[TMP13:%.*]] = getelementptr inbounds [2 x i32], ptr [[B:%.*]], i64 [[INDEX]], i32 0
; NO-VP-NEXT:    [[WIDE_VEC1:%.*]] = load <vscale x 8 x i32>, ptr [[TMP13]], align 4
; NO-VP-NEXT:    [[STRIDED_VEC2:%.*]] = call { <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.vector.deinterleave2.nxv8i32(<vscale x 8 x i32> [[WIDE_VEC1]])
; NO-VP-NEXT:    [[TMP18:%.*]] = extractvalue { <vscale x 4 x i32>, <vscale x 4 x i32> } [[STRIDED_VEC2]], 0
; NO-VP-NEXT:    [[TMP19:%.*]] = extractvalue { <vscale x 4 x i32>, <vscale x 4 x i32> } [[STRIDED_VEC2]], 1
; NO-VP-NEXT:    [[TMP21:%.*]] = add nsw <vscale x 4 x i32> [[TMP19]], [[TMP18]]
; NO-VP-NEXT:    [[TMP22:%.*]] = getelementptr inbounds i32, ptr [[A:%.*]], i64 [[INDEX]]
; NO-VP-NEXT:    [[TMP24:%.*]] = getelementptr inbounds i32, ptr [[TMP22]], i32 0
; NO-VP-NEXT:    store <vscale x 4 x i32> [[TMP21]], ptr [[TMP24]], align 4
; NO-VP-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], [[TMP8]]
; NO-VP-NEXT:    [[TMP28:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; NO-VP-NEXT:    br i1 [[TMP28]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; NO-VP:       middle.block:
; NO-VP-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[N]], [[N_VEC]]
; NO-VP-NEXT:    br i1 [[CMP_N]], label [[FOR_COND_CLEANUP:%.*]], label [[SCALAR_PH]]
; NO-VP:       scalar.ph:
; NO-VP-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], [[MIDDLE_BLOCK]] ], [ 0, [[ENTRY:%.*]] ]
; NO-VP-NEXT:    br label [[FOR_BODY:%.*]]
; NO-VP:       for.body:
; NO-VP-NEXT:    [[IV:%.*]] = phi i64 [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ], [ [[IV_NEXT:%.*]], [[FOR_BODY]] ]
; NO-VP-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x i32], ptr [[B]], i64 [[IV]], i32 0
; NO-VP-NEXT:    [[TMP29:%.*]] = load i32, ptr [[ARRAYIDX]], align 4
; NO-VP-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x i32], ptr [[B]], i64 [[IV]], i32 1
; NO-VP-NEXT:    [[TMP30:%.*]] = load i32, ptr [[ARRAYIDX2]], align 4
; NO-VP-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP30]], [[TMP29]]
; NO-VP-NEXT:    [[ARRAYIDX4:%.*]] = getelementptr inbounds i32, ptr [[A]], i64 [[IV]]
; NO-VP-NEXT:    store i32 [[ADD]], ptr [[ARRAYIDX4]], align 4
; NO-VP-NEXT:    [[IV_NEXT]] = add nuw nsw i64 [[IV]], 1
; NO-VP-NEXT:    [[EXITCOND_NOT:%.*]] = icmp eq i64 [[IV_NEXT]], [[N]]
; NO-VP-NEXT:    br i1 [[EXITCOND_NOT]], label [[FOR_COND_CLEANUP]], label [[FOR_BODY]], !llvm.loop [[LOOP3:![0-9]+]]
; NO-VP:       for.cond.cleanup:
; NO-VP-NEXT:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [2 x i32], ptr %b, i64 %iv, i32 0
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds [2 x i32], ptr %b, i64 %iv, i32 1
  %1 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %arrayidx4 = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %add, ptr %arrayidx4, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !0

for.cond.cleanup:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 true}
