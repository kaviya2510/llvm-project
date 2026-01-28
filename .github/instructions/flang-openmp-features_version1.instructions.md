---
applyTo: flang/lib/Parser/openmp*,flang/lib/Semantics/*omp*,flang/lib/Lower/OpenMP/**/*,flang/lib/Optimizer/OpenMP/**/*,flang/include/flang/Parser/parse-tree.h,flang/include/flang/Lower/OpenMP*,flang/include/flang/Support/OpenMP*,flang/include/flang/Optimizer/OpenMP/**/*,mlir/include/mlir/Dialect/OpenMP/**/*,mlir/lib/Dialect/OpenMP/**/*,mlir/lib/Target/LLVMIR/Dialect/OpenMP/**/*,llvm/include/llvm/Frontend/OpenMP/**/*,llvm/lib/Frontend/OpenMP/**/*
---

Status
- PRs included: 2
- Last modified: 2026-01-28

# OpenMP Features (Version 1 maintained to reduce the conflict with newer versions)

# PART 1: REFERENCE FEATURE IMPLEMENTATION DATABASE

This section contains a reference database of OpenMP features that are implemented in the Flang/LLVM/MLIR toolchain. Each feature is listed with its corresponding OpenMP version, description, and implementation status.

---

## [Flang][OpenMP] Implement device clause lowering for target directive

Overview
- Purpose: Lower `device(<int-expr>)` on `!$omp target` so the evaluated device id flows Flang → MLIR OpenMP → LLVM IR and reaches the host launch (e.g., `__tgt_target_kernel`).
- Scope: `device` on `target`. Related target data-motion ops are noted under Follow-ups.

Spec Reference
- OpenMP 5.2 — Target constructs and `device` clause semantics (host-side device selection by integer expression).

Semantics
- Evaluate the `device` scalar integer expression on the host.
- Pass the resulting device id to the offload runtime. A value of `-1` maps to runtime “default device” policy (do not reinterpret here).
- If the clause is absent, existing default-device behavior applies.

Lowering (Flang → MLIR)
- Collect the Fortran `device` expression during clause processing (Semantics ensures it’s a scalar integer, host-evaluable).
- Attach it to `omp.target` as the dialect’s device operand/attribute with the expected integer type.

MLIR → LLVM IR Translation
- Ensure `omp.target` carries the device operand/attr.
- Translation extracts the device id and plumbs it to the OpenMP runtime launch via existing builder/runtime attributes, culminating in the host call (e.g., `__tgt_target_kernel`).
- Validation helper: apply device checks to ops that support the clause (e.g., `omp.target`, selectively `omp.target_data`/`omp.target_update`).

Runtime Interaction
- No new API surface; encode device id in existing launch metadata/arguments rather than changing call signatures.
- Preserve runtime handling of default device selection when the id is `-1` or when the clause is omitted.

Testing
- MLIR translation tests: verify `omp.target device(%cst)` lowers to IR that passes the expected device id to the offload runtime.
- Add coverage for constants and SSA-propagated integers.
- Documentation: update `flang/docs/OpenMPSupport.md` with support status.

Diagnostics
- Emit clear errors when `device` appears where unsupported (until coverage is extended) or when the expression is not a scalar integer.
- Keep behavior consistent with “default device” when `-1` or no clause is used.

Follow-ups
- Unify device propagation/validation for `target enter data` / `target exit data` once legacy tests using hard-coded `-1` are updated.
- Add end-to-end tests (Fortran → MLIR → LLVM IR) for variable device ids, nested targets, and interaction with default device ICVs.
- Ensure uniform handling across all target-family ops through the builder and translation paths.

---
Reference
- Implementation PR: https://github.com/llvm/llvm-project/pull/173509 (merged)
- Summary: Propagate `device` clause on `omp.target` through MLIR → LLVM IR; pass device id to host offload call; add validation placement; update docs/tests.

---
## Change Log (PR #173509)
- **Docs:** `flang/docs/OpenMPSupport.md` — target construct marked supported (device clause now implemented).
```diff
@@ -38,7 +38,7 @@ Note : No distinction is made between the support in Parser/Semantics, MLIR, Low
| declare simd construct | N | | |
| do simd construct | P | linear clause is not supported | |
| target data construct | P | device clause not supported | |
-| target construct | P | device clause not supported | |
+| target construct | Y | | |
| target update construct | P | device clause not supported | |
| declare target directive | Y | | |
| teams construct | Y | | |
```
- **Flang Lowering:** `flang/lib/Lower/OpenMP/OpenMP.cpp` — include `Device` in clause handling to emit `omp.target device(...)` correctly.
```diff
@@ -4087,7 +4087,8 @@ static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable, |
 !std::holds_alternative<clause::Mergeable>(clause.u) &&
 !std::holds_alternative<clause::Untied>(clause.u) &&
 !std::holds_alternative<clause::TaskReduction>(clause.u) &&
-!std::holds_alternative<clause::Detach>(clause.u)) {
+!std::holds_alternative<clause::Detach>(clause.u) &&
+!std::holds_alternative<clause::Device>(clause.u)) {
```
- **Flang Tests:** `flang/test/Lower/OpenMP/target.f90` — add device clause cases for `i16/i32/i64` variables and constants; FileCheck verifies `omp.target device(%… : i16/i32/i64)`.
```diff
@@ -694,3 +694,44 @@ subroutine target_unstructured |
 !$omp end target
 !CHECK: }
 end subroutine target_unstructured
+  
+ !===============================================================================
+ ! Target `device` clause
+ !===============================================================================
+  
+ !CHECK-LABEL: func.func @_QPomp_target_device() {
+ subroutine omp_target_device
+ integer :: dev32
+ integer(kind=8) :: dev64
+ integer(kind=2) :: dev16
+  
+ dev32 = 1
+ dev64 = 2_8
+ dev16 = 3_2
+  
+ !$omp target device(dev32)
+ !$omp end target
+ ! CHECK: %[[DEV32:.*]] = fir.load %{{.*}} : !fir.ref<i32>
+ ! CHECK: omp.target device(%[[DEV32]] : i32)
+  
+ !$omp target device(dev64)
+ !$omp end target
+ ! CHECK: %[[DEV64:.*]] = fir.load %{{.*}} : !fir.ref<i64>
+ ! CHECK: omp.target device(%[[DEV64]] : i64)
+  
+ !$omp target device(dev16)
+ !$omp end target
+ ! CHECK: %[[DEV16:.*]] = fir.load %{{.*}} : !fir.ref<i16>
+ ! CHECK: omp.target device(%[[DEV16]] : i16)
+  
+ !$omp target device(2)
+ !$omp end target
+ ! CHECK: %[[C2:.*]] = arith.constant 2 : i32
+ ! CHECK: omp.target device(%[[C2]] : i32)
+  
+ !$omp target device(5_8)
+ !$omp end target
+ ! CHECK: %[[C5:.*]] = arith.constant 5 : i64
+ ! CHECK: omp.target device(%[[C5]] : i64)
+  
+ end subroutine omp_target_device
```
- **LLVM API:** `llvm/include/llvm/Frontend/OpenMP/OMPIRBuilder.h` — add `RuntimeAttrs.DeviceID` member for kernel launches.
```diff
@@ -2515,6 +2515,9 @@ class OpenMPIRBuilder {
 /// Total number of iterations of the SPMD or Generic-SPMD kernel or null if
 /// it is a generic kernel.
 Value *LoopTripCount = nullptr;
+  
+ /// Device ID value used in the kernel launch.
+ Value *DeviceID = nullptr;
 };
```
- **LLVM Impl:** `llvm/lib/Frontend/OpenMP/OMPIRBuilder.cpp` — remove placeholder device id; pass `RuntimeAttrs.DeviceID` to `emitTargetTask` and `emitKernelLaunch`.
```diff
@@ -8680,8 +8680,6 @@ static void emitTargetCall( |
 }
  
 unsigned NumTargetItems = Info.NumberOfPtrs;
- // TODO: Use correct device ID
- Value *DeviceID = Builder.getInt64(OMP_DEVICEID_UNDEF);
 uint32_t SrcLocStrSize;
 Constant *SrcLocStr = OMPBuilder.getOrCreateDefaultSrcLocStr(SrcLocStrSize);
@@ -8707,13 +8705,13 @@ static void emitTargetCall( |
 // The presence of certain clauses on the target directive require the
 // explicit generation of the target task.
 if (RequiresOuterTargetTask)
- return OMPBuilder.emitTargetTask(TaskBodyCB, DeviceID, RTLoc, AllocaIP,
- Dependencies, KArgs.RTArgs,
- Info.HasNoWait);
+ return OMPBuilder.emitTargetTask(TaskBodyCB, RuntimeAttrs.DeviceID,
+ RTLoc, AllocaIP, Dependencies,
+ KArgs.RTArgs, Info.HasNoWait);
  
- return OMPBuilder.emitKernelLaunch(Builder, OutlinedFnID,
- EmitTargetCallFallbackCB, KArgs,
- DeviceID, RTLoc, AllocaIP);
+ return OMPBuilder.emitKernelLaunch(
+ Builder, OutlinedFnID, EmitTargetCallFallbackCB, KArgs,
+ RuntimeAttrs.DeviceID, RTLoc, AllocaIP);
 }());
  
 Builder.restoreIP(AfterIP);
```
- **LLVM Tests:** `llvm/unittests/Frontend/OpenMPIRBuilderTest.cpp` — initialize `RuntimeAttrs.DeviceID` to `OMP_DEVICEID_UNDEF` in target region tests.
```diff
@@ -6501,6 +6501,7 @@ TEST_F(OpenMPIRBuilderTest, TargetRegion) {
 RuntimeAttrs.TargetThreadLimit[0] = Builder.getInt32(20);
 RuntimeAttrs.TeamsThreadLimit[0] = Builder.getInt32(30);
 RuntimeAttrs.MaxThreads = Builder.getInt32(40);
+ RuntimeAttrs.DeviceID = Builder.getInt64(llvm::omp::OMP_DEVICEID_UNDEF);
@@ -6834,6 +6835,7 @@ TEST_F(OpenMPIRBuilderTest, TargetRegionSPMD) {
 /*ExecFlags=*/omp::OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_SPMD,
 /*MaxTeams=*/{-1}, /*MinTeams=*/0, /*MaxThreads=*/{0}, /*MinThreads=*/0};
 RuntimeAttrs.LoopTripCount = Builder.getInt64(1000);
+ RuntimeAttrs.DeviceID = Builder.getInt64(llvm::omp::OMP_DEVICEID_UNDEF);
```
- **MLIR→LLVM Translation:** `mlir/lib/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.cpp` — set default device id; when present, lookup and `sextOrTrunc` to `i64`; add device checks for `target_update` and `target_data`.
```diff
@@ -444,12 +444,16 @@ static LogicalResult checkImplementationStatus(Operation &op) {
 .Case([&](omp::SimdOp op) { checkReduction(op, result); })
 .Case<omp::AtomicReadOp, omp::AtomicWriteOp, omp::AtomicUpdateOp,
 omp::AtomicCaptureOp>([&](auto op) { checkHint(op, result); })
- .Case<omp::TargetEnterDataOp, omp::TargetExitDataOp, omp::TargetUpdateOp>(
+ .Case<omp::TargetEnterDataOp, omp::TargetExitDataOp>(
 [&](auto op) { checkDepend(op, result); })
+ .Case<omp::TargetUpdateOp>([&](auto op) {
+ checkDepend(op, result);
+ checkDevice(op, result);
+ })
+ .Case<omp::TargetDataOp>([&](auto op) { checkDevice(op, result); })
 .Case([&](omp::TargetOp op) {
 checkAllocate(op, result);
 checkBare(op, result);
- checkDevice(op, result);
 checkInReduction(op, result);
 })
 .Default([](Operation &) {
@@ -5951,6 +5955,13 @@ initTargetRuntimeAttrs(llvm::IRBuilderBase &builder, |
 {}, /*HasNUW=*/true);
 }
 }
+  
+ attrs.DeviceID = builder.getInt64(llvm::omp::OMP_DEVICEID_UNDEF);
+ if (mlir::Value devId = targetOp.getDevice()) {
+ attrs.DeviceID = moduleTranslation.lookupValue(devId);
+ attrs.DeviceID =
+ builder.CreateSExtOrTrunc(attrs.DeviceID, builder.getInt64Ty());
+ }
 }
  
 static LogicalResult
```
- **MLIR Tests:** `mlir/test/Target/LLVMIR/omptarget-device.mlir` — new tests validate device id threading to `__tgt_target_kernel` (2nd arg as `i64`, constants/vars cast as needed); update `openmp-todo.mlir` to remove now-implemented expected errors.
```diff
@@ -0,0 +1,68 @@
+// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s
+ 
+module attributes {omp.is_target_device = false, omp.target_triples = ["nvptx64-nvidia-cuda"]} {
+  llvm.func @foo(%d16 : i16, %d32 : i32, %d64 : i64) {
+    %x = llvm.mlir.constant(0 : i32) : i32
+ 
+    // Constant i16 -> i64 in the runtime call.
+    %c1_i16 = llvm.mlir.constant(1 : i16) : i16
+    omp.target device(%c1_i16 : i16)
+    host_eval(%x -> %lb, %x -> %ub, %x -> %step : i32, i32, i32) {
+      omp.terminator
+    }
+ 
+    // Constant i32 -> i64 in the runtime call.
+    %c2_i32 = llvm.mlir.constant(2 : i32) : i32
+    omp.target device(%c2_i32 : i32)
+    host_eval(%x -> %lb, %x -> %ub, %x -> %step : i32, i32, i32) {
+      omp.terminator
+    }
+ 
+    // Constant i64 stays i64 in the runtime call.
+    %c3_i64 = llvm.mlir.constant(3 : i64) : i64
+    omp.target device(%c3_i64 : i64)
+    host_eval(%x -> %lb, %x -> %ub, %x -> %step : i32, i32, i32) {
+      omp.terminator
+    }
+ 
+    // Variable i16 -> cast to i64.
+    omp.target device(%d16 : i16)
+    host_eval(%x -> %lb, %x -> %ub, %x -> %step : i32, i32, i32) {
+      omp.terminator
+    }
+ 
+    // Variable i32 -> cast to i64.
+    omp.target device(%d32 : i32)
+    host_eval(%x -> %lb, %x -> %ub, %x -> %step : i32, i32, i32) {
+      omp.terminator
+    }
+ 
+    // Variable i64 stays i64.
+    omp.target device(%d64 : i64)
+    host_eval(%x -> %lb, %x -> %ub, %x -> %step : i32, i32, i32) {
+      omp.terminator
+    }
+ 
+    llvm.return
+  }
+}
+ 
+// CHECK-LABEL: define void @foo(i16 %{{.*}}, i32 %{{.*}}, i64 %{{.*}}) {
+// CHECK: br label %entry
+// CHECK: entry:
+ 
+// ---- Constant cases (device id is 2nd argument) ----
+// CHECK-DAG: call i32 @__tgt_target_kernel(ptr {{.*}}, i64 1, i32 {{.*}}, i32 {{.*}}, ptr {{.*}}, ptr {{.*}})
+// CHECK-DAG: call i32 @__tgt_target_kernel(ptr {{.*}}, i64 2, i32 {{.*}}, i32 {{.*}}, ptr {{.*}}, ptr {{.*}})
+// CHECK-DAG: call i32 @__tgt_target_kernel(ptr {{.*}}, i64 3, i32 {{.*}}, i32 {{.*}}, ptr {{.*}}, ptr {{.*}})
+ 
+// Variable i16 -> i64
+// CHECK: %[[D16_I64:.*]] = sext i16 %{{.*}} to i64
+// CHECK: call i32 @__tgt_target_kernel(ptr {{.*}}, i64 %[[D16_I64]], i32 {{.*}}, i32 {{.*}}, ptr {{.*}}, ptr {{.*}})
+ 
+// Variable i32 -> i64
+// CHECK: %[[D32_I64:.*]] = sext i32 %{{.*}} to i64
+// CHECK: call i32 @__tgt_target_kernel(ptr {{.*}}, i64 %[[D32_I64]], i32 {{.*}}, i32 {{.*}}, ptr {{.*}}, ptr {{.*}})
+ 
+// Variable i64
+// CHECK: call i32 @__tgt_target_kernel(ptr {{.*}}, i64 %{{.*}}, i32 {{.*}}, i32 {{.*}}, ptr {{.*}}, ptr {{.*}})
```
```diff
@@ -174,8 +174,6 @@ llvm.func @target_allocate(%x : !llvm.ptr) {
 // -----
  
 llvm.func @target_device(%x : i32) {
- // expected-error@below {{not yet implemented: Unhandled clause device in omp.target operation}}
- // expected-error@below {{LLVM Translation failed for operation: omp.target}}
 omp.target device(%x : i32) {
 omp.terminator
 }
```

**Runtime Semantics**
- **Device ID propagation:** The `omp.target` device clause lowers to an `i64` device id passed as the 2nd argument to `__tgt_target_kernel`. Narrower types (`i16/i32`) are sign-extended; `i64` passes through.
- **Default behavior:** If no device clause is present, `DeviceID` defaults to `OMP_DEVICEID_UNDEF` via `OpenMPIRBuilder` runtime attributes.

Pitfalls & Reviewer Notes
- Ensure device id is sign-extended or truncated to `i64` before calling the offload runtime; mismatched width can break targets expecting `i64`.
- Do not reinterpret `-1`; let the runtime handle “default device”. Keep logic consistent with `OMP_DEVICEID_UNDEF`.
- Clause validation: current device checks are added for `target_update` and `target_data`; keep `target` checks aligned with existing validation helpers to avoid duplicate diagnostics.
- ABI ordering: The device id is the second argument to `__tgt_target_kernel`; changing ordering breaks compatibility.
- Tests should cover both SSA values and immediates (multiple widths) to guard against regressions in casts.

---

## [flang][mlir][OpenMP] Add support for uniform clause in declare simd

Overview
- Purpose: Add support for the `uniform(...)` clause on `declare simd`, enabling SIMD variants where specified arguments are uniform (identical across lanes).
- Scope: Declarative directive `declare simd` in Flang (semantics + lowering) and MLIR OpenMP dialect representation.

Spec Reference
- OpenMP 5.2 — Declare SIMD directive and `uniform` clause semantics.

Semantics
- Parameters listed in `uniform(...)` are treated as uniform across SIMD lanes for the SIMD variant of the routine.
- Enforce that `uniform` parameters are dummy arguments of the associated procedure.

Reference
- Implementation PR: https://github.com/llvm/llvm-project/pull/176046 (merged)
- Summary: Define `uniform` clause in MLIR OpenMP and emit from Flang; add semantic checks and tests.

## Change Log (PR #176046)
- **Flang Lowering (impl):** `flang/lib/Lower/OpenMP/ClauseProcessor.cpp` — add processor for `uniform` clause.
```diff
@@ -1716,6 +1716,16 @@ bool ClauseProcessor::processUseDevicePtr(
	return clauseFound;
 }
 
+bool ClauseProcessor::processUniform(
+    mlir::omp::UniformClauseOps &result) const {
+  return findRepeatableClause<omp::clause::Uniform>(
+      [&](const omp::clause::Uniform &clause, const parser::CharBlock &) {
+        const auto &objects = clause.v;
+        if (!objects.empty())
+          genObjectList(objects, converter, result.uniformVars);
+      });
+}
```

- **Flang Lowering (decl):** `flang/lib/Lower/OpenMP/ClauseProcessor.h` — declare `processUniform`.
```diff
@@ -168,6 +168,7 @@ class ClauseProcessor {
	lower::StatementContext &stmtCtx,
	mlir::omp::UseDevicePtrClauseOps &result,
	llvm::SmallVectorImpl<const semantics::Symbol *> &useDeviceSyms) const;
+  bool processUniform(mlir::omp::UniformClauseOps &result) const;
 
	// Call this method for these clauses that should be supported but are not
	// implemented yet. It triggers a compilation error if any of the given
```

- **Flang Lowering:** `flang/lib/Lower/OpenMP/OpenMP.cpp` — wire up uniform processing for `declare simd`.
```diff
@@ -3848,7 +3848,7 @@ genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
	cp.processAligned(clauseOps);
	cp.processLinear(clauseOps);
	cp.processSimdlen(clauseOps);
-  cp.processTODO<clause::Uniform>(loc, llvm::omp::Directive::OMPD_declare_simd);
+  cp.processUniform(clauseOps);
 
	mlir::omp::DeclareSimdOp::create(converter.getFirOpBuilder(), loc, clauseOps);
 }
```

- **Flang Semantics:** `flang/lib/Semantics/check-omp-structure.cpp` — validate `uniform` names are dummy arguments of the enclosing procedure.
```diff
@@ -1446,6 +1446,28 @@ void OmpStructureChecker::Enter(const parser::OpenMPDeclareSimdConstruct &x) {
	const parser::OmpDirectiveName &dirName{x.v.DirName()};
	PushContextAndClauseSets(dirName.source, dirName.v);
 
+  const Scope &containingScope = context_.FindScope(dirName.source);
+  const Scope &progUnitScope = GetProgramUnitContaining(containingScope);
+
+  for (const parser::OmpClause &clause : x.v.Clauses().v) {
+    const auto *u = std::get_if<parser::OmpClause::Uniform>(&clause.u);
+    if (!u)
+      continue;
+    assert(clause.Id() == llvm::omp::Clause::OMPC_uniform);
+
+    for (const parser::Name &name : u->v) {
+      const Symbol *sym{name.symbol};
+      if (!sym || !IsDummy(*sym) ||
+          &GetProgramUnitContaining(sym->owner()) != &progUnitScope) {
+        context_.Say(name.source,
+            "Variable '%s' in UNIFORM clause must be a dummy argument of the "
+            "enclosing procedure"_err_en_US,
+            name.ToString());
+      }
+    }
+  }
 
	const parser::OmpArgumentList &args{x.v.Arguments()};
	if (args.v.empty()) {
```

- **Flang Tests (Lowering):** `flang/test/Lower/OpenMP/declare-simd.f90` — add uniform clause cases and combined clause coverage.
```diff
@@ -73,11 +73,40 @@ end subroutine declare_simd_simdlen
 
+subroutine declare_simd_uniform(x, y, n, i)
+#ifdef OMP_60
+!$omp declare_simd uniform(x, y)
+#else
+!$omp declare simd uniform(x, y)
+#endif
+
+  real(8), pointer, intent(inout) :: x(:)
+  real(8), pointer, intent(in) :: y(:)
+  integer, intent(in) :: n, i
+
+  if (i <= n) then
+    x(i) = x(i) + y(i)
+  end if
+end subroutine declare_simd_uniform
+
@@ -105,4 +134,7 @@ end subroutine declare_simd_combined
-!$omp declare simd aligned(x, y : 64) linear(i) simdlen(8)
+!$omp declare simd aligned(x, y : 64) linear(i) simdlen(8) uniform(x, y)
 ! CHECK-SAME: simdlen(8)
 ! CHECK-SAME: {linear_var_types = [i32]}
+! CHECK-SAME: uniform(%[[X_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>,
+! CHECK-SAME: %[[Y_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>)
```

- **Flang Tests (Semantics):** `flang/test/Semantics/OpenMP/declare-simd-uniform.f90` — add negative test and update RUN line.
```diff
@@ -1,4 +1,4 @@
-! RUN: %python %S/../test_errors.py %s %flang -fopenmp
+! RUN: %python %S/../test_errors.py %s %flang -fopenmp -cpp -DNEGATIVE
@@
+#ifdef NEGATIVE
+function bad_uniform(a,b,i) result(c)
+  double precision :: a(*), b(*), c
+  integer :: i
+  double precision :: local
+  !ERROR: Variable 'local' in UNIFORM clause must be a dummy argument of the enclosing procedure
+  !$omp declare simd(bad_uniform) uniform(a,local)
+  c = a(i) + b(i) + local
+end function
+#endif
```

- **MLIR Dialect (clauses):** `mlir/include/mlir/Dialect/OpenMP/OpenMPClauses.td` — define `OpenMP_UniformClause` with assembly format.
```diff
@@ -1532,4 +1532,28 @@
+//===----------------------------------------------------------------------===//
+// V5.2: [5.10] `uniform` clause
+//===----------------------------------------------------------------------===//
+class OpenMP_UniformClauseSkip<
+  bit traits = false, bit arguments = false, bit assemblyFormat = false,
+  bit description = false, bit extraClassDeclaration = false>
+  : OpenMP_Clause<traits, arguments, assemblyFormat, description,
+                  extraClassDeclaration> {
+  let arguments = (ins Variadic<OpenMP_PointerLikeType>:$uniform_vars);
+  let optAssemblyFormat = [{
+    `uniform` `(` custom<UniformClause>($uniform_vars, type($uniform_vars)) `)`
+  }];
+  let description = [{
+    The `uniform` clause declares one or more arguments to have an invariant
+    value for all concurrent invocations of the function in the execution of
+    a single SIMD loop.
+  }];
+}
+def OpenMP_UniformClause : OpenMP_UniformClauseSkip<>;
```

- **MLIR Dialect (ops):** `mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td` — add `OpenMP_UniformClause` to `omp.declare_simd`.
```diff
@@ -2248,7 +2248,7 @@ def DeclareSimdOp
-    OpenMP_SimdlenClause]>
+    OpenMP_SimdlenClause, OpenMP_UniformClause]>
```

- **MLIR Impl:** `mlir/lib/Dialect/OpenMP/IR/OpenMPDialect.cpp` — parse/print uniform clause and plumb through builder.
```diff
@@ -4485,7 +4485,37 @@ void DeclareSimdOp::build(OpBuilder &odsBuilder, OperationState &odsState,
-  clauses.linearVarTypes, clauses.simdlen);
+  clauses.linearVarTypes, clauses.simdlen,
+  clauses.uniformVars);
@@
+// Parser and printer for Uniform Clause
+static ParseResult parseUniformClause(OpAsmParser &parser,
+    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &uniformVars,
+    SmallVectorImpl<Type> &uniformTypes) {
+  return parser.parseCommaSeparatedList([&]() -> mlir::ParseResult {
+    if (parser.parseOperand(uniformVars.emplace_back()) ||
+        parser.parseColonType(uniformTypes.emplace_back()))
+      return mlir::failure();
+    return mlir::success();
+  });
+}
+static void printUniformClause(OpAsmPrinter &p, Operation *op,
+    ValueRange uniformVars, TypeRange uniformTypes) {
+  for (unsigned i = 0; i < uniformVars.size(); ++i) {
+    if (i != 0)
+      p << ", ";
+    p << uniformVars[i] << " : " << uniformTypes[i];
+  }
+}
```

- **MLIR Tests:** `mlir/test/Dialect/OpenMP/ops.mlir` — add dedicated uniform test and all-clauses coverage.
```diff
@@ -3414,6 +3414,17 @@ func.func @omp_declare_simd_linear(%a: f64, %b: f64, %iv: i32, %step: i32) -> ()
+// CHECK-LABEL: func.func @omp_declare_simd_uniform
+func.func @omp_declare_simd_uniform(%a: f64, %b: f64,
+  %p0: memref<i32>, %p1: memref<i32>) -> () {
+  // CHECK: omp.declare_simd
+  // CHECK-SAME: uniform(
+  // CHECK-SAME: %{{.*}} : memref<i32>,
+  // CHECK-SAME: %{{.*}} : memref<i32>)
+  omp.declare_simd uniform(%p0 : memref<i32>, %p1 : memref<i32>)
+  return
+}
@@ -3424,9 +3435,13 @@ func.func @omp_declare_simd_all_clauses(%a: f64, %b: f64,
 // CHECK-SAME: simdlen(8)
+// CHECK-SAME: uniform(
+// CHECK-SAME: %{{.*}} : memref<i32>,
+// CHECK-SAME: %{{.*}} : memref<i32>)
	omp.declare_simd simdlen(8)
	aligned(%p0 : memref<i32> -> 32 : i64,
			  %p1 : memref<i32> -> 128 : i64)
	linear(%iv = %step : i32)
+  uniform(%p0 : memref<i32>, %p1 : memref<i32>)
	return
 }
```

Pitfalls & Reviewer Notes
- Uniform variables must be dummy arguments of the enclosing procedure (reviewer request); emit clear diagnostics for locals or host-associated symbols.
- MLIR assembly: `uniform(%arg : type, …)` printing/parsing must preserve both SSA ids and types; keep tests resilient to spacing/ordering.
- Clause interactions: Combining `uniform` with `aligned`, `linear`, and `simdlen` changes operand segment sizes; ensure builder and traits stay consistent.
- Frontend differences: For OMP 5.0 vs 5.2 spelling (`declare simd` vs `declare_simd`), guard tests with `-cpp` defines as needed.
- Negative testing: Enable `-cpp -DNEGATIVE` path in semantics tests to exercise error cases reliably in CI.

---
