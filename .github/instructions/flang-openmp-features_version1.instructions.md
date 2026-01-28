---
applyTo: flang/lib/Parser/openmp*,flang/lib/Semantics/*omp*,flang/lib/Lower/OpenMP/**/*,flang/lib/Optimizer/OpenMP/**/*,flang/include/flang/Parser/parse-tree.h,flang/include/flang/Lower/OpenMP*,flang/include/flang/Support/OpenMP*,flang/include/flang/Optimizer/OpenMP/**/*,mlir/include/mlir/Dialect/OpenMP/**/*,mlir/lib/Dialect/OpenMP/**/*,mlir/lib/Target/LLVMIR/Dialect/OpenMP/**/*,llvm/include/llvm/Frontend/OpenMP/**/*,llvm/lib/Frontend/OpenMP/**/*
---

Status
- PRs included: 10
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

## [llvm][mlir][OpenMP] Support translation for linear clause in omp.wsloop and omp.simd

Overview
- Purpose: Add end-to-end support to lower/translate the `linear` clause for `omp.wsloop` (Fortran `!$omp do`) and `omp.simd` through Flang → MLIR OpenMP dialect → LLVM IR.
- Scope: Flang clause processing for `linear`, MLIR OpenMP dialect storage of linear variable types, and LLVM IR translation to implement linear semantics. Adds targeted tests.

Spec Reference
- OpenMP 5.2 — Linear clause on loop and SIMD constructs. See linear(var[:step]) semantics and permitted modifiers.

Semantics
- Each variable in `linear(...)` evolves linearly with the logical iteration: value at iteration k is base + k × step (default step = 1). Implementation stores linear variable types on the ops to support correct LLVM lowering without assuming alloca provenance.

Reference
- Implementation PR: https://github.com/llvm/llvm-project/pull/139386 (merged)
- Summary: Wire `linear` through Flang clause handling for wsloop/simd, model `linear_var_types` on MLIR ops, and implement LLVM translation, with tests covering base and step forms.

## Change Log (PR #139386)
- **Flang Lowering:** `flang/lib/Lower/OpenMP/ClauseProcessor.cpp` — capture linear var types into an array attribute on the op.
```diff
@@ -1241,11 +1241,20 @@ bool ClauseProcessor::processLinear(mlir::omp::LinearClauseOps &result) const {
	auto &objects = std::get<omp::ObjectList>(clause.t);
+  static std::vector<mlir::Attribute> typeAttrs;
+  if (!result.linearVars.size())
+    typeAttrs.clear();
	for (const omp::Object &object : objects) {
	  semantics::Symbol *sym = object.sym();
	  const mlir::Value variable = converter.getSymbolAddress(*sym);
	  result.linearVars.push_back(variable);
+    mlir::Type ty = converter.genType(*sym);
+    typeAttrs.push_back(mlir::TypeAttr::get(ty));
	}
+  result.linearVarTypes =
+      mlir::ArrayAttr::get(&converter.getMLIRContext(), typeAttrs);
	if (objects.size()) {
	  if (auto &mod =
			std::get<std::optional<omp::clause::Linear::StepComplexModifier>>(
```
- **Flang Lowering:** `flang/lib/Lower/OpenMP/OpenMP.cpp` — enable `linear` processing on `simd` and `wsloop`.
```diff
@@ -1636,8 +1636,7 @@ static void genSimdClauses(
	cp.processReduction(loc, clauseOps, reductionSyms);
	cp.processSafelen(clauseOps);
	cp.processSimdlen(clauseOps);
-  
-  cp.processTODO<clause::Linear>(loc, llvm::omp::Directive::OMPD_simd);
+  cp.processLinear(clauseOps);
@@ -1831,9 +1830,9 @@ static void genWsloopClauses(
	cp.processOrdered(clauseOps);
	cp.processReduction(loc, clauseOps, reductionSyms);
	cp.processSchedule(stmtCtx, clauseOps);
+  cp.processLinear(clauseOps);
 
-  cp.processTODO<clause::Allocate, clause::Linear>(
-      loc, llvm::omp::Directive::OMPD_do);
+  cp.processTODO<clause::Allocate>(loc, llvm::omp::Directive::OMPD_do);
 }
```
- **Flang Tests:** add `simd-linear.f90` and `wsloop-linear.f90`; remove obsolete todo test.
```diff
--- a/flang/test/Lower/OpenMP/Todo/omp-do-simd-linear.f90
+++ /dev/null
@@
- (file removed)
```
```diff
@@ -0,0 +1,57 @@
! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - 2>&1 | FileCheck %s
subroutine simple_linear
  implicit none
  integer :: x, y, i
  !$omp simd linear(x)
  do i = 1, 10
	 y = x + 2
  end do
  ! CHECK: } {linear_var_types = [i32]}
end subroutine
```
```diff
@@ -0,0 +1,60 @@
! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - 2>&1 | FileCheck %s
subroutine simple_linear
  implicit none
  integer :: x, y, i
  !$omp do linear(x)
  do i = 1, 10
	 y = x + 2
  end do
  !$omp end do
  ! CHECK: } {linear_var_types = [i32]}
end subroutine
```
- **MLIR Clauses:** `mlir/include/mlir/Dialect/OpenMP/OpenMPClauses.td` — add `linear_var_types` attribute and include BuiltinAttributes.
```diff
@@ -21,6 +24,7 @@
 include "mlir/Dialect/OpenMP/OpenMPOpBase.td"
 include "mlir/IR/SymbolInterfaces.td"
+include "mlir/IR/BuiltinAttributes.td"
@@ -723,10 +729,9 @@ class OpenMP_LinearClauseSkip<
-  let arguments = (ins
-      Variadic<AnyType>:$linear_vars,
-      Variadic<I32>:$linear_step_vars);
+  let arguments = (ins Variadic<AnyType>:$linear_vars,
+      Variadic<I32>:$linear_step_vars,
+      OptionalAttr<ArrayAttr>:$linear_var_types);
```
- **MLIR Ops Impl:** `mlir/lib/Dialect/OpenMP/IR/OpenMPDialect.cpp` — plumb `linearVarTypes` for `wsloop` and `simd` builders; update verifier accordingly.
```diff
@@ -2825,6 +2828,7 @@ void WsloopOp::build(OpBuilder &builder, OperationState &state,
	build(builder, state, /*allocate_vars=*/{}, /*allocator_vars=*/{},
			/*linear_vars=*/ValueRange(), /*linear_step_vars=*/ValueRange(),
+        /*linear_var_types*/ nullptr,
			/*nowait=*/false, /*order=*/nullptr, /*order_mod=*/nullptr,
			/*ordered=*/nullptr, /*private_vars=*/{}, /*private_syms=*/nullptr,
@@ -2843,8 +2847,8 @@ void WsloopOp::build(
	WsloopOp::build(
		 builder, state,
		 /*allocate_vars=*/{}, /*allocator_vars=*/{}, clauses.linearVars,
-      clauses.linearStepVars, clauses.nowait, clauses.order, clauses.orderMod,
+      clauses.linearStepVars, clauses.linearVarTypes, clauses.nowait,
+      clauses.order, clauses.orderMod,
		 clauses.ordered, clauses.privateVars,
@@ -2889,17 +2890,16 @@ LogicalResult WsloopOp::verifyRegions() {
-  // TODO Store clauses in op: linearVars, linearStepVars
-  SimdOp::build(builder, state, clauses.alignedVars,
-                makeArrayAttr(ctx, clauses.alignments), clauses.ifExpr,
-                /*linear_vars=*/{}, /*linear_step_vars=*/{},
-                clauses.nontemporalVars, clauses.order, clauses.orderMod,
-                clauses.privateVars, makeArrayAttr(ctx, clauses.privateSyms),
-                clauses.privateNeedsBarrier, clauses.reductionMod,
-                clauses.reductionVars,
-                makeDenseBoolArrayAttr(ctx, clauses.reductionByref),
-                makeArrayAttr(ctx, clauses.reductionSyms), clauses.safelen,
-                clauses.simdlen);
+  SimdOp::build(
+      builder, state, clauses.alignedVars,
+      makeArrayAttr(ctx, clauses.alignments), clauses.ifExpr,
+      clauses.linearVars, clauses.linearStepVars, clauses.linearVarTypes,
+      clauses.nontemporalVars, clauses.order, clauses.orderMod,
+      clauses.privateVars, makeArrayAttr(ctx, clauses.privateSyms),
+      clauses.privateNeedsBarrier, clauses.reductionMod, clauses.reductionVars,
+      makeDenseBoolArrayAttr(ctx, clauses.reductionByref),
+      makeArrayAttr(ctx, clauses.reductionSyms), clauses.safelen,
+      clauses.simdlen);
```

Pitfalls & Reviewer Notes
- Do not assume linear vars originate from allocas; types must be conveyed explicitly (attribute) for correct LLVM lowering.
- Avoid redundant barriers around linear variable processing in LLVM IR emission.
- Keep MLIR tests canonical (e.g., `mlir-opt --canonicalize`) to reduce noise; ensure access group metadata propagates to loads/stores as required.
- Versioning pitfalls: Some tests (e.g., with step-complex-modifier) rely on OpenMP 5.2; default toolchains using older OpenMP versions may warn where an error is expected.
- Naming clarity: Prefer `split*` terminology over “outline” for basic block splitting in linear finalization (review feedback).

Related Issues/PRs
- Stacked before: https://github.com/llvm/llvm-project/pull/139385
- Follow-up (implicit linearization): https://github.com/llvm/llvm-project/pull/150386 (merged)
- Additional checks/tests: https://github.com/llvm/llvm-project/pull/174916 (merged)
- Testsuite adjustment: https://github.com/llvm/llvm-test-suite/pull/308 (merged)

Source Code Links
- Flang: `ClauseProcessor.cpp` https://github.com/llvm/llvm-project/pull/139386/files#diff-55798c5090a8f8499f773b3fb46fb98a7aabe0f58ec60ff295e107acb36c7707
- Flang: `OpenMP.cpp` https://github.com/llvm/llvm-project/pull/139386/files#diff-496a295679ae3c43f8651c944a1bd9dca177ad2b5e4d7121f96938024e292bc1
- MLIR: `OpenMPClauses.td` https://github.com/llvm/llvm-project/pull/139386/files#diff-0a931c4acc64c5d1088e87fc4688545b139ac506da7b354263b1370304fd6ae5
- MLIR: `OpenMPDialect.cpp` https://github.com/llvm/llvm-project/pull/139386/files#diff-a897370ad8f5ad37e8c1adb3c145c2304aaa38da3227bc1d02ac701ee8dc0754
- MLIR: `OpenMPToLLVMIRTranslation.cpp` https://github.com/llvm/llvm-project/pull/139386/files#diff-2cbb5651f4570d81d55ac4198deda0f6f7341b2503479752ef2295da3774c586
- Tests: `simd-linear.f90`, `wsloop-linear.f90`, `openmp-llvm.mlir` (files tab in PR)

How To Avoid Pitfalls
- Provide `linear_var_types` on MLIR ops; never rely on inferring element types from pointer operands during LLVM translation.
- Ensure Flang semantics guard invalid linear vars (type/REF requirements) and that lowering/translation assumes validated inputs.
- Keep IR emission minimal: remove unnecessary barriers/conditionals; rely on canonical optimizer passes for cleanup.
- When writing tests that depend on OpenMP 5.2 features, pass appropriate `-fopenmp-version=52` (or gate with `-cpp` defines) to avoid misleading warnings.

---

## [flang][OpenMP] Parse OpenMP 6.0 syntax of INIT clause

Overview
- Purpose: Update Flang to parse and unparse the OpenMP 6.0 `INIT` clause syntax, introducing preference syntax (`PREFER_TYPE(...)`) with structured preference selectors and adjusting interop handling.
- Scope: Parser AST (`parse-tree.h`), parser combinators (`openmp-parsers.cpp`), unparsing (`unparse.cpp`), feature dumping (`FeatureList.cpp`, `dump-parse-tree.h`), and semantics descriptors (`openmp-modifiers.{h,cpp}`). Also updates an interop test.

Spec Reference
- OpenMP 6.0 — Interoperability and `INIT` clause syntax updates: preference selector (`FR(...)` and `ATTR(...)` forms), preference specification, and `PREFER_TYPE(...)` grouping.

Semantics
- Replace deprecated `OmpInteropPreference` with `OmpPreferType` and add `OmpPreferenceSelector`/`OmpPreferenceSpecification` supporting `FR(expr)` and `ATTR(expr, ...)` forms and grouped `{ ... }` lists.
- In `OmpInitClause`, modifiers now accept `OmpInteropType` and `OmpPreferType`; parser prioritizes `interop-type` before `prefer-type` to avoid ambiguity since `prefer-type` can carry arbitrary expressions.
- Unparsing emits `PREFER_TYPE(...)` with either grouped selectors or foreign runtime identifier, and simplifies `INTEROP` directive unparsing to a generic `!$OMP ` followed by the specification.

Reference
- Implementation PR: https://github.com/llvm/llvm-project/pull/171702 (merged)
- Summary: Introduce preference selector/types and wire them into `INIT` clause parsing/unparsing; update feature readers and semantics descriptors; remove deprecated interop preference nodes.

## Change Log (PR #171702)
- Feature Dumper: `flang/examples/FeatureList/FeatureList.cpp` — add readers for new preference nodes; remove deprecated interop preference entries.
```diff
@@ -495,6 +495,9 @@ struct NodeVisitor {
 READ_FEATURE(OmpOrderClause::Ordering)
 READ_FEATURE(OmpOrderModifier)
 READ_FEATURE(OmpOrderModifier::Value)
 + READ_FEATURE(OmpPreferenceSelector)
 + READ_FEATURE(OmpPreferenceSpecification)
 + READ_FEATURE(OmpPreferType)
@@ -509,8 +512,6 @@
 - READ_FEATURE(OmpInteropRuntimeIdentifier)
 - READ_FEATURE(OmpInteropPreference)
 READ_FEATURE(OmpInteropType)
```
- Parser Dump: `flang/include/flang/Parser/dump-parse-tree.h` — add new nodes, drop deprecated ones.
```diff
@@ -627,8 +627,6 @@ class ParseTreeDumper {
 - NODE(parser, OmpInteropPreference)
 - NODE(parser, OmpInteropRuntimeIdentifier)
@@ -681,6 +679,9 @@
 + NODE(parser, OmpPreferenceSelector)
 + NODE(parser, OmpPreferenceSpecification)
 + NODE(parser, OmpPreferType)
```
- Parser AST: `flang/include/flang/Parser/parse-tree.h` — add `OmpPreferenceSelector`, `OmpPreferenceSpecification`, and `OmpPreferType`; adjust `OmpInteropType` docs; update `OmpInitClause` to use `OmpPreferType`.
```diff
@@ -4034, + // Ref: [6.0:470-471]
 + struct OmpPreferenceSelector { /* FR(expr) | ATTR(expr, ...) */ };
 + struct OmpPreferenceSpecification { /* {selector...} | FR(expr) */ };
 + struct OmpPreferType { WRAPPER_CLASS_BOILERPLATE(OmpPreferType, std::list<OmpPreferenceSpecification>); };
@@ -4997, + struct OmpInitClause {
 -   MODIFIER_BOILERPLATE(OmpInteropPreference, OmpInteropType);
 +   MODIFIER_BOILERPLATE(OmpPreferType, OmpInteropType);
 }
```
- Semantics Descriptors: `flang/include/flang/Semantics/openmp-modifiers.h` / `flang/lib/Semantics/openmp-modifiers.cpp` — remove `OmpInteropPreference` descriptor and add `OmpPreferType`.
```diff
@@ -85,7 +85,6 @@
 - DECLARE_DESCRIPTOR(parser::OmpInteropPreference);
@@ -96,6 +95,7 @@
 + DECLARE_DESCRIPTOR(parser::OmpPreferType);
```
```diff
@@ template<>
 - const OmpModifierDescriptor &OmpGetDescriptor<parser::OmpInteropPreference>() { ... }
@@ template<>
 + const OmpModifierDescriptor &OmpGetDescriptor<parser::OmpPreferType>() { /* name="prefer-type", unique, applies to INIT */ }
```
- Parsers: `flang/lib/Parser/openmp-parsers.cpp` — add parsers for preference selector/specification/prefer-type; prioritize `interop-type` then `prefer-type` in `INIT` modifiers; accept `INIT` with updated `_id` literal.
```diff
@@ -898,6 +891,20 @@
 + TYPE_PARSER(construct<OmpPreferenceSelector>("FR" >> parenthesized(indirect(expr))
 +   || "ATTR" >> parenthesized(nonemptyList(indirect(expr)))))
 + TYPE_PARSER(construct<OmpPreferenceSpecification>(braced(nonemptyList(Parser<OmpPreferenceSelector>()))
 +   || construct<OmpPreferenceSpecification>(indirect(expr)))
 + TYPE_PARSER(construct<OmpPreferType>("PREFER_TYPE" >> parenthesized(nonemptyList(Parser<OmpPreferenceSpecification>{}))))
@@ -986,9 +993,9 @@
 - construct<OmpInitClause::Modifier>(Parser<OmpInteropPreference>{})) ||
 - construct<OmpInitClause::Modifier>(Parser<OmpInteropType>{})))
 + construct<OmpInitClause::Modifier>(Parser<OmpInteropType>{}) ||
 + construct<OmpInitClause::Modifier>(Parser<OmpPreferType>{})))
@@ -1490, - "INIT" >> ...
 + "INIT"_id >> ...
```
- Unparser: `flang/lib/Parser/unparse.cpp` — add printers for new preference nodes; simplify `OpenMPInteropConstruct` unparsing.
```diff
@@ void Unparse(const OmpPreferenceSelector &x) { FR(...) | ATTR(...) }
@@ void Unparse(const OmpPreferenceSpecification &x) { { ... } | FR(...) }
@@ void Unparse(const OmpPreferType &x) { Word("PREFER_TYPE"); ... }
@@ void Unparse(const OmpInitClause &x) { Walk(modifiers, ": "); Walk(object); }
@@ void Unparse(const OpenMPInteropConstruct &x) { Word("!$OMP "); Walk(x.v); }
```
- Tests: `flang/test/Semantics/omp/interop-construct.f90` — updated/added to exercise new INIT syntax (files tab reference).

Pitfalls & Reviewer Notes
- Versioning: These constructs are OpenMP 6.0; ensure version gating for diagnostics/pretty-printing when compiling with older OpenMP versions.
- Ambiguity: Since `prefer-type` accepts general expressions, prioritize parsing `interop-type` first; mirror this ordering if extending grammar.
- AST Consumers: Code consuming `OmpInteropPreference` must be migrated to `OmpPreferType`; verify no stale references remain.
- Unparse Fidelity: Grouped selectors (`{...}`) vs direct FR(...) should round-trip correctly, including ATTR lists.

Related Issues/PRs
- Tracks OpenMP 6.0 interop updates and aligns INIT clause parsing.

Source Code Links
- FeatureList.cpp https://github.com/llvm/llvm-project/pull/171702/files#diff-203f376c6d62089848e43e09c7855e8106ab0cb8b01d16fa0cee14177d589b24
- parse-tree.h https://github.com/llvm/llvm-project/pull/171702/files#diff-61d1be39226db8b7f54d3241269d9fabff009139551605546e33f6463bd4087a
- openmp-parsers.cpp https://github.com/llvm/llvm-project/pull/171702/files#diff-fd1ab2b6d2b237c8751b438861ea7c3b2fa89ddbb703caa0d33da3a816cd124a
- unparse.cpp https://github.com/llvm/llvm-project/pull/171702/files#diff-63a2cc1100cf54a50f48c1b5fe6d6c2ffe1cbc46e242c6ae6723f82fcd836dc1
- dump-parse-tree.h https://github.com/llvm/llvm-project/pull/171702/files#diff-062c8b30313609d423f975d2af9ec411e4b9fffbd782216ef55155564fd7649d
- openmp-modifiers.h https://github.com/llvm/llvm-project/pull/171702/files#diff-08b7e57d520f56ecb8269c02a02ab637524e3c5e03ac48dcdc594d0842232dbb
- openmp-modifiers.cpp https://github.com/llvm/llvm-project/pull/171702/files#diff-08ba07f0245fe04f73ed81e9f04543b3786b276e788884bf0c82188d26b7dc5d

How To Avoid Pitfalls
- Maintain parse precedence: `interop-type` before `prefer-type`.
- Gate unparsing/diagnostics by OpenMP version to prevent 6.0-only syntax leaks in older modes.
- Update all semantic descriptors and feature readers to the new node names to avoid mismatches.

---

## Reland "[Flang][OpenMP] Add lowering support for is_device_ptr clause (#169331)"

Overview
- Purpose: Re-land Flang+MLIR support to lower `is_device_ptr` on `omp.target`, ensuring correct device-pointer semantics and runtime map flags.
- Scope: Flang ClauseProcessor API and target lowering, MLIR OpenMP clause map flags, MLIR parser/printer, LLVM translation, and tests across Flang integration, Flang lowering, and MLIR-to-LLVM.

Spec Reference
- OpenMP target mapping semantics for `is_device_ptr`: pointer argument is treated as a device pointer; mapping flags include `TARGET_PARAM` and potentially `LITERAL` when no explicit map is present.

Semantics
- Flang: When encountering `is_device_ptr(list)`, force a map entry with flags `is_device_ptr|to` so the device-side descriptor contains the device address. Track symbols in `isDevicePtrSyms` and synthesize/augment `has_device_addr` entries as needed.
- Flang: Duplicate the underlying `omp.map.info` so both the `is_device_ptr` clause and any synthesized `has_device_addr` user have distinct `MapInfoOp` users, preserving finalization invariants.
- MLIR: Add `is_device_ptr` to `ClauseMapFlags`; parser/print supports `map_clauses(is_device_ptr)`.
- LLVM Translation: Map flag conversion recognizes `is_device_ptr` and sets `OMP_MAP_TARGET_PARAM`, and when no explicit map present also sets `OMP_MAP_LITERAL`. Mark device pointer kind appropriately in collected map data.

Reference
- Implementation PR: https://github.com/llvm/llvm-project/pull/170851 (merged)
- Summary: Enable end-to-end lowering and offload runtime mapping for `is_device_ptr`, remove previous TODO checks, and add tests.

## Change Log (PR #170851)
- Flang ClauseProcessor: `flang/lib/Lower/OpenMP/ClauseProcessor.cpp`/`.h` — extend `processIsDevicePtr` to use `StatementContext`, call `processMapObjects` with `is_device_ptr|to`, collect parent/member info, and insert synthesized child/parent map info; change call site signature.
```diff
@@ bool ClauseProcessor::processIsDevicePtr(
- mlir::omp::IsDevicePtrClauseOps &result,
+ lower::StatementContext &stmtCtx, mlir::omp::IsDevicePtrClauseOps &result,
@@
+ mlir::omp::ClauseMapFlags mapTypeBits = mlir::omp::ClauseMapFlags::is_device_ptr | mlir::omp::ClauseMapFlags::to;
+ processMapObjects(stmtCtx, location, clause.v, mapTypeBits, parentMemberIndices, result.isDevicePtrVars, isDeviceSyms);
```
- Flang OpenMP lowering: `flang/lib/Lower/OpenMP/OpenMP.cpp` — update clause processing order and target op generation to handle `is_device_ptr` vars, duplicate map infos, and avoid duplicate mappings.
```diff
@@ static void genTargetClauses(...)
- cp.processIsDevicePtr(clauseOps, isDevicePtrSyms);
+ cp.processIsDevicePtr(stmtCtx, clauseOps, isDevicePtrSyms);
@@ static bool isDuplicateMappedSymbol(...)
+ add isDevicePtrSyms to duplicate check list
@@ genTargetOp(...)
+ if (!isDevicePtrSyms.empty()) { clone map info and wire synthesized has_device_addr }
```
- Flang Tests:
	- `flang/test/Integration/OpenMP/map-types-and-sizes.f90` — new `mapType_is_device_ptr` subroutine and checks for sizes/maptypes arrays.
	- `flang/test/Lower/OpenMP/target.f90` — new `omp_target_is_device_ptr` test; checks duplicated `omp.map.info` and target clause operands.
- MLIR Enums: `mlir/include/mlir/Dialect/OpenMP/OpenMPEnums.td` — add `ClauseMapFlagsIsDevicePtr` bit and include in `ClauseMapFlags`.
- MLIR Dialect: `mlir/lib/Dialect/OpenMP/IR/OpenMPDialect.cpp` — parser/print support for `is_device_ptr` map clause modifier.
- MLIR→LLVM Translation: `mlir/lib/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.cpp` — remove TODO check; translate `is_device_ptr` into `OMP_MAP_TARGET_PARAM` and conditionally `OMP_MAP_LITERAL`; mark `DevicePointers` element kind accordingly.
- MLIR Tests: `mlir/test/Target/LLVMIR/omptarget-llvm.mlir` — add end-to-end test for `is_device_ptr`; `openmp-todo.mlir` remove obsolete expected errors.

Pitfalls & Reviewer Notes
- Map duplication: Ensure duplication occurs after the original `MapInfoOp` to keep SSA use-def and finalization passes happy.
- Duplicate detection: Include `isDevicePtrSyms` in duplicate-mapped-symbol checks to avoid contradictory mappings.
- Runtime flags: Only set `OMP_MAP_LITERAL` when there is no other explicit map on the operand; preserve `TARGET_PARAM` for `is_device_ptr`.
- Interactions with `has_device_addr`: Synthesize it only if not already present to avoid redundant entries.

Related Issues/PRs
- Original change: #169331 (this PR relands it).

Source Code Links
- Flang: `ClauseProcessor.cpp` https://github.com/llvm/llvm-project/pull/170851/files#diff-55798c5090a8f8499f773b3fb46fb98a7aabe0f58ec60ff295e107acb36c7707
- Flang: `ClauseProcessor.h` https://github.com/llvm/llvm-project/pull/170851/files#diff-68e0b910d8ab50461fbd44f967c20b09881d98da534eb78f3dc88dd254a1904b
- Flang: `OpenMP.cpp` https://github.com/llvm/llvm-project/pull/170851/files#diff-496a295679ae3c43f8651c944a1bd9dca177ad2b5e4d7121f96938024e292bc1
- MLIR: `OpenMPEnums.td` https://github.com/llvm/llvm-project/pull/170851/files#diff-46cd10c37bc5b882a52ea72e11c245202a338d50c6f0f5a8d31de5195f115253
- MLIR: `OpenMPDialect.cpp` https://github.com/llvm/llvm-project/pull/170851/files#diff-a897370ad8f5ad37e8c1adb3c145c2304aaa38da3227bc1d02ac701ee8dc0754
- MLIR: `OpenMPToLLVMIRTranslation.cpp` https://github.com/llvm/llvm-project/pull/170851/files#diff-2cbb5651f4570d81d55ac4198deda0f6f7341b2503479752ef2295da3774c586
- Tests: `map-types-and-sizes.f90`, `target.f90`, `omptarget-llvm.mlir`, `openmp-todo.mlir`

How To Avoid Pitfalls
- Clone `MapInfoOp` users for `is_device_ptr` variables when augmenting `has_device_addr` to maintain unique user invariants.
- Thread `isDevicePtrSyms` through duplicate mapping checks to prevent conflicting mappings.
- Validate pointer-kind handling in translation to ensure device pointer vs address distinctions reach the runtime.

---
## [flang][OpenMP] Implement COMBINER clause

Overview
- Purpose: Introduce the OpenMP 6.0 `COMBINER(...)` clause support in Flang for `declare reduction`, representing the combiner expression as a first-class clause to unify lowering. Reuse existing combiner-expression evaluation to build the reduction combiner callback.
- Scope: Flang parser and utilities, clause lowering helpers, declare-reduction lowering flow; minor structure/test updates.

Spec Reference
- OpenMP 6.0 — `declare reduction` with `COMBINER(combiner-expr)` clause; stylized instances permit `omp_out`/`omp_in` parameters and optional variable decls.

Semantics
- If a `COMBINER` clause appears, treat it as the authoritative combiner definition for the `declare reduction` identifier.
- If missing, synthesize an internal `combiner` clause from the reduction specifier’s combiner expression so downstream lowering paths are uniform. The synthesized clause’s source location points at the original combiner expression for accurate debug info.
- Build the combiner callback by evaluating the stylized instance with `omp_out`/`omp_in`, handling by-ref/by-val appropriately.
- The initializer clause is still required in current implementation; declare-reduction without an initializer emits a TODO diagnostic.

Reference
- Implementation PR: https://github.com/llvm/llvm-project/pull/172036 (merged)
- Summary: Add `OmpCombinerClause` to the parser, wire `Combiner` through lowering with `StylizedInstance`, introduce `appendCombiner` to normalize presence of combiner, update declare-reduction lowering to consume the clause, and adjust helpers/utilities/tests.

## Change Log (PR #172036)
- Flang Lowering Types: `flang/include/flang/Lower/OpenMP/Clauses.h` — add aliases and helpers for stylized instances and combiner clause.
```diff
@@ -104,6 +104,7 @@ struct hash<Fortran::lower::omp::IdTy> {
 namespace Fortran::lower::omp {
	 using Object = tomp::ObjectT<IdTy, ExprTy>;
	 using ObjectList = tomp::ObjectListT<IdTy, ExprTy>;
 + using StylizedInstance = tomp::type::StylizedInstanceT<IdTy, ExprTy>;
@@
	 std::optional<Object> getBaseObject(const Object &object,
																				semantics::SemanticsContext &semaCtx);
 + StylizedInstance makeStylizedInstance(const parser::OmpStylizedInstance &inp,
 +                                       semantics::SemanticsContext &semaCtx);
@@
	 using Collector = tomp::clause::CollectorT<TypeTy, IdTy, ExprTy>;
 + using Combiner = tomp::clause::CombinerT<TypeTy, IdTy, ExprTy>;
	 using Compare = tomp::clause::CompareT<TypeTy, IdTy, ExprTy>;
```
- Parser Dump: `flang/include/flang/Parser/dump-parse-tree.h` — print the combiner clause node.
```diff
@@ -562,6 +562,7 @@ class ParseTreeDumper {
	 NODE(parser, OmpClauseList)
	 NODE(parser, OmpCloseModifier)
	 NODE_ENUM(OmpCloseModifier, Value)
 + NODE(parser, OmpCombinerClause)
	 NODE(parser, OmpCombinerExpression)
```
- Parser Utils: `flang/include/flang/Parser/openmp-utils.h` — provide overloads to retrieve combiner/initializer from either specifier or clause.
```diff
@@ -226,9 +226,9 @@
- const OmpCombinerExpression *GetCombinerExpr(
-     const OmpReductionSpecifier &rspec);
- const OmpInitializerExpression *GetInitializerExpr(const OmpClause &init);
+ const OmpCombinerExpression *GetCombinerExpr(const OmpReductionSpecifier &x);
+ const OmpCombinerExpression *GetCombinerExpr(const OmpClause &x);
+ const OmpInitializerExpression *GetInitializerExpr(const OmpClause &x);
```
- Parser AST: `flang/include/flang/Parser/parse-tree.h` — add `OmpCombinerClause` wrapper (since 6.0) around `OmpCombinerExpression`.
```diff
@@ -4395,6 +4395,14 @@
 // combiner-clause -> // since 6.0
 // COMBINER(combiner-expr)
 struct OmpCombinerClause {
	 WRAPPER_CLASS_BOILERPLATE(OmpCombinerClause, OmpCombinerExpression);
 };
```
- ClauseProcessor: `flang/lib/Lower/OpenMP/ClauseProcessor.cpp` — use `StylizedInstance` in initializer processing and emit a TODO if initializer is absent.
```diff
@@ -390,10 +390,10 @@ bool ClauseProcessor::processInitializer(
-  const clause::StylizedInstance &inst = clause->v.front();
+  const StylizedInstance &inst = clause->v.front();
@@ -412,7 +412,7 @@
-  const semantics::SomeExpr &initExpr =
-      std::get<clause::StylizedInstance::Instance>(inst.t);
+  const semantics::SomeExpr &initExpr =
+      std::get<StylizedInstance::Instance>(inst.t);
@@ -439,7 +439,9 @@
-  return false;
+  TODO(converter.getCurrentLocation(),
+       "declare reduction without an initializer clause is not yet supported");
```
- Clause Helpers: `flang/lib/Lower/OpenMP/Clauses.cpp` — introduce `makeStylizedInstance`, use it for both `Combiner` and `Initializer`.
```diff
@@ -197,6 +197,24 @@ std::optional<Object> getBaseObject(...)
 + StylizedInstance makeStylizedInstance(const parser::OmpStylizedInstance &inp,
 +                                       semantics::SemanticsContext &semaCtx) { ... }
@@
 + Combiner make(const parser::OmpClause::Combiner &inp,
 +               semantics::SemanticsContext &semaCtx) {
 +   const parser::OmpCombinerExpression &cexpr = inp.v.v;
 +   Combiner combiner;
 +   for (const parser::OmpStylizedInstance &sinst : cexpr.v)
 +     combiner.v.push_back(makeStylizedInstance(sinst, semaCtx));
 +   return combiner;
 + }
@@ -988,24 +1017,8 @@ Initializer make(...)
- for (const parser::OmpStylizedInstance &sinst : iexpr.v) { ... }
+ for (const parser::OmpStylizedInstance &sinst : iexpr.v)
+   initializer.v.push_back(makeStylizedInstance(sinst, semaCtx));
```
- Declare Reduction Lowering: `flang/lib/Lower/OpenMP/OpenMP.cpp` — normalize/synthesize combiner clause, then generate combiner callback from clause.
```diff
@@ -3605,10 +3607,11 @@
- static ReductionProcessor::GenCombinerCBTy processReductionCombiner(...,
-     const parser::OmpReductionSpecifier &specifier) {
+ static ReductionProcessor::GenCombinerCBTy processReductionCombiner(...,
+     const clause::Combiner &combiner) {
@@
- const auto &combinerExpression = ...
- const parser::OmpStylizedInstance &combinerInstance = ...
+ const StylizedInstance &inst = combiner.v.front();
+ semantics::SomeExpr evalExpr = std::get<StylizedInstance::Instance>(inst.t);
@@
- for (const parser::OmpStylizedDeclaration &decl : declList) { ... }
+ for (const Object &object : std::get<StylizedInstance::Variables>(inst.t)) { ... }
@@ -3714,0 +3718,27 @@
+ // Ensure there's a combiner clause; synthesize one from the specifier if absent.
+ static const clause::Combiner &appendCombiner(
+     const parser::OpenMPDeclareReductionConstruct &construct,
+     List<Clause> &clauses, semantics::SemanticsContext &semaCtx) { ... }
@@ -3742,6 +3776,6 @@ static void genOMP(..., const parser::OpenMPDeclareReductionConstruct &construct)
- ReductionProcessor::GenCombinerCBTy genCombinerCB =
-   processReductionCombiner(converter, symTable, semaCtx, specifier);
- if (initializer.v.size() > 0) { ... } else { TODO(...); }
+ List<Clause> clauses = makeClauses(construct.v.Clauses(), semaCtx);
+ const clause::Combiner &combiner = appendCombiner(construct, clauses, semaCtx);
+ ReductionProcessor::GenCombinerCBTy genCombinerCB =
+   processReductionCombiner(converter, symTable, semaCtx, combiner);
+ ClauseProcessor cp(converter, semaCtx, clauses);
+ cp.processInitializer(symTable, genInitValueCB);
```
- Structure/Test Updates:
	- `flang/lib/Semantics/check-omp-structure.cpp` — recognize `COMBINER` clause in structure checking (Files tab).
	- `flang/test/Lower/OpenMP/declare-reduction-combiner.f90` — new coverage for declare reduction with explicit combiner clause.
	- Minor headers/types: `ClauseT.h`, `OMP.td` touched for clause plumbing.

Pitfalls & Reviewer Notes
- Versioning: This introduces an OpenMP 6.0 clause. The implementation synthesizes an internal `combiner` clause even when compiling older OpenMP versions; reviewers questioned debug info impact. Author confirmed the clause’s source location points to the original combiner expression, avoiding debug info pollution. If you enable pretty-printing/round-trips, ensure version-aware gating to not emit 6.0 syntax accidentally.
- Initializer requirement: `declare reduction` without an initializer currently triggers a TODO diagnostic in `processInitializer`. Provide an initializer until support is extended.
- Symbol mapping: The combiner expects `omp_out`/`omp_in` bindings. Lowering now maps stylized declarations via `tomp::Object` entries; ensure semantic validation prevents unexpected names or missing symbols.
- By-ref vs by-val: The combiner callback handles pass-by-reference types; verify addressability is established (create temporaries when needed) for non-byref operands.

Related Issues/PRs
- OpenMP 6.0 combiner introduction (spec item). Consider interactions with user-defined reductions across versions.

Source Code Links
- Flang: `Clauses.h` https://github.com/llvm/llvm-project/pull/172036/files#diff-678c35fa95a6f92b3ccf755d1a13c3d9ce0d97938485cc4e0befcedcf7317431
- Flang: `ClauseProcessor.cpp` https://github.com/llvm/llvm-project/pull/172036/files#diff-55798c5090a8f8499f773b3fb46fb98a7aabe0f58ec60ff295e107acb36c7707
- Flang: `Clauses.cpp` https://github.com/llvm/llvm-project/pull/172036/files#diff-99d9e5b508d4a4d583713a6f04701826652f16e69bb03d98014b82be66fb5807
- Flang: `OpenMP.cpp` https://github.com/llvm/llvm-project/pull/172036/files#diff-496a295679ae3c43f8651c944a1bd9dca177ad2b5e4d7121f96938024e292bc1
- Parser: `parse-tree.h` https://github.com/llvm/llvm-project/pull/172036/files#diff-61d1be39226db8b7f54d3241269d9fabff009139551605546e33f6463bd4087a
- Parser: `dump-parse-tree.h` https://github.com/llvm/llvm-project/pull/172036/files#diff-062c8b30313609d423f975d2af9ec411e4b9fffbd782216ef55155564fd7649d
- Parser Utils: `openmp-utils.h` https://github.com/llvm/llvm-project/pull/172036/files#diff-51997fc66e1482fe156eacca8178e9d98023027796ec5ce254db012d51e545b9
- Semantics/Structure: `check-omp-structure.cpp` https://github.com/llvm/llvm-project/pull/172036/files#diff-be225bc038e95a229024c9f227284112f98fe31762bc0bb80605de425aadd084
- Tests: `declare-reduction-combiner.f90` https://github.com/llvm/llvm-project/pull/172036/files#diff-201f7b087b6c84fa02ad3e26033ce3a58a550d46837ef34075cc48f136d00607
- Misc: `ClauseT.h` https://github.com/llvm/llvm-project/pull/172036/files#diff-94ad6eac7c2c4bee1e4005bb660a43340224437e9ee4ab4266396bbffe9f1a81, `OMP.td` https://github.com/llvm/llvm-project/pull/172036/files#diff-b273fa9eb2357c7ff376e659e5e73c1f67859f750c3d0705ee6206200eed5bdf

How To Avoid Pitfalls
- Keep the combiner internal: prefer synthesizing/consuming the clause in lowering; avoid emitting 6.0 syntax in diagnostics/printing when targeting older versions.
- Always run `appendCombiner` in declare-reduction lowering so both explicit and implicit forms are handled identically downstream.
- Point the clause source location at the original combiner expression to preserve debug info fidelity.
- Provide an initializer for declare-reduction today; do not rely on the no-initializer path until implemented.

---

## [Flang][OpenMP] Implement workdistribute construct lowering

Overview
- Purpose: Lower Fortran array-notation style kernels inside `teams { workdistribute { ... } }` into executable OpenMP dialect using `teams { parallel { distribute { wsloop { ... } } } }`, including region fission and hoisting of non-parallel code for correctness on devices.
- Scope: New Flang Optimizer/OpenMP pass, verifier/legality checks, runtime-oriented lowering for array assignments, and tests.

Spec Reference
- OpenMP 6.0 — Workdistribute: worksharing semantics bound to the innermost TEAMS region; no device-wide barrier between arbitrary regions.

Semantics
- Transformations:
	- Fission/Hoist: Split `teams{workdistribute}` to isolate parallel parts and hoist legal non-OpenMP ops out of device kernels where possible.
	- Loop Lowering: Convert `fir.do_loop unordered` within workdistribute regions to `teams { parallel { distribute { wsloop { ... } } } }`.
	- Runtime Copies: Replace intrinsic array-to-array assigns with `omp.target_memcpy`; handle scalar-to-array assigns via specialized lowering; manage temporaries via `omp.target_allocmem/target_freemem`.
- Verifier ensures placement legality, supported shapes, and `omp.teams` binding.

Reference
- Implementation PR: https://github.com/llvm/llvm-project/pull/140523 (merged)

## Change Log (PR #140523)
- Flang Optimizer: `flang/lib/Optimizer/OpenMP/LowerWorkdistribute.cpp` — introduce pass to rewrite `teams{workdistribute}` regions; perform region fission, loop conversions, and runtime copy lowering.
```diff
@@
+#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
+#include "mlir/IR/PatternMatch.h"
+#include "mlir/Pass/Pass.h"
+using namespace mlir;
+namespace {
+struct LowerWorkdistributePass
+    : public PassWrapper<LowerWorkdistributePass, OperationPass<ModuleOp>> {
+  void runOnOperation() final {
+    ModuleOp mod = getOperation();
+    mod.walk([&](omp::TeamsOp teams) {
+      // Find nested workdistribute and lower to teams{parallel{distribute{wsloop}}}
+    });
+  }
+};
+} // namespace
```
- Build System: `flang/lib/Optimizer/OpenMP/CMakeLists.txt` — add the new source to the OpenMP optimizer library.
```diff
@@
-  OpenMPToFIR.cpp
+  OpenMPToFIR.cpp
+  LowerWorkdistribute.cpp
	 DEPENDS
			FIRDialect
	 )
```
- Tests: `flang/test/Optimizer/OpenMP/lower-workdistribute.fir` — check teams→parallel→distribute→wsloop nesting.
```diff
@@
// RUN: bbc -fopenmp -o - %s | FileCheck %s

func.func @kernel(%n: index, %A: memref<?xf32>, %B: memref<?xf32>) {
	omp.teams {
		// CHECK: omp.teams
		// CHECK: omp.parallel
		// CHECK: omp.distribute
		// CHECK: omp.wsloop
	}
	return
}
```

Pitfalls & Reviewer Notes
- Correctness vs performance: Without a device-wide barrier, some programs require splitting into multiple kernels; ensure cross-split values are preserved via temporaries and mapped appropriately.
- Nesting legality: Enforce that `workdistribute` binds to the innermost `teams` region; avoid silently fixing illegal nesting in lowering.
- Runtime APIs: Prefer `omp.target_memcpy` and allocator APIs over ad-hoc intrinsics to match OpenMP runtime semantics.

Related Issues/PRs
- Complements: #154376 (OpenMP dialect op), #154377 (Flang parser/semantics), #154378 (Flang lowering to dialect op).

Source Code Links
- Files changed view: https://github.com/llvm/llvm-project/pull/140523/files

How To Avoid Pitfalls
- Validate preconditions before transforming; when inserting splits, materialize per-team temporaries for values live across kernels.
- Keep semantic enforcement in parser/semantics; lowering should assume verified inputs and avoid silent legalization.

---

## [OpenMP] Add workdistribute construct in openMP dialect and in llvm frontend

Overview
- Purpose: Introduce `omp.workdistribute` operation in the MLIR OpenMP dialect and add corresponding directive spellings in LLVM’s `OMP.td`. Provide verifier, parser/printer, and tests.
- Scope: MLIR dialect op definition and verifier; LLVM OMP.td directive additions; dialect tests.

Spec Reference
- OpenMP 6.0 — Workdistribute: block construct nested under `teams`; structured block with specific semantics, no implicit barrier.

Semantics
- `workdistribute` divides execution of the enclosed structured block into separate units, each executed once by each initial thread in the league; must be nested directly under `teams`.
- Region requirements: single entry, single exit (`omp.terminator`), no explicit `omp.barrier`, no nested `omp.parallel` or nested `omp.teams`.

Reference
- Implementation PR: https://github.com/llvm/llvm-project/pull/154376 (merged)

## Change Log (PR #154376)
- LLVM OpenMP Directives: `llvm/include/llvm/Frontend/OpenMP/OMP.td` — add spellings for `workdistribute`, `end workdistribute`, and combined `target teams workdistribute`/`teams workdistribute` directives.
```diff
@@ -1322,6 +1322,17 @@ def OMP_EndWorkshare : Directive<[Spelling<"end workshare">]> {
	 let category = OMP_Workshare.category;
	 let languages = [L_Fortran];
 }
 + def OMP_Workdistribute : Directive<[Spelling<"workdistribute">]> {
 +   let association = AS_Block;
 +   let category = CA_Executable;
 +   let languages = [L_Fortran];
 + }
 + def OMP_EndWorkdistribute : Directive<[Spelling<"end workdistribute">]> {
 +   let leafConstructs = OMP_Workdistribute.leafConstructs;
 +   let association = OMP_Workdistribute.association;
 +   let category = OMP_Workdistribute.category;
 +   let languages = [L_Fortran];
 + }
@@ -2482,6 +2493,35 @@ def OMP_TargetTeamsDistributeSimd
	 let leafConstructs = [OMP_Target, OMP_Teams, OMP_Distribute, OMP_Simd];
	 let category = CA_Executable;
 }
 + def OMP_TargetTeamsWorkdistribute : Directive<[Spelling<"target teams workdistribute">]> {
 +   let allowedClauses = [
 +     VersionedClause<OMPC_Allocate>,
 +     VersionedClause<OMPC_Depend>,
 +     VersionedClause<OMPC_FirstPrivate>,
 +     VersionedClause<OMPC_HasDeviceAddr, 51>,
 +     VersionedClause<OMPC_If>,
 +     VersionedClause<OMPC_IsDevicePtr>,
 +     VersionedClause<OMPC_Map>,
 +     VersionedClause<OMPC_OMPX_Attribute>,
 +     VersionedClause<OMPC_Private>,
 +     VersionedClause<OMPC_Reduction>,
 +     VersionedClause<OMPC_Shared>,
 +     VersionedClause<OMPC_UsesAllocators, 50>,
 +   ];
 +   let allowedOnceClauses = [
 +     VersionedClause<OMPC_Default>,
 +     VersionedClause<OMPC_DefaultMap>,
 +     VersionedClause<OMPC_Device>,
 +     VersionedClause<OMPC_NoWait>,
 +     VersionedClause<OMPC_NumTeams>,
 +     VersionedClause<OMPC_OMPX_DynCGroupMem>,
 +     VersionedClause<OMPC_OMPX_Bare>,
 +     VersionedClause<OMPC_ThreadLimit>,
 +   ];
 +   let leafConstructs = [OMP_Target, OMP_Teams, OMP_Workdistribute];
 +   let category = CA_Executable;
 +   let languages = [L_Fortran];
 + }
@@ -2723,6 +2763,25 @@ def OMP_TeamsDistributeSimd : Directive<[Spelling<"teams distribute simd">]> {
	 let leafConstructs = [OMP_Teams, OMP_Distribute, OMP_Simd];
	 let category = CA_Executable;
 }
 + def OMP_TeamsWorkdistribute : Directive<[Spelling<"teams workdistribute">]> {
 +   let allowedClauses = [
 +     VersionedClause<OMPC_Allocate>,
 +     VersionedClause<OMPC_FirstPrivate>,
 +     VersionedClause<OMPC_OMPX_Attribute>,
 +     VersionedClause<OMPC_Private>,
 +     VersionedClause<OMPC_Reduction>,
 +     VersionedClause<OMPC_Shared>,
 +   ];
 +   let allowedOnceClauses = [
 +     VersionedClause<OMPC_Default>,
 +     VersionedClause<OMPC_If, 52>,
 +     VersionedClause<OMPC_NumTeams>,
 +     VersionedClause<OMPC_ThreadLimit>,
 +   ];
 +   let leafConstructs = [OMP_Teams, OMP_Workdistribute];
 +   let category = CA_Executable;
 +   let languages = [L_Fortran];
 + }
```
- MLIR Dialect Op: `mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td` — define `WorkdistributeOp` with verifier and assembly format.
```diff
@@ -2209,4 +2209,27 @@ def TargetFreeMemOp : OpenMP_Op<"target_freemem",
	 let assemblyFormat = "$device `,` $heapref attr-dict `:` type($device) `,` qualified(type($heapref))";
 }
   
 + //===----------------------------------------------------------------------===//
 + // workdistribute Construct
 + //===----------------------------------------------------------------------===//
 +
 + def WorkdistributeOp : OpenMP_Op<"workdistribute"> {
 +   let summary = "workdistribute directive";
 +   let description = [{
 +     workdistribute divides execution of the enclosed structured block into
 +     separate units of work, each executed only once by each
 +     initial thread in the league.
 +     ```
 +     !$omp target teams
 +     !$omp workdistribute
 +     y = a * x + y  
 +     !$omp end workdistribute
 +     !$omp end target teams
 +     ```
 +   }];
 +   let regions = (region AnyRegion:$region);
 +   let hasVerifier = 1;
 +   let assemblyFormat = "$region attr-dict";
 + }
@@ -2212,2235 @@ #endif // OPENMP_OPS
```
- MLIR Verifier: `mlir/lib/Dialect/OpenMP/IR/OpenMPDialect.cpp` — implement `WorkdistributeOp::verify` with nesting and region checks.
```diff
@@ -3975,6 +3975,58 @@ llvm::LogicalResult omp::TargetAllocMemOp::verify() {
	 return mlir::success();
 }
   
 + //===----------------------------------------------------------------------===//
 + // WorkdistributeOp
 + //===----------------------------------------------------------------------===//
 +
 + LogicalResult WorkdistributeOp::verify() {
 +   // Check that region exists and is not empty
 +   Region &region = getRegion();
 +   if (region.empty())
 +     return emitOpError("region cannot be empty");
 +   // Verify single entry point.
 +   Block &entryBlock = region.front();
 +   if (entryBlock.empty())
 +     return emitOpError("region must contain a structured block");
 +   // Verify single exit point.
 +   bool hasTerminator = false;
 +   for (Block &block : region) {
 +     if (isa<TerminatorOp>(block.back())) {
 +       if (hasTerminator) {
 +         return emitOpError("region must have exactly one terminator");
 +       }
 +       hasTerminator = true;
 +     }
 +   }
 +   if (!hasTerminator) {
 +     return emitOpError("region must be terminated with omp.terminator");
 +   }
 +   auto walkResult = region.walk([&](Operation *op) -> WalkResult {
 +     // No implicit barrier at end
 +     if (isa<BarrierOp>(op)) {
 +       return emitOpError(
 +           "explicit barriers are not allowed in workdistribute region");
 +     }
 +     // Check for invalid nested constructs
 +     if (isa<ParallelOp>(op)) {
 +       return emitOpError(
 +           "nested parallel constructs not allowed in workdistribute");
 +     }
 +     if (isa<TeamsOp>(op)) {
 +       return emitOpError(
 +           "nested teams constructs not allowed in workdistribute");
 +     }
 +     return WalkResult::advance();
 +   });
 +   if (walkResult.wasInterrupted())
 +     return failure();
 +
 +   Operation *parentOp = (*this)->getParentOp();
 +   if (!llvm::dyn_cast<TeamsOp>(parentOp))
 +     return emitOpError("workdistribute must be nested under teams");
 +   return success();
 + }
```
- MLIR Tests (invalid): `mlir/test/Dialect/OpenMP/invalid.mlir` — add error cases for empty region, missing terminator, multiple terminators, barriers, nested parallel/teams, and missing teams nesting.
```diff
@@ -3017,3 +3119,110 @@ func.func @invalid_allocate_allocator(%arg0 : memref<i32>) -> () {
	 return
 }
 +
 + // -----
 + func.func @invalid_workdistribute_empty_region() -> () {
 +   omp.teams {
 +     // expected-error @below {{region cannot be empty}}
 +     omp.workdistribute {
 +     }
 +     omp.terminator
 +   }
 +   return
 + }
 +
 + // -----
 + func.func @invalid_workdistribute_no_terminator() -> () {
 +   omp.teams {
 +     // expected-error @below {{region must be terminated with omp.terminator}}
 +     omp.workdistribute {
 +       %c0 = arith.constant 0 : i32
 +     }
 +     omp.terminator
 +   }
 +   return
 + }
 +
 + // -----
 + func.func @invalid_workdistribute_wrong_terminator() -> () {
 +   omp.teams {
 +     // expected-error @below {{region must be terminated with omp.terminator}}
 +     omp.workdistribute {
 +       %c0 = arith.constant 0 : i32
 +       func.return
 +     }
 +     omp.terminator
 +   }
 +   return
 + }
 +
 + // -----
 + func.func @invalid_workdistribute_multiple_terminators() -> () {
 +   omp.teams {
 +     // expected-error @below {{region must have exactly one terminator}}
 +     omp.workdistribute {
 +       %cond = arith.constant true
 +       cf.cond_br %cond, ^bb1, ^bb2
 +       ^bb1:
 +       omp.terminator
 +       ^bb2:
 +       omp.terminator
 +     }
 +     omp.terminator
 +   }
 +   return
 + }
 +
 + // -----
 + func.func @invalid_workdistribute_with_barrier() -> () {
 +   omp.teams {
 +     // expected-error @below {{explicit barriers are not allowed in workdistribute region}}
 +     omp.workdistribute {
 +       %c0 = arith.constant 0 : i32
 +       omp.barrier
 +       omp.terminator
 +     }
 +     omp.terminator
 +   }
 +   return
 + }
 +
 + // -----
 + func.func @invalid_workdistribute_nested_parallel() -> () {
 +   omp.teams {
 +     // expected-error @below {{nested parallel constructs not allowed in workdistribute}}
 +     omp.workdistribute {
 +       omp.parallel {
 +         omp.terminator
 +       }
 +       omp.terminator
 +     }
 +     omp.terminator
 +   }
 +   return
 + }
 +
 + // Test: nested teams not allowed in workdistribute
 + func.func @invalid_workdistribute_nested_teams() -> () {
 +   omp.teams {
 +     // expected-error @below {{nested teams constructs not allowed in workdistribute}}
 +     omp.workdistribute {
 +       omp.teams {
 +         omp.terminator
 +       }
 +       omp.terminator
 +     }
 +     omp.terminator
 +   }
 +   return
 + }
 +
 + // -----
 + func.func @invalid_workdistribute() -> () {
 +   // expected-error @below {{workdistribute must be nested under teams}}
 +   omp.workdistribute {
 +     omp.terminator
 +   }
 +   return
 + }
```
- MLIR Tests (ops): `mlir/test/Dialect/OpenMP/ops.mlir` — add positive construction under teams.
```diff
@@ -3238,3 +3252,15 @@ func.func @omp_allocate_dir(%arg0 : memref<i32>, %arg1 : memref<i32>) -> () {
	 return
 }
 +
 + // CHECK-LABEL: func.func @omp_workdistribute
 + func.func @omp_workdistribute() {
 +   // CHECK: omp.teams
 +   omp.teams {
 +     // CHECK: omp.workdistribute
 +     omp.workdistribute {
 +       omp.terminator
 +     }
 +     omp.terminator
 +   }
 +   return
 + }
```

Pitfalls & Reviewer Notes
- Ensure verifier enforces all region properties and teams nesting; tests cover common violations.
- Keep dialect printer/parser synchronized with ODS; assembly format is region-only.
- Directive spellings added only for Fortran in `OMP.td` per scope.

Related Links
- OpenMP 6.0 Workdistribute definition and nesting constraints.

---

## [flang][openmp] Add parser/semantic support for workdistribute

Overview
- Purpose: Parse OpenMP 6.0 `WORKDISTRIBUTE` (and combined `TEAMS WORKDISTRIBUTE`, `TARGET TEAMS WORKDISTRIBUTE`) and enforce structure rules: nesting under TEAMS only; body is assignment-only; version gating (requires -fopenmp-version=60).
- Scope: Parser (`openmp-parsers.cpp`), directive sets (`openmp-directive-sets.h`), semantics structure checks (`check-omp-structure.cpp/.h`, `resolve-directives.cpp`), and tests.

Spec Reference
- OpenMP 6.0 — `workdistribute` block construct semantics and constraints.

Reference
- Implementation PR: https://github.com/llvm/llvm-project/pull/154377 (merged)

## Change Log (PR #154377)
- Directive Sets: `flang/include/flang/Semantics/openmp-directive-sets.h` — include new directives in target/teams sets and block construct set; allow nested teams to contain `workdistribute`.
```diff
@@ -143,6 +143,7 @@ static const OmpDirectiveSet topTargetSet{
	 Directive::OMPD_target_teams_distribute_parallel_do_simd,
	 Directive::OMPD_target_teams_distribute_simd,
	 Directive::OMPD_target_teams_loop,
 + Directive::OMPD_target_teams_workdistribute,
 };
@@ -172,6 +173,7 @@ static const OmpDirectiveSet topTeamsSet{
	 Directive::OMPD_teams_distribute_parallel_do_simd,
	 Directive::OMPD_teams_distribute_simd,
	 Directive::OMPD_teams_loop,
 + Directive::OMPD_teams_workdistribute,
 };
@@ -187,6 +189,7 @@ static const OmpDirectiveSet allTeamsSet{
	 Directive::OMPD_target_teams_distribute_parallel_do_simd,
	 Directive::OMPD_target_teams_distribute_simd,
	 Directive::OMPD_target_teams_loop,
 + Directive::OMPD_target_teams_workdistribute,
 } | topTeamsSet,
@@ -230,6 +233,9 @@ static const OmpDirectiveSet blockConstructSet{
	 Directive::OMPD_taskgroup,
	 Directive::OMPD_teams,
	 Directive::OMPD_workshare,
 + Directive::OMPD_target_teams_workdistribute,
 + Directive::OMPD_teams_workdistribute,
 + Directive::OMPD_workdistribute,
 };
@@ -376,6 +382,7 @@ static const OmpDirectiveSet nestedReduceWorkshareAllowedSet{
 };
@@ -378,384 | static const OmpDirectiveSet nestedTeamsAllowedSet{
 + Directive::OMPD_workdistribute,
	 Directive::OMPD_distribute,
	 Directive::OMPD_distribute_parallel_do,
	 Directive::OMPD_distribute_parallel_do_simd,
 }
```
- Parser: `flang/lib/Parser/openmp-parsers.cpp` — add block construct productions for `target teams workdistribute`, `teams workdistribute`, and `workdistribute`.
```diff
@@ -1870,11 +1870,15 @@ TYPE_PARSER( //
	 MakeBlockConstruct(llvm::omp::Directive::OMPD_target_data) ||
	 MakeBlockConstruct(llvm::omp::Directive::OMPD_target_parallel) ||
	 MakeBlockConstruct(llvm::omp::Directive::OMPD_target_teams) ||
 + MakeBlockConstruct(
 +   llvm::omp::Directive::OMPD_target_teams_workdistribute) ||
	 MakeBlockConstruct(llvm::omp::Directive::OMPD_target) ||
	 MakeBlockConstruct(llvm::omp::Directive::OMPD_task) ||
	 MakeBlockConstruct(llvm::omp::Directive::OMPD_taskgroup) ||
	 MakeBlockConstruct(llvm::omp::Directive::OMPD_teams) ||
 - MakeBlockConstruct(llvm::omp::Directive::OMPD_workshare))
 + MakeBlockConstruct(llvm::omp::Directive::OMPD_teams_workdistribute) ||
 + MakeBlockConstruct(llvm::omp::Directive::OMPD_workshare) ||
 + MakeBlockConstruct(llvm::omp::Directive::OMPD_workdistribute))
 #undef MakeBlockConstruct
```
- Semantics Structure: `flang/lib/Semantics/check-omp-structure.cpp` — add checks for teams-only nesting and assignment-only body; gate by OpenMP version.
```diff
@@ -815,6 +873,12 @@ void OmpStructureChecker::Enter(const parser::OpenMPBlockConstruct &x) {
	 "TARGET construct with nested TEAMS region contains statements or "
	 "directives outside of the TEAMS construct"_err_en_US);
 }
 + if (GetContext().directive == llvm::omp::Directive::OMPD_workdistribute &&
 +     GetContextParent().directive != llvm::omp::Directive::OMPD_teams) {
 +   context_.Say(x.BeginDir().DirName().source,
 +     "%s region can only be strictly nested within TEAMS region"_err_en_US,
 +     ContextDirectiveAsFortran());
 + }
@@ -898,6 +962,17 @@ void OmpStructureChecker::Enter(const parser::OpenMPBlockConstruct &x) {
	 HasInvalidWorksharingNesting(
	 beginSpec.source, llvm::omp::nestedWorkshareErrSet);
	 break;
 + case llvm::omp::OMPD_workdistribute:
 +   if (!CurrentDirectiveIsNested()) {
 +     context_.Say(beginSpec.source,
 +       "A WORKDISTRIBUTE region must be nested inside TEAMS region only."_err_en_US);
 +   }
 +   CheckWorkdistributeBlockStmts(block, beginSpec.source);
 +   break;
 + case llvm::omp::OMPD_teams_workdistribute:
 + case llvm::omp::OMPD_target_teams_workdistribute:
 +   CheckWorkdistributeBlockStmts(block, beginSpec.source);
 +   break;
```
- Semantics Helpers: `flang/lib/Semantics/check-omp-structure.h` — declare helper.
```diff
@@ -245,6 +245,7 @@ class OmpStructureChecker
	 bool CheckTargetBlockOnlyTeams(const parser::Block &);
	 void CheckWorkshareBlockStmts(const parser::Block &, parser::CharBlock);
 + void CheckWorkdistributeBlockStmts(const parser::Block &, parser::CharBlock);
```
- Semantics Helpers impl: `flang/lib/Semantics/check-omp-structure.cpp` — implement version gate and assignment-only enforcement.
```diff
@@ -4546,6 +4621,27 @@ void OmpStructureChecker::CheckWorkshareBlockStmts(
 }
 
 + void OmpStructureChecker::CheckWorkdistributeBlockStmts(
 +     const parser::Block &block, parser::CharBlock source) {
 +   unsigned version{context_.langOptions().OpenMPVersion};
 +   unsigned since{60};
 +   if (version < since)
 +     context_.Say(source,
 +       "WORKDISTRIBUTE construct is not allowed in %s, %s"_err_en_US,
 +       ThisVersion(version), TryVersion(since));
 +
 +   OmpWorkdistributeBlockChecker ompWorkdistributeBlockChecker{context_, source};
 +
 +   for (auto it{block.begin()}; it != block.end(); ++it) {
 +     if (parser::Unwrap<parser::AssignmentStmt>(*it)) {
 +       parser::Walk(*it, ompWorkdistributeBlockChecker);
 +     } else {
 +       context_.Say(source,
 +         "The structured block in a WORKDISTRIBUTE construct may consist of only SCALAR or ARRAY assignments"_err_en_US);
 +     }
 +   }
 + }
```
- Resolve/Context: `flang/lib/Semantics/resolve-directives.cpp` — push/pop context for workdistribute variants.
```diff
@@ -1740,10 +1740,13 @@ bool OmpAttributeVisitor::Pre(const parser::OpenMPBlockConstruct &x) {
	 case llvm::omp::Directive::OMPD_task:
	 case llvm::omp::Directive::OMPD_taskgroup:
	 case llvm::omp::Directive::OMPD_teams:
 + case llvm::omp::Directive::OMPD_workdistribute:
	 case llvm::omp::Directive::OMPD_workshare:
	 case llvm::omp::Directive::OMPD_parallel_workshare:
	 case llvm::omp::Directive::OMPD_target_teams:
 + case llvm::omp::Directive::OMPD_target_teams_workdistribute:
	 case llvm::omp::Directive::OMPD_target_parallel:
 + case llvm::omp::Directive::OMPD_teams_workdistribute: {
	 PushContext(dirSpec.source, dirId);
	 break;
 }
@@ -1773,9 +1776,12 @@ void OmpAttributeVisitor::Post(const parser::OpenMPBlockConstruct &x) {
	 case llvm::omp::Directive::OMPD_target:
	 case llvm::omp::Directive::OMPD_task:
	 case llvm::omp::Directive::OMPD_teams:
 + case llvm::omp::Directive::OMPD_workdistribute:
	 case llvm::omp::Directive::OMPD_parallel_workshare:
	 case llvm::omp::Directive::OMPD_target_teams:
 - case llvm::omp::Directive::OMPD_target_parallel: {
 + case llvm::omp::Directive::OMPD_target_parallel:
 + case llvm::omp::Directive::OMPD_target_teams_workdistribute:
 + case llvm::omp::Directive::OMPD_teams_workdistribute: {
	 bool hasPrivate;
	 for (const auto *allocName : allocateNames_) {
		 hasPrivate = false;
```
- Parser/Semantics Tests: add parsing/unparsing and error tests.
```diff
@@ -0,0 +27 @@ flang/test/Parser/OpenMP/workdistribute.f90
 + !RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=60 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
 + !RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=60 %s | FileCheck --check-prefix="PARSE-TREE" %s
 +
 + !UNPARSE: !$OMP TEAMS WORKDISTRIBUTE
 + !PARSE-TREE: OmpDirectiveName -> llvm::omp::Directive = teams workdistribute
 + !$omp teams workdistribute
 + y = a * x + y
 + !$omp end teams workdistribute
```
```diff
@@ -0,0 +16 @@ flang/test/Semantics/OpenMP/workdistribute01.f90
 + !ERROR: A WORKDISTRIBUTE region must be nested inside TEAMS region only.
 + !ERROR: The structured block in a WORKDISTRIBUTE construct may consist of only SCALAR or ARRAY assignments
 + !$omp workdistribute
 + do i = 1, n
 +   print *, "omp workdistribute"
 + end do
 + !$omp end workdistribute
```
```diff
@@ -0,0 +34 @@ flang/test/Semantics/OpenMP/workdistribute02.f90
 + !ERROR: User defined non-ELEMENTAL function 'my_func' is not allowed in a WORKDISTRIBUTE construct
 + !$omp teams
 + !$omp workdistribute
 + aa = my_func()
 + aa = bb * cc
 + !$omp end workdistribute
 + !$omp end teams
```
```diff
@@ -0,0 +34 @@ flang/test/Semantics/OpenMP/workdistribute03.f90
 + !ERROR: Defined assignment statement is not allowed in a WORKDISTRIBUTE construct
 + !$omp teams
 + !$omp workdistribute
 + a = l
 + aa = bb
 + !$omp end workdistribute
 + !$omp end teams
```
```diff
@@ -0,0 +15 @@ flang/test/Semantics/OpenMP/workdistribute04.f90
 + ! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=50
 + !ERROR: WORKDISTRIBUTE construct is not allowed in OpenMP v5.0, try -fopenmp-version=60
 + !$omp teams workdistribute
 + y = a * x + y
 + !$omp end teams workdistribute
```

Pitfalls & Reviewer Notes
- Body restriction: Only scalar/array assignments allowed; defined assignments and non-elemental/impure functions diagnosed.
- Version gating: Emit diagnostic when compiling with OpenMP < 6.0.
- Nesting: Enforce TEAMS-only nesting for standalone WORKDISTRIBUTE.

---

## [flang][openmp] Add Lowering to omp mlir for workdistribute construct

Overview
- Purpose: Lower the parsed `WORKDISTRIBUTE` constructs to MLIR `omp.workdistribute`, and handle composite forms `TEAMS WORKDISTRIBUTE` / `TARGET TEAMS WORKDISTRIBUTE` by processing team clauses.
- Scope: Flang lowering in `OpenMP.cpp` and tests.

Reference
- Implementation PR: https://github.com/llvm/llvm-project/pull/154378 (merged)

## Change Log (PR #154378)
- Flang Lowering: `flang/lib/Lower/OpenMP/OpenMP.cpp` — dispatch and generation helpers for `workdistribute` and composite forms.
```diff
@@ -534,6 +543,13 @@ static void processHostEvalClauses(lower::AbstractConverter &converter,
	 cp.processCollapse(loc, eval, hostInfo->ops, hostInfo->iv);
	 break;
 
 + case OMPD_teams_workdistribute:
 +   cp.processThreadLimit(stmtCtx, hostInfo->ops);
 +   [[fallthrough]];
 + case OMPD_target_teams_workdistribute:
 +   cp.processNumTeams(stmtCtx, hostInfo->ops);
 +   break;
```
```diff
@@ -2820,6 +2827,17 @@ genTeamsOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
 }
 
 + static mlir::omp::WorkdistributeOp genWorkdistributeOp(
 +     lower::AbstractConverter &converter, lower::SymMap &symTable,
 +     semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
 +     mlir::Location loc, const ConstructQueue &queue,
 +     ConstructQueue::const_iterator item) {
 +   return genOpWithBody<mlir::omp::WorkdistributeOp>(
 +       OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
 +                         llvm::omp::Directive::OMPD_workdistribute),
 +       queue, item);
 + }
```
```diff
@@ -3459,7 +3477,10 @@ static void genOMPDispatch(lower::AbstractConverter &converter,
	 case llvm::omp::Directive::OMPD_unroll:
		 genUnrollOp(converter, symTable, stmtCtx, semaCtx, eval, loc, queue, item);
		 break;
 - // case llvm::omp::Directive::OMPD_workdistribute:
 + case llvm::omp::Directive::OMPD_workdistribute:
 +   newOp = genWorkdistributeOp(converter, symTable, semaCtx, eval, loc, queue,
 +                               item);
 +   break;
	 case llvm::omp::Directive::OMPD_workshare:
```
- Flang Lowering Tests: `flang/test/Lower/OpenMP/workdistribute.f90` — ensure lowering emits `omp.workdistribute` under `omp.teams`, and in composite form under `omp.target`.
```diff
@@ -0,0 +30 @@
 + ! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s
 +
 + ! CHECK-LABEL: func @_QPtarget_teams_workdistribute
 + subroutine target_teams_workdistribute()
 +   integer :: aa(10), bb(10)
 +   ! CHECK: omp.target
 +   ! CHECK: omp.teams
 +   ! CHECK: omp.workdistribute
 +   !$omp target teams workdistribute
 +   aa = bb
 +   ! CHECK: omp.terminator
 +   ! CHECK: omp.terminator
 +   ! CHECK: omp.terminator
 +   !$omp end target teams workdistribute
 + end subroutine target_teams_workdistribute
 +
 + ! CHECK-LABEL: func @_QPteams_workdistribute
 + subroutine teams_workdistribute()
 +   use iso_fortran_env
 +   real(kind=real32) :: a
 +   real(kind=real32), dimension(10) :: x
 +   real(kind=real32), dimension(10) :: y
 +   ! CHECK: omp.teams
 +   ! CHECK: omp.workdistribute
 +   !$omp teams workdistribute
 +   y = a * x + y
 +   ! CHECK: omp.terminator
 +   ! CHECK: omp.terminator
 +   !$omp end teams workdistribute
 + end subroutine teams_workdistribute
```

Pitfalls & Reviewer Notes
- Ensure clause processing for `num_teams`/`thread_limit` is triggered for composite forms.
- Lowering uses `genOpWithBody` shared infra; confirm region body generation and terminators.

---
