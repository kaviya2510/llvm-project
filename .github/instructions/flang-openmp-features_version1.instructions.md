---
applyTo: flang/lib/Parser/openmp*,flang/lib/Semantics/*omp*,flang/lib/Lower/OpenMP/**/*,flang/lib/Optimizer/OpenMP/**/*,flang/include/flang/Parser/parse-tree.h,flang/include/flang/Lower/OpenMP*,flang/include/flang/Support/OpenMP*,flang/include/flang/Optimizer/OpenMP/**/*,mlir/include/mlir/Dialect/OpenMP/**/*,mlir/lib/Dialect/OpenMP/**/*,mlir/lib/Target/LLVMIR/Dialect/OpenMP/**/*,llvm/include/llvm/Frontend/OpenMP/**/*,llvm/lib/Frontend/OpenMP/**/*
---

Status
- PRs included: 4
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
