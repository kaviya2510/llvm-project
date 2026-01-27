---
applyTo: flang/lib/Parser/openmp*,flang/lib/Semantics/*omp*,flang/lib/Lower/OpenMP/**/*,flang/lib/Optimizer/OpenMP/**/*,flang/include/flang/Parser/parse-tree.h,flang/include/flang/Lower/OpenMP*,flang/include/flang/Support/OpenMP*,flang/include/flang/Optimizer/OpenMP/**/*,mlir/include/mlir/Dialect/OpenMP/**/*,mlir/lib/Dialect/OpenMP/**/*,mlir/lib/Target/LLVMIR/Dialect/OpenMP/**/*,llvm/include/llvm/Frontend/OpenMP/**/*,llvm/lib/Frontend/OpenMP/**/*
---

# Flang OpenMP Feature Implementation Guide

## Scope and Purpose

**What this file covers:**
This knowledge base tracks OpenMP clause and directive implementations across the **entire Flang/MLIR compiler stack**, from parsing Fortran source to generating LLVM IR. It documents feature additions, implementation patterns, and the interaction between compiler stages.

**When to use this guide:**
- Implementing new OpenMP clauses or directives
- Understanding how OpenMP features flow through the compilation pipeline
- Fixing OpenMP-related bugs (parser, semantics, lowering, or codegen)
- Learning patterns for clause validation, MLIR operation design, or runtime calls
- Reviewing PRs that add OpenMP functionality

**GitHub Copilot Integration:**
This file is automatically read by GitHub Copilot when you work on OpenMP-related files (via `applyTo` pattern matching). Ask questions like:
- "How do I add a new OpenMP clause?"
- "What validation is needed for reduction clauses?"
- "How does the PARALLEL directive get lowered to MLIR?"
- "What LLVM runtime functions are used for tasks?"

**Compiler Stack Coverage:**
1. **Parser** (`flang/lib/Parser/`) - Recognize OpenMP syntax
2. **Semantics** (`flang/lib/Semantics/`) - Validate usage and resolve symbols
3. **MLIR Dialect** (`mlir/include/mlir/Dialect/OpenMP/`) - Define operations
4. **Lowering** (`flang/lib/Lower/OpenMP/`) - Convert parse tree to MLIR
5. **LLVM Translation** (`mlir/lib/Target/LLVMIR/Dialect/OpenMP/`) - Generate LLVM IR

---

# PART 0: NEW CONTRIBUTOR ONBOARDING

## Welcome to Flang OpenMP Development!

This section helps new contributors get started with OpenMP implementation in Flang. Whether you're fixing bugs or adding features, these resources will guide you through the compiler architecture.

---

## Prerequisites & Required Knowledge

### Essential Background

**Must Have:**
- Basic Fortran programming knowledge
- Familiarity with OpenMP directives (PARALLEL, DO, TASK, etc.)
- Understanding of compiler phases (parsing, semantic analysis, code generation)
- Git and command-line proficiency

**Helpful to Have:**
- C++ experience (Flang is written in C++)
- LLVM/MLIR basics (intermediate representations)
- CMake build system knowledge
- Experience reading compiler source code

### Learning Resources

**OpenMP Specification:**
- [OpenMP 5.2 Specification](https://www.openmp.org/specifications/) - Official language standard
- [OpenMP Examples Document](https://www.openmp.org/wp-content/uploads/openmp-examples-5.2.pdf) - Practical usage examples

**LLVM/Flang Documentation:**
- [Flang Documentation](https://flang.llvm.org/docs/) - Official Flang docs
- [LLVM Documentation](https://llvm.org/docs/) - Core LLVM concepts
- [MLIR Documentation](https://mlir.llvm.org/) - Multi-Level IR framework
- [OpenMP Runtime Library](https://openmp.llvm.org/Reference.pdf) - Runtime function reference

**Tutorials:**
- [LLVM Programmer's Manual](https://llvm.org/docs/ProgrammersManual.html) - C++ idioms used in LLVM
- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/) - Understanding MLIR syntax
- [Flang Design Docs](https://github.com/llvm/llvm-project/tree/main/flang/docs) - Architecture decisions

---

## Essential Terminology Glossary

### Compiler Concepts

**Parse Tree:**
- In-memory representation of source code structure built during parsing
- Each OpenMP directive becomes a parse tree node (e.g., `parser::OmpClause::Private`)
- Defined in `flang/include/flang/Parser/parse-tree.h`

**Semantic Analysis:**
- Validation phase that checks OpenMP directive/clause usage correctness
- Verifies restrictions (e.g., "PRIVATE variables cannot be POINTER")
- Resolves symbols (variable names) and checks scoping rules

**MLIR Operation (Op):**
- Intermediate representation of OpenMP constructs in MLIR dialect
- Examples: `omp.parallel`, `omp.wsloop`, `omp.task`
- Defined in `mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td`

**Lowering:**
- Process of converting parse tree → MLIR operations
- Transforms Fortran OpenMP directives into MLIR OpenMP ops
- Implemented in `flang/lib/Lower/OpenMP/OpenMP.cpp`

**Translation (MLIR → LLVM IR):**
- Final stage converting MLIR ops → LLVM IR
- Generates calls to OpenMP runtime library (`libgomp` or `libomp`)
- Implemented in `mlir/lib/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.cpp`

**Outlined Function:**
- Parallel/task region body extracted into a separate function
- Runtime spawns threads/tasks that execute the outlined function
- Example: `!$omp parallel` body becomes `function_name.omp_outlined()`

**Canonical Loop:**
- Normalized loop representation (start, end, step, induction variable)
- Used for loop-based constructs (DO, SIMD, DISTRIBUTE, etc.)
- Class: `llvm::CanonicalLoopInfo` in OMPIRBuilder

### Common Abbreviations

**DSP:** DataSharingProcessor - Handles private/firstprivate/shared variable processing  
**CLI:** CanonicalLoopInfo - Normalized loop representation  
**IV:** Induction Variable - Loop counter variable  
**OMPIRBuilder:** OpenMPIRBuilder - Helper for generating OpenMP LLVM IR  
**RTL:** Runtime Library - OpenMP runtime functions (`__kmpc_*`)  
**OpOperands:** Operation operands - MLIR operation inputs/attributes  

### OpenMP-Specific Terms

**Data-Sharing Attributes:**
- **SHARED:** Variable shared across all threads (single storage)
- **PRIVATE:** Each thread gets its own uninitialized copy
- **FIRSTPRIVATE:** Each thread gets its own copy initialized from original
- **LASTPRIVATE:** Last iteration's value copied back to original

**Worksharing Constructs:**
- Directives that divide work among threads in a team
- Examples: DO (loop), SECTIONS, SINGLE, WORKSHARE

**Tasking Constructs:**
- Directives that create explicit tasks
- Examples: TASK, TASKLOOP, TASKGROUP

**Device Constructs:**
- Directives for offloading to accelerators
- Examples: TARGET, TEAMS, DISTRIBUTE

---

## Getting Started: Build & Setup

### Building Flang with OpenMP Support

```bash
# Clone LLVM project
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

# Create build directory
mkdir build && cd build

# Configure with CMake (enable Flang)
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="clang;flang;mlir" \
  -DLLVM_ENABLE_RUNTIMES="openmp" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
  -DCMAKE_INSTALL_PREFIX=/path/to/install

# Build Flang
ninja flang-new

# Run OpenMP tests
ninja check-flang-openmp
```

### Testing Your Changes

```bash
# Run specific test file
llvm-lit flang/test/Lower/OpenMP/parallel.f90

# Run all OpenMP semantic tests
ninja check-flang-semantics

# Run all OpenMP lowering tests
ninja check-flang-lower

# Run MLIR → LLVM IR translation tests
ninja check-mlir-openmp
```

### Debugging Tips

**Dumping Parse Tree:**
```bash
flang-new -fc1 -fdebug-dump-parse-tree test.f90
```

**Dumping MLIR:**
```bash
flang-new -fc1 -emit-mlir test.f90 -o test.mlir
```

**Dumping LLVM IR:**
```bash
flang-new -S -emit-llvm test.f90 -o test.ll
```

**Enable Debug Logging:**
```bash
flang-new -mllvm -debug-only=omp-lowering test.f90
```

---

## First Contribution Roadmap

### Level 1: Understanding (1-2 weeks)

**Goal:** Learn how OpenMP flows through the compiler

1. **Read existing code:**
   - Study simple directive implementation (e.g., `omp.barrier`)
   - Trace `PARALLEL` directive from parser → LLVM IR
   - Read test files to understand expected behavior

2. **Run tests:**
   - Execute existing OpenMP tests: `ninja check-flang-openmp`
   - Analyze test failures to understand validation logic
   - Compare generated MLIR/LLVM IR with expected output

3. **Recommended reading:**
   - OpenMP 5.2 specification sections on basic constructs
   - MLIR OpenMP dialect documentation
   - Flang design documents

### Level 2: Simple Fixes (2-4 weeks)

**Goal:** Make small, focused contributions

**Good First Issues:**

1. **Add missing test cases:**
   - Find untested clause combinations
   - Write Fortran test with expected errors
   - Submit test-only PR

2. **Fix diagnostic messages:**
   - Improve error message clarity
   - Add source location information
   - File: `flang/lib/Semantics/check-omp-structure.cpp`

3. **Document existing features:**
   - Add code comments explaining complex logic
   - Update this instruction file with new patterns
   - Clarify ambiguous implementation decisions

**Example Tasks:**
- Add test for `PRIVATE(x, y, z)` with derived type variables
- Improve error message for incompatible clause combinations
- Document why certain clauses are mutually exclusive

### Level 3: Feature Implementation (1-3 months)

**Goal:** Add missing OpenMP functionality

**Suggested Projects:**

1. **Add new clause to existing directive:**
   - Start with parse tree modification
   - Add semantic validation
   - Implement lowering logic
   - Example: Add `HINT` clause to `CRITICAL` directive

2. **Implement missing modifier:**
   - Parse modifier syntax
   - Validate usage restrictions
   - Pass modifier info to MLIR/LLVM
   - Example: `ITERATOR` modifier on `DEPEND` clause

3. **Complete partial implementation:**
   - Find "TODO" or "not yet implemented" in code
   - Complete the missing functionality
   - Add comprehensive tests
   - Example: Finish `TASKLOOP` clause support (see PR #166903)

### Level 4: Complex Features (3-6 months)

**Goal:** Implement sophisticated OpenMP constructs

**Advanced Projects:**

1. **New directive implementation:**
   - Full 5-stage implementation (parser → translation)
   - Handle all applicable clauses
   - Example: `MASKED` construct (OpenMP 5.1)

2. **Interoperability features:**
   - Implement `interop` construct
   - Handle device pointers and memory management
   - Coordinate with runtime team

3. **Performance optimizations:**
   - Optimize reduction lowering
   - Improve loop scheduling strategies
   - Profile and optimize code generation

---

## COMPLETE IMPLEMENTATION EXAMPLE: Adding UNTIED Clause to TASK Directive

This section demonstrates a **complete end-to-end implementation** of adding the `UNTIED` clause to the `TASK` directive, covering all 5 compilation stages. Use this as a reference template for implementing simple boolean clauses.

### Background: What is UNTIED?

**OpenMP Specification**: OpenMP 5.2 Section 2.12.1  
**Directive**: TASK  
**Clause Type**: Boolean modifier (no arguments)  
**Semantics**: 
- By default, tasks are "tied" to the thread that starts executing them
- `UNTIED` allows a suspended task to resume on a different thread
- Enables better load balancing for long-running tasks

**Fortran Syntax**:
```fortran
!$omp task untied
  ! Task body can migrate between threads
!$omp end task
```

### Stage 1: Parser Implementation

**Files to modify:**
1. `llvm/include/llvm/Frontend/OpenMP/OMPKinds.def`
2. `flang/include/flang/Parser/parse-tree.h`
3. `flang/lib/Parser/openmp-parsers.cpp`

**Step 1.1: Register clause kind in OMPKinds.def**

Location: `llvm/include/llvm/Frontend/OpenMP/OMPKinds.def` (around line 500)

```cpp
// Add to the clause enum list
__OMP_CLAUSE(untied, OMPC_untied)
```

**Step 1.2: Define parse tree node in parse-tree.h**

Location: `flang/include/flang/Parser/parse-tree.h` (around line 3800)

```cpp
// Add to OmpClause namespace
struct OmpClause {
  // ... existing clauses ...
  
  // UNTIED clause - no arguments, simple keyword
  EMPTY_CLASS(Untied);
  
  // ... more clauses ...
};
```

**Explanation**: `EMPTY_CLASS` is used because UNTIED has no parameters (it's just a keyword).

**Step 1.3: Implement grammar in openmp-parsers.cpp**

Location: `flang/lib/Parser/openmp-parsers.cpp` (around line 600)

```cpp
// Add to clause parser list
TYPE_PARSER(construct<OmpClause>(Parser<OmpClause::Allocate>{}) ||
    construct<OmpClause>(Parser<OmpClause::If>{}) ||
    // ... existing clauses ...
    construct<OmpClause>(Parser<OmpClause::Untied>{}) ||  // ADD THIS LINE
    // ... more clauses ...
)

// Define the UNTIED parser (add around line 400)
TYPE_PARSER(construct<OmpClause::Untied>("UNTIED"_tok))
```

**Step 1.4: Add to TASK directive's allowed clauses**

Location: `flang/lib/Parser/openmp-parsers.cpp` (around line 250)

Find the `OmpClauseList` definition for TASK directive and ensure UNTIED is allowed:

```cpp
// The parser automatically allows all clauses; semantic checks validate compatibility
```

**Step 1.5: Create parser test**

File: `flang/test/Parser/OpenMP/task-untied.f90`

```fortran
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck %s

! Test UNTIED clause parsing

program test_untied
  integer :: x
  
  ! CHECK: OmpBeginBlockDirective
  ! CHECK: OmpBlockDirective -> llvm::omp::Directive = task
  ! CHECK: OmpClauseList -> OmpClause -> Untied
  !$omp task untied
    x = 42
  !$omp end task
  
  ! Test with other clauses
  ! CHECK: OmpClauseList -> OmpClause -> Untied
  ! CHECK: OmpClauseList -> OmpClause -> Private
  !$omp task untied private(x)
    x = x + 1
  !$omp end task

end program
```

**Test parser**:
```bash
llvm-lit flang/test/Parser/OpenMP/task-untied.f90
```

---

### Stage 2: Semantic Validation

**Files to modify:**
1. `flang/lib/Semantics/check-omp-structure.cpp`

**Step 2.1: Add clause to TASK directive's allowed list**

Location: `flang/lib/Semantics/check-omp-structure.cpp` (around line 1800)

```cpp
void OmpStructureChecker::Enter(const parser::OmpClauseList &list) {
  // ...
}

// Find the TASK directive validation function
void OmpStructureChecker::Enter(const parser::OpenMPBlockConstruct &x) {
  const auto &beginBlockDir{std::get<parser::OmpBeginBlockDirective>(x.t)};
  const auto &beginDir{std::get<parser::OmpBlockDirective>(beginBlockDir.t)};
  
  switch (beginDir.v) {
    case llvm::omp::Directive::OMPD_task:
      PushContext(beginDir.source, beginDir.v);
      SetClauseSets(llvm::omp::Directive::OMPD_task);
      break;
    // ... other cases ...
  }
}
```

Find `SetClauseSets` definition and add UNTIED to TASK's allowed clauses:

```cpp
void OmpStructureChecker::SetClauseSets(llvm::omp::Directive dir) {
  switch (dir) {
    case llvm::omp::Directive::OMPD_task:
      allowedClauses_ = {
        llvm::omp::Clause::OMPC_default,
        llvm::omp::Clause::OMPC_private,
        llvm::omp::Clause::OMPC_firstprivate,
        llvm::omp::Clause::OMPC_shared,
        llvm::omp::Clause::OMPC_depend,
        llvm::omp::Clause::OMPC_if,
        llvm::omp::Clause::OMPC_final,
        llvm::omp::Clause::OMPC_priority,
        llvm::omp::Clause::OMPC_untied,     // ADD THIS LINE
        llvm::omp::Clause::OMPC_mergeable,
        llvm::omp::Clause::OMPC_allocate,
        llvm::omp::Clause::OMPC_in_reduction,
      };
      break;
    // ... other directives ...
  }
}
```

**Step 2.2: Add restriction checks (if needed)**

For UNTIED, check it's mutually exclusive with MERGED (if MERGED is implemented):

```cpp
void OmpStructureChecker::Leave(const parser::OmpClauseList &) {
  // Check for conflicting clauses
  if (FindClause(llvm::omp::Clause::OMPC_untied) &&
      FindClause(llvm::omp::Clause::OMPC_merged)) {
    context_.Say(GetContext().clauseSource,
        "UNTIED and MERGED clauses are mutually exclusive"_err_en_US);
  }
}
```

**Step 2.3: Create semantic validation test**

File: `flang/test/Semantics/OpenMP/task-untied.f90`

```fortran
! RUN: %python %S/test_errors.py %s %flang_fc1 -fopenmp

! Test semantic validation for UNTIED clause

program test_untied_semantics
  integer :: x

  ! Valid: UNTIED alone
  !$omp task untied
    x = 42
  !$omp end task

  ! Valid: UNTIED with compatible clauses
  !$omp task untied private(x) if(.true.)
    x = x + 1
  !$omp end task

  ! Valid: UNTIED with FIRSTPRIVATE
  !$omp task untied firstprivate(x)
    x = x * 2
  !$omp end task

  ! Future error case (if MERGED is implemented):
  ! !ERROR: UNTIED and MERGED clauses are mutually exclusive
  ! !$omp task untied merged
  !   x = 1
  ! !$omp end task

end program
```

**Test semantics**:
```bash
llvm-lit flang/test/Semantics/OpenMP/task-untied.f90
```

---

### Stage 3: MLIR Dialect Definition

**Files to modify:**
1. `mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td`

**Step 3.1: Add UNTIED attribute to omp.task operation**

Location: `mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td` (around line 800)

```tablegen
def TaskOp : OpenMP_Op<"task", [AttrSizedOperandSegments]> {
  let summary = "task directive";
  let description = [{
    The task construct defines an explicit task.
    
    OpenMP 5.2 Section 2.12.1
  }];

  let arguments = (ins
    Optional<I1>:$if_expr,
    Optional<I1>:$final_expr,
    Optional<I32>:$priority,
    Variadic<AnyType>:$depend_vars,
    Optional<UnitAttr>:$untied,        // ADD THIS LINE
    Optional<UnitAttr>:$mergeable,
    Variadic<AnyType>:$private_vars,
    Variadic<AnyType>:$firstprivate_vars,
    OptionalAttr<DependKindArrayAttr>:$depend_kinds
  );

  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    oilist(
      `if` `(` $if_expr `)`
      | `final` `(` $final_expr `)`
      | `priority` `(` $priority `)`
      | `untied` $untied                  // ADD THIS LINE
      | `mergeable` $mergeable
      | `depend` `(` custom<DependVars>($depend_kinds, $depend_vars) `)`
      | `private` `(` $private_vars `:` type($private_vars) `)`
      | `firstprivate` `(` $firstprivate_vars `:` type($firstprivate_vars) `)`
    )
    $region attr-dict
  }];

  let hasVerifier = 1;
}
```

**Explanation**: `UnitAttr` is used for boolean flags (present/absent).

**Step 3.2: Add verifier (if needed)**

Location: `mlir/lib/Dialect/OpenMP/OpenMPDialect.cpp` (around line 600)

```cpp
LogicalResult TaskOp::verify() {
  // Verify UNTIED and MERGED are mutually exclusive (if MERGED exists)
  if (getUntied() && getMergeable()) {
    // Both can coexist; no restriction between untied and mergeable
  }
  
  // Additional verifications...
  return success();
}
```

**Step 3.3: Create MLIR operation test**

File: `mlir/test/Dialect/OpenMP/task-untied.mlir`

```mlir
// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @task_untied
func.func @task_untied() {
  // CHECK: omp.task untied
  omp.task untied {
    %c1 = arith.constant 1 : i32
    omp.terminator
  }
  return
}

// CHECK-LABEL: func @task_untied_with_private
func.func @task_untied_with_private(%arg0: i32) {
  // CHECK: omp.task untied private(%{{.*}} : i32)
  omp.task untied private(%arg0 : i32) {
    %c1 = arith.constant 1 : i32
    omp.terminator
  }
  return
}
```

**Test MLIR**:
```bash
llvm-lit mlir/test/Dialect/OpenMP/task-untied.mlir
```

---

### Stage 4: Lowering from Parse Tree to MLIR

**Files to modify:**
1. `flang/lib/Lower/OpenMP/Clauses.cpp`
2. `flang/lib/Lower/OpenMP/OpenMP.cpp`

**Step 4.1: Add UNTIED to clause processing**

Location: `flang/lib/Lower/OpenMP/Clauses.cpp` (around line 1200)

```cpp
// Find the clause processing function
void ClauseProcessor::processUntied(
    mlir::UnitAttr &result) const {
  const parser::CharBlock *source = nullptr;
  auto &context = converter.getFirOpBuilder().getContext();
  
  if (findClause<parser::OmpClause::Untied>(source)) {
    result = mlir::UnitAttr::get(context);
  }
}
```

Add declaration in header `flang/lib/Lower/OpenMP/Clauses.h`:

```cpp
class ClauseProcessor {
public:
  // ... existing methods ...
  
  void processUntied(mlir::UnitAttr &result) const;
  
  // ... more methods ...
};
```

**Step 4.2: Use in TASK directive lowering**

Location: `flang/lib/Lower/OpenMP/OpenMP.cpp` (around line 2400)

```cpp
static mlir::omp::TaskOp genTaskOp(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
    mlir::Location loc, const ConstructQueue &queue,
    const parser::OmpClauseList &clauseList) {
  
  ClauseProcessor cp(converter, semaCtx, clauseList);
  
  mlir::Value ifClauseOperand, finalClauseOperand, priorityClauseOperand;
  mlir::UnitAttr untiedAttr, mergeableAttr;  // ADD untiedAttr
  llvm::SmallVector<mlir::Value> dependOperands;
  llvm::SmallVector<mlir::Attribute> dependKinds;
  
  // Process clauses
  cp.processIf(llvm::omp::Directive::OMPD_task, ifClauseOperand);
  cp.processFinal(finalClauseOperand);
  cp.processPriority(priorityClauseOperand);
  cp.processUntied(untiedAttr);           // ADD THIS LINE
  cp.processMergeable(mergeableAttr);
  cp.processDepend(dependOperands, dependKinds);
  
  // Process data-sharing clauses
  llvm::SmallVector<mlir::Value> privateVars, firstprivateVars;
  cp.processPrivate(privateVars);
  cp.processFirstprivate(firstprivateVars);
  
  // Create task operation
  auto taskOp = converter.getFirOpBuilder().create<mlir::omp::TaskOp>(
      loc,
      ifClauseOperand,
      finalClauseOperand,
      priorityClauseOperand,
      dependOperands,
      untiedAttr,              // ADD THIS LINE
      mergeableAttr,
      privateVars,
      firstprivateVars,
      dependKinds.empty() 
          ? nullptr 
          : mlir::ArrayAttr::get(converter.getFirOpBuilder().getContext(), 
                                 dependKinds)
  );
  
  // Lower task body
  auto &block = taskOp.getRegion().emplaceBlock();
  auto &builder = converter.getFirOpBuilder();
  auto insertPt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(&block);
  
  // Generate task body code
  genNestedEvaluations(converter, eval);
  
  builder.create<mlir::omp::TerminatorOp>(loc);
  builder.restoreInsertionPoint(insertPt);
  
  return taskOp;
}
```

**Step 4.3: Create lowering test**

File: `flang/test/Lower/OpenMP/task-untied.f90`

```fortran
! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s

! Test lowering of UNTIED clause

program test_untied_lowering
  integer :: x

  ! CHECK-LABEL: func @_QQmain
  
  ! CHECK: omp.task untied {
  ! CHECK:   %[[C42:.*]] = arith.constant 42
  ! CHECK:   omp.terminator
  ! CHECK: }
  !$omp task untied
    x = 42
  !$omp end task

  ! CHECK: omp.task untied {
  ! CHECK:   omp.terminator
  ! CHECK: }
  !$omp task untied private(x)
    x = x + 1
  !$omp end task

  ! CHECK: omp.task if(%{{.*}}) untied {
  ! CHECK:   omp.terminator
  ! CHECK: }
  !$omp task if(.true.) untied
    x = x * 2
  !$omp end task

end program
```

**Test lowering**:
```bash
llvm-lit flang/test/Lower/OpenMP/task-untied.f90
```

---

### Stage 5: MLIR to LLVM IR Translation

**Files to modify:**
1. `mlir/lib/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.cpp`

**Step 5.1: Add UNTIED flag to task translation**

Location: `mlir/lib/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.cpp` (around line 1600)

```cpp
static LogicalResult
convertOmpTaskOp(omp::TaskOp taskOp, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  
  // Extract clause values
  llvm::Value *ifCondition = nullptr;
  if (taskOp.getIfExpr()) {
    ifCondition = moduleTranslation.lookupValue(taskOp.getIfExpr());
  }
  
  llvm::Value *finalCondition = nullptr;
  if (taskOp.getFinalExpr()) {
    finalCondition = moduleTranslation.lookupValue(taskOp.getFinalExpr());
  }
  
  llvm::Value *priority = nullptr;
  if (taskOp.getPriority()) {
    priority = moduleTranslation.lookupValue(taskOp.getPriority());
  }
  
  // Process UNTIED flag (ADD THIS SECTION)
  bool isUntied = taskOp.getUntied().has_value();
  
  // Process dependencies
  SmallVector<llvm::OpenMPIRBuilder::DependData> dependencies;
  processDependClauses(taskOp, moduleTranslation, dependencies);
  
  // Create task body outline function
  auto bodyCB = [&](InsertPointTy allocaIP, InsertPointTy codegenIP) {
    // ... body generation code ...
    return llvm::Error::success();
  };
  
  // Create task using OMPIRBuilder
  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);
  
  // Call OMPIRBuilder to generate task
  auto taskInfo = ompBuilder->createTask(
      llvm::omp::Directive::OMPD_task,
      allocaIP,
      bodyCB,
      !isUntied,          // ADD THIS: tied = !untied
      finalCondition,
      ifCondition,
      dependencies
  );
  
  // Set priority if present
  if (priority) {
    // Runtime call: __kmpc_taskloop_set_priority(priority)
    ompBuilder->createTaskPriority(taskInfo, priority);
  }
  
  return success();
}
```

**Key insight**: The OpenMP runtime represents "untied" as `tied = false`, so we pass `!isUntied`.

**Step 5.2: Verify runtime function signature**

The `OMPIRBuilder::createTask` function signature (from `llvm/include/llvm/Frontend/OpenMP/OMPIRBuilder.h`):

```cpp
InsertPointOrErrorTy createTask(
    const LocationDescription &Loc,
    InsertPointTy AllocaIP,
    BodyGenCallbackTy BodyGenCB,
    bool Tied = true,              // This is where untied goes
    Value *Final = nullptr,
    Value *IfCondition = nullptr,
    SmallVector<DependData> Dependencies = {});
```

**Step 5.3: Create translation test**

File: `mlir/test/Target/LLVMIR/openmp-task-untied.mlir`

```mlir
// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @task_untied() {
  // CHECK: call {{.*}} @__kmpc_omp_task_alloc
  // CHECK-SAME: i32 0  
  // Note: flags parameter bit 0 = tied/untied
  // tied=0 means untied task
  omp.task untied {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    omp.terminator
  }
  llvm.return
}

llvm.func @task_tied() {
  // CHECK: call {{.*}} @__kmpc_omp_task_alloc
  // CHECK-SAME: i32 1
  // tied=1 means tied task (default)
  omp.task {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    omp.terminator
  }
  llvm.return
}
```

**Test translation**:
```bash
llvm-lit mlir/test/Target/LLVMIR/openmp-task-untied.mlir
```

---

### Summary: Files Modified

| **Stage** | **File** | **Change** |
|-----------|----------|------------|
| 1. Parser | `llvm/include/llvm/Frontend/OpenMP/OMPKinds.def` | Add `__OMP_CLAUSE(untied, OMPC_untied)` |
| 1. Parser | `flang/include/flang/Parser/parse-tree.h` | Add `EMPTY_CLASS(Untied)` |
| 1. Parser | `flang/lib/Parser/openmp-parsers.cpp` | Add parser for UNTIED |
| 2. Semantics | `flang/lib/Semantics/check-omp-structure.cpp` | Add UNTIED to TASK allowed clauses |
| 3. MLIR | `mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td` | Add `Optional<UnitAttr>:$untied` to TaskOp |
| 4. Lowering | `flang/lib/Lower/OpenMP/Clauses.cpp` | Add `processUntied()` method |
| 4. Lowering | `flang/lib/Lower/OpenMP/Clauses.h` | Declare `processUntied()` |
| 4. Lowering | `flang/lib/Lower/OpenMP/OpenMP.cpp` | Use untied in `genTaskOp()` |
| 5. Translation | `mlir/lib/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.cpp` | Pass `!isUntied` to OMPIRBuilder |

### Testing Checklist

- [ ] Parser test passes: `llvm-lit flang/test/Parser/OpenMP/task-untied.f90`
- [ ] Semantic test passes: `llvm-lit flang/test/Semantics/OpenMP/task-untied.f90`
- [ ] MLIR dialect test passes: `llvm-lit mlir/test/Dialect/OpenMP/task-untied.mlir`
- [ ] Lowering test passes: `llvm-lit flang/test/Lower/OpenMP/task-untied.f90`
- [ ] Translation test passes: `llvm-lit mlir/test/Target/LLVMIR/openmp-task-untied.mlir`
- [ ] End-to-end test: Compile and run Fortran program with UNTIED task

### Debugging Commands

```bash
# 1. Check parse tree
flang-new -fc1 -fdebug-dump-parse-tree -fopenmp test.f90

# 2. Check MLIR output
flang-new -fc1 -emit-mlir -fopenmp test.f90 -o test.mlir

# 3. Check LLVM IR
flang-new -S -emit-llvm -fopenmp test.f90 -o test.ll

# 4. Run with debug logging
flang-new -mllvm -debug-only=omp-lowering -fopenmp test.f90

# 5. Inspect runtime calls
flang-new -S -emit-llvm -fopenmp test.f90 -o - | grep kmpc
```

### Common Issues & Solutions

**Issue 1**: Parser doesn't recognize UNTIED  
**Solution**: Ensure `OMPKinds.def` was modified and LLVM was rebuilt

**Issue 2**: Semantic checker rejects UNTIED on TASK  
**Solution**: Check `SetClauseSets()` includes `OMPC_untied` for `OMPD_task`

**Issue 3**: MLIR operation fails to build  
**Solution**: Verify `UnitAttr` is correctly passed (can be `nullptr` if absent)

**Issue 4**: LLVM IR doesn't show untied flag  
**Solution**: Check translation code passes `!isUntied` (note the negation)

**Issue 5**: Runtime assertion failure  
**Solution**: Verify OpenMP runtime version supports untied tasks

### Related Implementation Patterns

This pattern applies to other boolean clauses:
- `NOWAIT` (on various directives)
- `MERGEABLE` (on TASK)
- `NOGROUP` (on TASKLOOP)
- `SIMD` (modifier on various directives)

For clauses with arguments, refer to:
- `IF` clause (expression argument)
- `PRIVATE` clause (variable list)
- `REDUCTION` clause (operator + variable list)

---

## COMPLETE IMPLEMENTATION EXAMPLE: Adding REDUCTION Clause to PARALLEL Directive

This section demonstrates a **complete end-to-end implementation** of the `REDUCTION` clause, covering all 5 compilation stages. Unlike the simpler UNTIED clause, REDUCTION involves complex data flow, operator handling, and runtime coordination. Use this as a reference template for implementing clauses with operators and variable lists.

### Background: What is REDUCTION?

**OpenMP Specification**: OpenMP 5.2 Section 2.21.5  
**Directive**: Applicable to PARALLEL, DO, SIMD, SECTIONS, and combined constructs  
**Clause Type**: Complex clause with operator and variable list  
**Semantics**: 
- Creates private copies of reduction variables for each thread
- Each thread performs local reduction operations
- At the end of the construct, all private copies are combined into the original variable
- Supports intrinsic operators (+, *, -, max, min, .and., .or., etc.) and user-defined reduction operators

**Fortran Syntax**:
```fortran
!$omp parallel reduction(+:sum) reduction(max:maxval)
  sum = sum + local_value
  maxval = max(maxval, local_max)
!$omp end parallel
```

### Stage 1: Parser Implementation

**Files to modify:**
1. `llvm/include/llvm/Frontend/OpenMP/OMPKinds.def`
2. `flang/include/flang/Parser/parse-tree.h`
3. `flang/lib/Parser/openmp-parsers.cpp`

**Step 1.1: Register clause kind in OMPKinds.def**

Location: `llvm/include/llvm/Frontend/OpenMP/OMPKinds.def` (around line 500)

```cpp
// Add to the clause enum list
__OMP_CLAUSE(reduction, OMPC_reduction)
```

**Step 1.2: Define parse tree structures in parse-tree.h**

Location: `flang/include/flang/Parser/parse-tree.h` (around line 3500)

```cpp
// Define reduction identifier (operator or procedure)
struct OmpReductionOperator {
  UNION_CLASS_BOILERPLATE(OmpReductionOperator);
  ENUM_CLASS(Operator, Add, Subtract, Multiply, And, Or, Eqv, Neqv,
      Max, Min, Iand, Ior, Ieor)
  
  // Either an intrinsic operator or a user-defined procedure
  std::variant<Operator, Name> u;
};

// Define reduction modifier (optional in OpenMP 5.0+)
struct OmpReductionModifier {
  ENUM_CLASS(Kind, Inscan, Task, Default)
  WRAPPER_CLASS_BOILERPLATE(OmpReductionModifier, Kind);
};

// Define the complete REDUCTION clause
struct OmpClause::Reduction {
  TUPLE_CLASS_BOILERPLATE(OmpClause::Reduction);
  // (optional modifier, operator/procedure, variable list)
  std::tuple<std::optional<OmpReductionModifier>, 
             OmpReductionOperator, 
             OmpObjectList> t;
};
```

**Explanation**: 
- `OmpReductionOperator` is a variant that can hold either an intrinsic operator enum or a user-defined procedure name
- `OmpReductionModifier` handles modifiers like `inscan`, `task`, or `default`
- `OmpClause::Reduction` combines all three components

**Step 1.3: Implement grammar in openmp-parsers.cpp**

Location: `flang/lib/Parser/openmp-parsers.cpp` (around line 300)

```cpp
// Parse reduction operator
TYPE_PARSER(construct<OmpReductionOperator>(
    construct<OmpReductionOperator::Operator>(
        "+" >> pure(OmpReductionOperator::Operator::Add) ||
        "-" >> pure(OmpReductionOperator::Operator::Subtract) ||
        "*" >> pure(OmpReductionOperator::Operator::Multiply) ||
        ".AND." >> pure(OmpReductionOperator::Operator::And) ||
        ".OR." >> pure(OmpReductionOperator::Operator::Or) ||
        ".EQV." >> pure(OmpReductionOperator::Operator::Eqv) ||
        ".NEQV." >> pure(OmpReductionOperator::Operator::Neqv) ||
        "MAX" >> pure(OmpReductionOperator::Operator::Max) ||
        "MIN" >> pure(OmpReductionOperator::Operator::Min) ||
        "IAND" >> pure(OmpReductionOperator::Operator::Iand) ||
        "IOR" >> pure(OmpReductionOperator::Operator::Ior) ||
        "IEOR" >> pure(OmpReductionOperator::Operator::Ieor)) ||
    construct<OmpReductionOperator>(name)))

// Parse reduction modifier
TYPE_PARSER(construct<OmpReductionModifier>(
    construct<OmpReductionModifier::Kind>(
        "INSCAN" >> pure(OmpReductionModifier::Kind::Inscan) ||
        "TASK" >> pure(OmpReductionModifier::Kind::Task) ||
        "DEFAULT" >> pure(OmpReductionModifier::Kind::Default))))

// Parse complete REDUCTION clause
TYPE_PARSER(construct<OmpClause::Reduction>(
    "REDUCTION" >> parenthesized(
        // Optional modifier followed by colon
        maybe(Parser<OmpReductionModifier>{} / ":"_tok),
        // Reduction operator followed by colon
        Parser<OmpReductionOperator>{} / ":"_tok,
        // Variable list
        Parser<OmpObjectList>{})))

// Add to main clause parser list
TYPE_PARSER(construct<OmpClause>(Parser<OmpClause::Allocate>{}) ||
    construct<OmpClause>(Parser<OmpClause::If>{}) ||
    // ... existing clauses ...
    construct<OmpClause>(Parser<OmpClause::Reduction>{}) ||  // ADD THIS LINE
    // ... more clauses ...
)
```

**Step 1.4: Create parser test**

File: `flang/test/Parser/OpenMP/reduction-clause.f90`

```fortran
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck %s

! Test REDUCTION clause parsing with various operators

program test_reduction_parsing
  integer :: sum, product, imax, imin
  logical :: land, lor
  integer :: i, array(100)
  
  sum = 0
  product = 1
  
  ! CHECK: OmpBeginBlockDirective
  ! CHECK: OmpBlockDirective -> llvm::omp::Directive = parallel
  ! CHECK: OmpClauseList -> OmpClause -> Reduction
  ! CHECK: OmpReductionOperator -> Operator = Add
  !$omp parallel reduction(+:sum)
    do i = 1, 100
      sum = sum + i
    end do
  !$omp end parallel
  
  ! Test multiple reductions
  ! CHECK: OmpClause -> Reduction
  ! CHECK: OmpReductionOperator -> Operator = Multiply
  ! CHECK: OmpClause -> Reduction
  ! CHECK: OmpReductionOperator -> Operator = Max
  !$omp parallel reduction(*:product) reduction(max:imax)
    product = product * i
    imax = max(imax, array(i))
  !$omp end parallel

end program
```

**Test parser**:
```bash
llvm-lit flang/test/Parser/OpenMP/reduction-clause.f90
```

---

### Stage 2: Semantic Validation

**Files to modify:**
1. `flang/lib/Semantics/check-omp-structure.cpp`

**Step 2.1: Add clause to allowed lists and validation**

Location: `flang/lib/Semantics/check-omp-structure.cpp` (around line 2500)

```cpp
void OmpStructureChecker::Enter(const parser::OmpClause::Reduction &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_reduction);
  
  const auto &[modifier, operator_, objList] = x.t;
  
  // Validate modifier if present
  if (modifier) {
    if (modifier->v == parser::OmpReductionModifier::Kind::Inscan) {
      if (GetContext().directive != llvm::omp::Directive::OMPD_do &&
          GetContext().directive != llvm::omp::Directive::OMPD_simd) {
        context_.Say(GetContext().clauseSource,
            "INSCAN modifier only allowed on worksharing-loop or SIMD"_err_en_US);
      }
    }
  }
  
  // Validate each variable in the reduction list
  for (const auto &obj : objList.v) {
    if (const auto *name = parser::Unwrap<parser::Name>(obj)) {
      if (const auto *symbol = name->symbol) {
        // Check variable type matches operator
        ValidateReductionTypeOperatorMatch(*symbol, operator_, name->source);
        
        // Check for conflicts with other data-sharing clauses
        CheckDataSharingConflicts(*symbol, llvm::omp::Clause::OMPC_reduction);
      }
    }
  }
}

// Helper: validate type-operator compatibility
void OmpStructureChecker::ValidateReductionTypeOperatorMatch(
    const Symbol &symbol, 
    const parser::OmpReductionOperator &op,
    const parser::CharBlock &source) {
  
  const auto *type = symbol.GetType();
  if (!type) return;
  
  if (const auto *intrinsicOp = 
      std::get_if<parser::OmpReductionOperator::Operator>(&op.u)) {
    
    using Operator = parser::OmpReductionOperator::Operator;
    
    switch (*intrinsicOp) {
      case Operator::Add:
      case Operator::Multiply:
        // Numeric types only
        if (!type->IsNumeric()) {
          context_.Say(source,
              "Reduction operator requires numeric type"_err_en_US);
        }
        break;
        
      case Operator::And:
      case Operator::Or:
        // Logical types only
        if (type->category() != TypeCategory::Logical) {
          context_.Say(source,
              "Logical reduction operator requires LOGICAL type"_err_en_US);
        }
        break;
        
      case Operator::Max:
      case Operator::Min:
        // Numeric or character types
        if (!type->IsNumeric() && 
            type->category() != TypeCategory::Character) {
          context_.Say(source,
              "MAX/MIN requires numeric or character type"_err_en_US);
        }
        break;
    }
  }
}
```

**Step 2.2: Create semantic test**

File: `flang/test/Semantics/OpenMP/reduction-errors.f90`

```fortran
! RUN: %python %S/test_errors.py %s %flang_fc1 -fopenmp

program test_reduction_errors
  integer :: int_var
  logical :: log_var
  
  ! Valid
  !$omp parallel reduction(+:int_var)
    int_var = int_var + 1
  !$omp end parallel
  
  ! ERROR: Type mismatch
  !ERROR: Reduction operator requires numeric type
  !$omp parallel reduction(+:log_var)
    log_var = .true.
  !$omp end parallel
  
  ! ERROR: Conflict with PRIVATE
  !ERROR: Variable cannot appear in both REDUCTION and PRIVATE
  !$omp parallel reduction(+:int_var) private(int_var)
    int_var = 1
  !$omp end parallel

end program
```

---

### Stage 3: MLIR Dialect Definition

**Files to modify:**
1. `mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td`

**Step 3.1: Add reduction to parallel operation**

Location: `mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td` (around line 500)

```tablegen
def ParallelOp : OpenMP_Op<"parallel", [AttrSizedOperandSegments]> {
  let summary = "parallel construct";
  
  let arguments = (ins
    Optional<I1>:$if_expr,
    Optional<I32>:$num_threads,
    Variadic<AnyType>:$private_vars,
    Variadic<AnyType>:$reduction_vars,        // ADD THIS
    OptionalAttr<ArrayAttr>:$reduction_syms   // ADD THIS
  );

  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    oilist(
      `if` `(` $if_expr `)`
      | `num_threads` `(` $num_threads `)`
      | `private` `(` $private_vars `:` type($private_vars) `)`
      | `reduction` custom<ReductionClause>($reduction_syms, $reduction_vars)
    )
    $region attr-dict
  }];
}

// Reduction declaration operation
def ReductionDeclareOp : OpenMP_Op<"reduction.declare", [Symbol]> {
  let summary = "declares a reduction operation";
  
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$type
  );
  
  let regions = (region 
    AnyRegion:$combiner,
    AnyRegion:$initializer
  );
  
  let assemblyFormat = [{
    $sym_name `:` $type
    `combiner` $combiner
    (`initializer` $initializer^)?
    attr-dict
  }];
}
```

**Step 3.2: Create MLIR test**

File: `mlir/test/Dialect/OpenMP/reduction.mlir`

```mlir
// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: omp.reduction.declare @add_i32
omp.reduction.declare @add_i32 : i32
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %sum = arith.addi %arg0, %arg1 : i32
  omp.yield(%sum : i32)
}

// CHECK-LABEL: func @parallel_reduction
func.func @parallel_reduction() {
  %sum = memref.alloca() : memref<i32>
  
  // CHECK: omp.parallel reduction(@add_i32 %{{.*}} -> %{{.*}} : memref<i32>)
  omp.parallel reduction(@add_i32 %sum -> %arg0 : memref<i32>) {
    %one = arith.constant 1 : i32
    omp.terminator
  }
  return
}
```

---

### Stage 4: Lowering from Parse Tree to MLIR

**Files to modify:**
1. `flang/lib/Lower/OpenMP/Clauses.cpp`
2. `flang/lib/Lower/OpenMP/OpenMP.cpp`

**Step 4.1: Add reduction clause processing**

Location: `flang/lib/Lower/OpenMP/Clauses.cpp` (around line 1500)

```cpp
void ClauseProcessor::processReduction(
    mlir::Location currentLocation,
    llvm::SmallVectorImpl<mlir::Value> &reductionVars,
    llvm::SmallVectorImpl<mlir::Attribute> &reductionSyms) const {
  
  auto reductionClauseList = 
      findRepeatableClause<parser::OmpClause::Reduction>();
  
  for (const auto &reductionClause : reductionClauseList) {
    const auto &[modifier, operator_, objList] = reductionClause->t;
    
    // Process each variable
    for (const parser::OmpObject &ompObject : objList.v) {
      if (const auto *name = parser::Unwrap<parser::Name>(ompObject)) {
        if (auto *symbol = name->symbol) {
          // Create reduction declaration
          mlir::SymbolRefAttr reductionSym = 
              getOrCreateReductionDecl(operator_, symbol->GetType());
          
          // Get variable reference
          mlir::Value var = converter.getSymbolAddress(*symbol);
          
          reductionVars.push_back(var);
          reductionSyms.push_back(reductionSym);
        }
      }
    }
  }
}

// Create or retrieve reduction declaration
mlir::SymbolRefAttr ClauseProcessor::getOrCreateReductionDecl(
    const parser::OmpReductionOperator &op,
    const semantics::DeclTypeSpec *type) {
  
  auto &builder = converter.getFirOpBuilder();
  auto module = builder.getModule();
  
  // Generate unique name
  std::string reductionName = getReductionName(op, type);
  
  // Check if already declared
  if (module.lookupSymbol<mlir::omp::ReductionDeclareOp>(reductionName)) {
    return mlir::SymbolRefAttr::get(builder.getContext(), reductionName);
  }
  
  // Create new reduction declaration
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&module.getBodyRegion().front());
  
  mlir::Type mlirType = converter.genType(*type);
  auto reductionDecl = builder.create<mlir::omp::ReductionDeclareOp>(
      builder.getUnknownLoc(),
      builder.getStringAttr(reductionName),
      mlir::TypeAttr::get(mlirType));
  
  // Create combiner and initializer regions
  createReductionCombiner(reductionDecl, op, mlirType);
  createReductionInitializer(reductionDecl, op, mlirType);
  
  return mlir::SymbolRefAttr::get(builder.getContext(), reductionName);
}

// Create combiner region
void ClauseProcessor::createReductionCombiner(
    mlir::omp::ReductionDeclareOp &reductionDecl,
    const parser::OmpReductionOperator &op,
    mlir::Type type) {
  
  auto &builder = converter.getFirOpBuilder();
  mlir::Block &combinerBlock = reductionDecl.getCombiner().emplaceBlock();
  
  combinerBlock.addArgument(type, builder.getUnknownLoc());
  combinerBlock.addArgument(type, builder.getUnknownLoc());
  
  builder.setInsertionPointToStart(&combinerBlock);
  
  mlir::Value lhs = combinerBlock.getArgument(0);
  mlir::Value rhs = combinerBlock.getArgument(1);
  mlir::Value result;
  
  // Generate operation based on operator
  if (const auto *intrinsicOp = 
      std::get_if<parser::OmpReductionOperator::Operator>(&op.u)) {
    using Operator = parser::OmpReductionOperator::Operator;
    
    switch (*intrinsicOp) {
      case Operator::Add:
        if (mlir::isa<mlir::IntegerType>(type)) {
          result = builder.create<mlir::arith::AddIOp>(
              builder.getUnknownLoc(), lhs, rhs);
        } else {
          result = builder.create<mlir::arith::AddFOp>(
              builder.getUnknownLoc(), lhs, rhs);
        }
        break;
      
      case Operator::Multiply:
        if (mlir::isa<mlir::IntegerType>(type)) {
          result = builder.create<mlir::arith::MulIOp>(
              builder.getUnknownLoc(), lhs, rhs);
        } else {
          result = builder.create<mlir::arith::MulFOp>(
              builder.getUnknownLoc(), lhs, rhs);
        }
        break;
      
      // ... other operators ...
    }
  }
  
  builder.create<mlir::omp::YieldOp>(builder.getUnknownLoc(), result);
}
```

**Step 4.2: Integrate into parallel lowering**

Location: `flang/lib/Lower/OpenMP/OpenMP.cpp` (around line 2000)

```cpp
static mlir::omp::ParallelOp genParallelOp(
    lower::AbstractConverter &converter,
    const parser::OmpClauseList &clauseList,
    mlir::Location loc) {
  
  ClauseProcessor cp(converter, clauseList);
  
  llvm::SmallVector<mlir::Value> reductionVars;
  llvm::SmallVector<mlir::Attribute> reductionSyms;
  
  cp.processReduction(loc, reductionVars, reductionSyms);
  
  auto &builder = converter.getFirOpBuilder();
  auto parallelOp = builder.create<mlir::omp::ParallelOp>(
      loc,
      /*if_expr=*/nullptr,
      /*num_threads=*/nullptr,
      /*private_vars=*/llvm::SmallVector<mlir::Value>(),
      reductionVars,
      reductionSyms.empty() ? nullptr : builder.getArrayAttr(reductionSyms)
  );
  
  return parallelOp;
}
```

**Step 4.3: Create lowering test**

File: `flang/test/Lower/OpenMP/parallel-reduction.f90`

```fortran
! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s

program test_reduction
  integer :: sum
  
  sum = 0
  
  ! CHECK: omp.reduction.declare @{{.*}} : i32
  ! CHECK: combiner
  ! CHECK:   arith.addi
  ! CHECK: omp.parallel reduction(@{{.*}} %{{.*}})
  !$omp parallel reduction(+:sum)
    sum = sum + 1
  !$omp end parallel

end program
```

---

### Stage 5: MLIR to LLVM IR Translation

**Files to modify:**
1. `mlir/lib/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.cpp`

**Step 5.1: Translate reduction to runtime calls**

Location: `mlir/lib/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.cpp` (around line 1200)

```cpp
static LogicalResult convertOmpParallelOp(
    omp::ParallelOp parallelOp, 
    llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) {
  
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  
  // Process reductions
  SmallVector<llvm::OpenMPIRBuilder::ReductionInfo> reductionInfos;
  
  if (!parallelOp.getReductionVars().empty()) {
    for (auto [var, sym] : llvm::zip(
        parallelOp.getReductionVars(),
        parallelOp.getReductionSyms().getAsRange<SymbolRefAttr>())) {
      
      llvm::Value *varLLVM = moduleTranslation.lookupValue(var);
      
      llvm::OpenMPIRBuilder::ReductionInfo info;
      info.Variable = varLLVM;
      info.ElementType = varLLVM->getType()->getPointerElementType();
      
      // Set combiner and initializer from declaration
      // ...
      
      reductionInfos.push_back(info);
    }
  }
  
  // Create parallel region with reductions
  auto bodyCB = [&](InsertPointTy allocaIP, InsertPointTy codegenIP) {
    // Translate body
    return llvm::Error::success();
  };
  
  ompBuilder->createParallel(
      builder.saveIP(),
      allocaIP,
      bodyCB,
      /*PrivCB=*/nullptr,
      /*FiniCB=*/nullptr,
      /*IfCondition=*/nullptr,
      /*NumThreads=*/nullptr);
  
  // Insert reduction finalization
  // Call __kmpc_reduce()
  // Handle atomic/critical strategies
  
  return success();
}
```

**Step 5.2: Create translation test**

File: `mlir/test/Target/LLVMIR/openmp-reduction.mlir`

```mlir
// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

omp.reduction.declare @add_i32 : i32
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %sum = arith.addi %arg0, %arg1 : i32
  omp.yield(%sum : i32)
}

// CHECK-LABEL: define void @test_reduction
llvm.func @test_reduction() {
  %sum = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr<i32>
  %c1 = llvm.mlir.constant(1 : i32) : i32
  
  // CHECK: call void @__kmpc_fork_call
  omp.parallel reduction(@add_i32 %sum -> %arg0 : !llvm.ptr<i32>) {
    omp.terminator
  }
  
  llvm.return
}

// CHECK: call i32 @__kmpc_reduce
```

---

### Summary: REDUCTION vs UNTIED Implementation

| **Aspect** | **UNTIED (Simple)** | **REDUCTION (Complex)** |
|------------|---------------------|-------------------------|
| Parse tree | Single `EMPTY_CLASS` | Operator + modifier + list |
| Semantic validation | Clause compatibility only | Type checking, operator matching |
| MLIR representation | `UnitAttr` (boolean) | Declaration op + symbol references |
| Lowering complexity | Single attribute | Region creation with combiner/initializer |
| Runtime calls | Simple flag in task creation | `__kmpc_reduce`, atomic ops, critical sections |
| Testing scope | Basic clause presence | Type compatibility, multiple operators |

### Files Modified for REDUCTION

| Stage | Files | Key Changes |
|-------|-------|-------------|
| Parser | `parse-tree.h`, `openmp-parsers.cpp` | Operator/modifier structures, grammar |
| Semantics | `check-omp-structure.cpp` | Type-operator validation |
| MLIR | `OpenMPOps.td` | ReductionDeclareOp, parallel op extensions |
| Lowering | `Clauses.cpp`, `OpenMP.cpp` | Declaration generation, combiner/initializer |
| Translation | `OpenMPToLLVMIRTranslation.cpp` | Runtime calls, reduction strategies |

---

## COMPLETE IMPLEMENTATION EXAMPLE: Adding ATOMIC Directive Support

This section demonstrates a **complete end-to-end implementation** of the `ATOMIC` directive, covering all 5 compilation stages. The ATOMIC directive is unique because it has multiple variants (READ, WRITE, UPDATE, CAPTURE) and requires special memory ordering semantics. This example shows how to handle directive variants and generate atomic LLVM IR instructions.

### Background: What is ATOMIC?

**OpenMP Specification**: OpenMP 5.2 Section 2.19.7  
**Directive Type**: Standalone directive with multiple variants  
**Semantics**: 
- Ensures atomic access to memory location (prevents race conditions)
- Four variants: READ, WRITE, UPDATE (default), CAPTURE
- Memory ordering clauses: SEQ_CST (default), ACQ_REL, RELEASE, ACQUIRE, RELAXED
- Atomic operations map to LLVM atomic instructions for lock-free performance

**Fortran Syntax Examples**:
```fortran
! ATOMIC UPDATE (default) - atomic read-modify-write
!$omp atomic
x = x + 1

! ATOMIC READ - atomic load
!$omp atomic read
v = x

! ATOMIC WRITE - atomic store
!$omp atomic write
x = expr

! ATOMIC CAPTURE - atomic operation + capture old/new value
!$omp atomic capture
v = x
x = x + 1
!$omp end atomic

! With memory ordering
!$omp atomic update seq_cst
counter = counter + 1
```

**Key Challenges**:
1. **Multiple variants** - Must parse and differentiate READ/WRITE/UPDATE/CAPTURE
2. **Memory ordering** - Must translate OpenMP memory clauses to LLVM atomic orderings
3. **Expression analysis** - Must identify the atomic operation (add, sub, multiply, etc.)
4. **No function calls** - ATOMIC maps directly to LLVM atomic instructions (no runtime calls)

---

### Stage 1: Parser Implementation

**Files to modify:**
1. `llvm/include/llvm/Frontend/OpenMP/OMPKinds.def`
2. `llvm/include/llvm/Frontend/OpenMP/OMP.td`
3. `flang/include/flang/Parser/parse-tree.h`
4. `flang/lib/Parser/openmp-parsers.cpp`

**Step 1.1: Register directive kinds in OMPKinds.def**

Location: `llvm/include/llvm/Frontend/OpenMP/OMPKinds.def` (around line 100)

```cpp
// Add ATOMIC directive and its variants
__OMP_DIRECTIVE_EXT(atomic, OMPD_atomic)

// Memory ordering clauses
__OMP_CLAUSE(seq_cst, OMPC_seq_cst)
__OMP_CLAUSE(acq_rel, OMPC_acq_rel)
__OMP_CLAUSE(release, OMPC_release)
__OMP_CLAUSE(acquire, OMPC_acquire)
__OMP_CLAUSE(relaxed, OMPC_relaxed)

// Atomic operation clauses
__OMP_CLAUSE(read, OMPC_read)
__OMP_CLAUSE(write, OMPC_write)
__OMP_CLAUSE(update, OMPC_update)
__OMP_CLAUSE(capture, OMPC_capture)
```

**Step 1.2: Define directive in OMP.td**

Location: `llvm/include/llvm/Frontend/OpenMP/OMP.td` (around line 300)

```tablegen
// Define ATOMIC directive with allowed clauses
def OMP_Atomic : Directive<"atomic"> {
  let allowedClauses = [
    VersionedClause<OMPC_Read>,
    VersionedClause<OMPC_Write>,
    VersionedClause<OMPC_Update>,
    VersionedClause<OMPC_Capture>,
    VersionedClause<OMPC_SeqCst>,
    VersionedClause<OMPC_AcqRel>,
    VersionedClause<OMPC_Release>,
    VersionedClause<OMPC_Acquire>,
    VersionedClause<OMPC_Relaxed>,
    VersionedClause<OMPC_Hint>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Read>,
    VersionedClause<OMPC_Write>,
    VersionedClause<OMPC_Update>,
    VersionedClause<OMPC_Capture>
  ];
}
```

**Step 1.3: Define parse tree structures in parse-tree.h**

Location: `flang/include/flang/Parser/parse-tree.h` (around line 3700)

```cpp
// Memory ordering clause
struct OmpMemoryOrderClause {
  ENUM_CLASS(Kind, SeqCst, AcqRel, Release, Acquire, Relaxed)
  WRAPPER_CLASS_BOILERPLATE(OmpMemoryOrderClause, Kind);
};

// Atomic operation variant
struct OmpAtomicClause {
  ENUM_CLASS(Kind, Read, Write, Update, Capture)
  WRAPPER_CLASS_BOILERPLATE(OmpAtomicClause, Kind);
};

// Simple clauses for atomic
EMPTY_CLASS(OmpClause::Read);
EMPTY_CLASS(OmpClause::Write);
EMPTY_CLASS(OmpClause::Update);
EMPTY_CLASS(OmpClause::Capture);
EMPTY_CLASS(OmpClause::SeqCst);
EMPTY_CLASS(OmpClause::AcqRel);
EMPTY_CLASS(OmpClause::Release);
EMPTY_CLASS(OmpClause::Acquire);
EMPTY_CLASS(OmpClause::Relaxed);

// ATOMIC directive structure
struct OmpAtomicDirective {
  TUPLE_CLASS_BOILERPLATE(OmpAtomicDirective);
  CharBlock source;
  std::tuple<OmpDirective,  // ATOMIC
             std::list<OmpClause>,  // Clauses (read/write/update/capture, memory order)
             OmpAtomicStatements> t;  // The atomic operation statement(s)
};

// Statements for atomic operations
struct OmpAtomicStatements {
  TUPLE_CLASS_BOILERPLATE(OmpAtomicStatements);
  // For UPDATE: single assignment
  // For CAPTURE: two assignments or compound statement
  // For READ: v = x
  // For WRITE: x = expr
  std::tuple<Statement<AssignmentStmt>,  // First statement
             std::optional<Statement<AssignmentStmt>>,  // Second statement (CAPTURE only)
             std::optional<OmpEndAtomic>> t;  // Optional END ATOMIC
};

struct OmpEndAtomic {
  WRAPPER_CLASS_BOILERPLATE(OmpEndAtomic, OmpDirective);
  CharBlock source;
};
```

**Step 1.4: Implement grammar in openmp-parsers.cpp**

Location: `flang/lib/Parser/openmp-parsers.cpp` (around line 200)

```cpp
// Memory ordering clause parsers
TYPE_PARSER(construct<OmpClause::SeqCst>("SEQ_CST"_tok))
TYPE_PARSER(construct<OmpClause::AcqRel>("ACQ_REL"_tok))
TYPE_PARSER(construct<OmpClause::Release>("RELEASE"_tok))
TYPE_PARSER(construct<OmpClause::Acquire>("ACQUIRE"_tok))
TYPE_PARSER(construct<OmpClause::Relaxed>("RELAXED"_tok))

// Atomic operation clause parsers
TYPE_PARSER(construct<OmpClause::Read>("READ"_tok))
TYPE_PARSER(construct<OmpClause::Write>("WRITE"_tok))
TYPE_PARSER(construct<OmpClause::Update>("UPDATE"_tok))
TYPE_PARSER(construct<OmpClause::Capture>("CAPTURE"_tok))

// ATOMIC directive parser
TYPE_PARSER(construct<OmpAtomicDirective>(
    verbatim("!$OMP ATOMIC"_tok),
    many(Parser<OmpClause>{}),  // Optional clauses
    Parser<OmpAtomicStatements>{}))

// ATOMIC statements parser
TYPE_PARSER(construct<OmpAtomicStatements>(
    statement(Parser<AssignmentStmt>{}),  // First statement (required)
    maybe(statement(Parser<AssignmentStmt>{})),  // Second statement (CAPTURE only)
    maybe(Parser<OmpEndAtomic>{})))  // Optional END ATOMIC

TYPE_PARSER(construct<OmpEndAtomic>(
    verbatim("!$OMP END ATOMIC"_tok)))
```

**Step 1.5: Create parser test**

File: `flang/test/Parser/OpenMP/atomic-variants.f90`

```fortran
! RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck --check-prefix=PARSE-TREE %s

program test_atomic
  integer :: x, v, expr
  
  ! CHECK: !$OMP ATOMIC
  !$omp atomic
  x = x + 1
  
  ! CHECK: !$OMP ATOMIC READ
  !$omp atomic read
  v = x
  
  ! CHECK: !$OMP ATOMIC WRITE
  !$omp atomic write
  x = expr
  
  ! CHECK: !$OMP ATOMIC UPDATE SEQ_CST
  !$omp atomic update seq_cst
  x = x * 2
  
  ! CHECK: !$OMP ATOMIC CAPTURE
  !$omp atomic capture
  v = x
  x = x + 1
  !$omp end atomic
  
  ! CHECK: !$OMP ATOMIC CAPTURE RELAXED
  !$omp atomic capture relaxed
  x = x - 1
  v = x
  !$omp end atomic
  
end program

! PARSE-TREE: OmpAtomicDirective
! PARSE-TREE: OmpClause -> Read
! PARSE-TREE: OmpClause -> SeqCst
```

---

### Stage 2: Semantic Analysis

**Files to modify:**
1. `flang/lib/Semantics/check-omp-structure.cpp`
2. `flang/lib/Semantics/resolve-directives.cpp`

**Step 2.1: Add semantic checks in check-omp-structure.cpp**

Location: `flang/lib/Semantics/check-omp-structure.cpp` (around line 800)

```cpp
void OmpStructureChecker::Enter(const parser::OmpAtomicDirective &x) {
  const auto &dir{std::get<parser::OmpDirective>(x.t)};
  const auto &clauses{std::get<std::list<parser::OmpClause>>(x.t)};
  
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_atomic);
  
  // Check for mutually exclusive operation clauses
  bool hasRead = HasClause(clauses, llvm::omp::Clause::OMPC_read);
  bool hasWrite = HasClause(clauses, llvm::omp::Clause::OMPC_write);
  bool hasUpdate = HasClause(clauses, llvm::omp::Clause::OMPC_update);
  bool hasCapture = HasClause(clauses, llvm::omp::Clause::OMPC_capture);
  
  int operationCount = hasRead + hasWrite + hasUpdate + hasCapture;
  if (operationCount > 1) {
    context_.Say(dir.source,
        "ATOMIC directive can have at most one of READ, WRITE, UPDATE, or CAPTURE clauses"_err_en_US);
  }
  
  // Check for mutually exclusive memory ordering clauses
  bool hasSeqCst = HasClause(clauses, llvm::omp::Clause::OMPC_seq_cst);
  bool hasAcqRel = HasClause(clauses, llvm::omp::Clause::OMPC_acq_rel);
  bool hasRelease = HasClause(clauses, llvm::omp::Clause::OMPC_release);
  bool hasAcquire = HasClause(clauses, llvm::omp::Clause::OMPC_acquire);
  bool hasRelaxed = HasClause(clauses, llvm::omp::Clause::OMPC_relaxed);
  
  int orderingCount = hasSeqCst + hasAcqRel + hasRelease + hasAcquire + hasRelaxed;
  if (orderingCount > 1) {
    context_.Say(dir.source,
        "ATOMIC directive can have at most one memory ordering clause"_err_en_US);
  }
  
  // Validate memory ordering for specific atomic variants
  if (hasRead && hasRelease) {
    context_.Say(dir.source,
        "ATOMIC READ cannot have RELEASE memory ordering"_err_en_US);
  }
  if (hasWrite && hasAcquire) {
    context_.Say(dir.source,
        "ATOMIC WRITE cannot have ACQUIRE memory ordering"_err_en_US);
  }
}

void OmpStructureChecker::Leave(const parser::OmpAtomicDirective &) {
  dirContext_.pop_back();
}

// Validate atomic operation statement
void OmpStructureChecker::CheckAtomicOperation(
    const parser::AssignmentStmt &assignment,
    llvm::omp::Clause::OMPC atomicVariant) {
  
  // Get LHS and RHS of assignment
  const auto &var{std::get<parser::Variable>(assignment.t)};
  const auto &expr{std::get<parser::Expr>(assignment.t)};
  
  // For ATOMIC READ: v = x (simple variable reference on RHS)
  if (atomicVariant == llvm::omp::Clause::OMPC_read) {
    // Verify RHS is a simple variable reference
    if (!IsSimpleVariableReference(expr)) {
      context_.Say(GetContext().directiveSource,
          "ATOMIC READ requires a simple variable on the right-hand side"_err_en_US);
    }
  }
  
  // For ATOMIC WRITE: x = expr (expr must not reference x)
  if (atomicVariant == llvm::omp::Clause::OMPC_write) {
    // Verify RHS doesn't reference LHS variable
    if (ReferencesVariable(expr, var)) {
      context_.Say(GetContext().directiveSource,
          "ATOMIC WRITE expression cannot reference the updated variable"_err_en_US);
    }
  }
  
  // For ATOMIC UPDATE: x = x op expr OR x = expr op x OR x = intrinsic(x, expr)
  if (atomicVariant == llvm::omp::Clause::OMPC_update) {
    // Verify it follows allowed patterns
    if (!IsValidAtomicUpdate(assignment)) {
      context_.Say(GetContext().directiveSource,
          "ATOMIC UPDATE must have form 'x = x op expr', 'x = expr op x', or 'x = intrinsic(x, expr)'"_err_en_US);
    }
  }
  
  // Check that atomic variable is scalar
  if (const auto *symbol = GetSymbolFromVariable(var)) {
    if (!symbol->Rank() == 0) {
      context_.Say(GetContext().directiveSource,
          "ATOMIC variable must be a scalar"_err_en_US);
    }
    
    // Check for allowed types (integer, real, logical for most operations)
    const auto &type = symbol->GetType();
    if (!IsValidAtomicType(type, atomicVariant)) {
      context_.Say(GetContext().directiveSource,
          "ATOMIC variable has invalid type for this operation"_err_en_US);
    }
  }
}
```

**Step 2.2: Create semantic test**

File: `flang/test/Semantics/OpenMP/atomic-errors.f90`

```fortran
! RUN: %python %S/../test_errors.py %s %flang -fopenmp

program test_atomic_errors
  integer :: x, v, y
  integer :: arr(10)
  real :: r
  
  ! OK: Valid ATOMIC UPDATE
  !$omp atomic
  x = x + 1
  
  !ERROR: ATOMIC directive can have at most one of READ, WRITE, UPDATE, or CAPTURE clauses
  !$omp atomic read write
  v = x
  
  !ERROR: ATOMIC directive can have at most one memory ordering clause
  !$omp atomic seq_cst relaxed
  x = x + 1
  
  !ERROR: ATOMIC READ cannot have RELEASE memory ordering
  !$omp atomic read release
  v = x
  
  !ERROR: ATOMIC WRITE cannot have ACQUIRE memory ordering
  !$omp atomic write acquire
  x = 5
  
  !ERROR: ATOMIC variable must be a scalar
  !$omp atomic
  arr = arr + 1
  
  !ERROR: ATOMIC READ requires a simple variable on the right-hand side
  !$omp atomic read
  v = x + 1
  
  !ERROR: ATOMIC WRITE expression cannot reference the updated variable
  !$omp atomic write
  x = x + 1
  
  !ERROR: ATOMIC UPDATE must have form 'x = x op expr'
  !$omp atomic update
  x = y + 1  ! LHS doesn't appear in RHS
  
end program
```

---

### Stage 3: MLIR Dialect Definition

**Files to modify:**
1. `mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td`
2. `mlir/include/mlir/Dialect/OpenMP/OpenMPOpsEnums.td`

**Step 3.1: Define atomic operation enum in OpenMPOpsEnums.td**

Location: `mlir/include/mlir/Dialect/OpenMP/OpenMPOpsEnums.td` (around line 150)

```tablegen
// Atomic operation kinds
def AtomicRead : I32EnumAttrCase<"read", 0>;
def AtomicWrite : I32EnumAttrCase<"write", 1>;
def AtomicUpdate : I32EnumAttrCase<"update", 2>;
def AtomicCapture : I32EnumAttrCase<"capture", 3>;

def AtomicKind : I32EnumAttr<"AtomicKind",
    "OpenMP atomic operation kind",
    [AtomicRead, AtomicWrite, AtomicUpdate, AtomicCapture]> {
  let cppNamespace = "::mlir::omp";
}

// Memory ordering for atomic operations
def OrderSeqCst : I32EnumAttrCase<"seq_cst", 0>;
def OrderAcqRel : I32EnumAttrCase<"acq_rel", 1>;
def OrderAcquire : I32EnumAttrCase<"acquire", 2>;
def OrderRelease : I32EnumAttrCase<"release", 3>;
def OrderRelaxed : I32EnumAttrCase<"relaxed", 4>;

def AtomicMemoryOrder : I32EnumAttr<"AtomicMemoryOrder",
    "OpenMP atomic memory ordering",
    [OrderSeqCst, OrderAcqRel, OrderAcquire, OrderRelease, OrderRelaxed]> {
  let cppNamespace = "::mlir::omp";
}

// Atomic binary operation kinds
def AtomicBinOpAdd : I32EnumAttrCase<"add", 0>;
def AtomicBinOpSub : I32EnumAttrCase<"sub", 1>;
def AtomicBinOpMul : I32EnumAttrCase<"mul", 2>;
def AtomicBinOpDiv : I32EnumAttrCase<"div", 3>;
def AtomicBinOpMin : I32EnumAttrCase<"min", 4>;
def AtomicBinOpMax : I32EnumAttrCase<"max", 5>;
def AtomicBinOpAnd : I32EnumAttrCase<"and", 6>;
def AtomicBinOpOr : I32EnumAttrCase<"or", 7>;
def AtomicBinOpXor : I32EnumAttrCase<"xor", 8>;

def AtomicBinOp : I32EnumAttr<"AtomicBinOp",
    "OpenMP atomic binary operation",
    [AtomicBinOpAdd, AtomicBinOpSub, AtomicBinOpMul, AtomicBinOpDiv,
     AtomicBinOpMin, AtomicBinOpMax, AtomicBinOpAnd, AtomicBinOpOr, AtomicBinOpXor]> {
  let cppNamespace = "::mlir::omp";
}
```

**Step 3.2: Define atomic operations in OpenMPOps.td**

Location: `mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td` (around line 900)

```tablegen
//===----------------------------------------------------------------------===//
// AtomicReadOp
//===----------------------------------------------------------------------===//

def AtomicReadOp : OpenMP_Op<"atomic.read", [AttrSizedOperandSegments]> {
  let summary = "performs an atomic read";
  let description = [{
    Atomically reads the value of `x` and stores it in `v`.
    
    Example:
    ```mlir
    omp.atomic.read %v = %x : !llvm.ptr<i32>, i32
    omp.atomic.read memory_order(acquire) %v = %x : !llvm.ptr<i32>, i32
    ```
  }];
  
  let arguments = (ins
    Arg<AnyType, "address to read from">:$x,
    OptionalAttr<AtomicMemoryOrder>:$memory_order,
    OptionalAttr<I64Attr>:$hint_val
  );
  
  let results = (outs AnyType:$v);
  
  let assemblyFormat = [{
    (`memory_order` `(` $memory_order^ `)`)? 
    $x `:` type($x) `,` type($v) attr-dict
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AtomicWriteOp
//===----------------------------------------------------------------------===//

def AtomicWriteOp : OpenMP_Op<"atomic.write", [AttrSizedOperandSegments]> {
  let summary = "performs an atomic write";
  let description = [{
    Atomically writes `expr` to the memory location `x`.
    
    Example:
    ```mlir
    omp.atomic.write %x = %expr : !llvm.ptr<i32>, i32
    omp.atomic.write memory_order(release) %x = %expr : !llvm.ptr<i32>, i32
    ```
  }];
  
  let arguments = (ins
    Arg<AnyType, "address to write to">:$x,
    Arg<AnyType, "value to write">:$expr,
    OptionalAttr<AtomicMemoryOrder>:$memory_order,
    OptionalAttr<I64Attr>:$hint_val
  );
  
  let assemblyFormat = [{
    (`memory_order` `(` $memory_order^ `)`)? 
    $x `=` $expr `:` type($x) `,` type($expr) attr-dict
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AtomicUpdateOp
//===----------------------------------------------------------------------===//

def AtomicUpdateOp : OpenMP_Op<"atomic.update", 
    [AttrSizedOperandSegments, SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "performs an atomic update";
  let description = [{
    Atomically performs a read-modify-write operation on `x`.
    The update operation is specified in the region.
    
    Example:
    ```mlir
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = arith.addi %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    
    // With memory ordering
    omp.atomic.update memory_order(seq_cst) %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = arith.muli %xval, %c2 : i32
      omp.yield(%newval : i32)
    }
    ```
  }];
  
  let arguments = (ins
    Arg<AnyType, "address to update">:$x,
    OptionalAttr<AtomicMemoryOrder>:$memory_order,
    OptionalAttr<I64Attr>:$hint_val
  );
  
  let regions = (region SizedRegion<1>:$region);
  
  let assemblyFormat = [{
    (`memory_order` `(` $memory_order^ `)`)? 
    $x `:` type($x) $region attr-dict
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AtomicCaptureOp
//===----------------------------------------------------------------------===//

def AtomicCaptureOp : OpenMP_Op<"atomic.capture", 
    [AttrSizedOperandSegments, SingleBlockImplicitTerminator<"TerminatorOp">]> {
  let summary = "performs an atomic capture";
  let description = [{
    Atomically performs an update operation and captures the old or new value.
    Contains two atomic operations in the region (read + update or update + read).
    
    Example:
    ```mlir
    // Capture old value: v = x; x = x + 1
    omp.atomic.capture {
      omp.atomic.read %v = %x : !llvm.ptr<i32>, i32
      omp.atomic.update %x : !llvm.ptr<i32> {
      ^bb0(%xval: i32):
        %newval = arith.addi %xval, %c1 : i32
        omp.yield(%newval : i32)
      }
    }
    
    // Capture new value: x = x + 1; v = x
    omp.atomic.capture memory_order(acq_rel) {
      omp.atomic.update %x : !llvm.ptr<i32> {
      ^bb0(%xval: i32):
        %newval = arith.addi %xval, %c1 : i32
        omp.yield(%newval : i32)
      }
      omp.atomic.read %v = %x : !llvm.ptr<i32>, i32
    }
    ```
  }];
  
  let arguments = (ins
    OptionalAttr<AtomicMemoryOrder>:$memory_order,
    OptionalAttr<I64Attr>:$hint_val
  );
  
  let regions = (region SizedRegion<1>:$region);
  
  let assemblyFormat = [{
    (`memory_order` `(` $memory_order^ `)`)? 
    $region attr-dict
  }];
  
  let hasVerifier = 1;
}
```

**Step 3.3: Add verifiers in OpenMPOps.cpp**

Location: `mlir/lib/Dialect/OpenMP/OpenMPDialect.cpp` (around line 600)

```cpp
LogicalResult AtomicReadOp::verify() {
  // Verify x is a pointer/reference type
  if (!x().getType().isa<LLVM::LLVMPointerType, MemRefType>()) {
    return emitOpError("x operand must be a pointer or memref type");
  }
  
  // Verify memory ordering is valid for READ
  if (memory_order() && 
      *memory_order() == AtomicMemoryOrder::OrderRelease) {
    return emitOpError("ATOMIC READ cannot use release memory ordering");
  }
  
  return success();
}

LogicalResult AtomicWriteOp::verify() {
  // Verify x is a pointer/reference type
  if (!x().getType().isa<LLVM::LLVMPointerType, MemRefType>()) {
    return emitOpError("x operand must be a pointer or memref type");
  }
  
  // Verify memory ordering is valid for WRITE
  if (memory_order() && 
      *memory_order() == AtomicMemoryOrder::OrderAcquire) {
    return emitOpError("ATOMIC WRITE cannot use acquire memory ordering");
  }
  
  return success();
}

LogicalResult AtomicUpdateOp::verify() {
  // Verify x is a pointer/reference type
  if (!x().getType().isa<LLVM::LLVMPointerType, MemRefType>()) {
    return emitOpError("x operand must be a pointer or memref type");
  }
  
  // Verify region has exactly one block
  if (region().getBlocks().size() != 1) {
    return emitOpError("region must have exactly one block");
  }
  
  // Verify block has one argument (the current value of x)
  Block &block = region().front();
  if (block.getNumArguments() != 1) {
    return emitOpError("region block must have exactly one argument");
  }
  
  // Verify block terminates with omp.yield
  if (!isa<YieldOp>(block.back())) {
    return emitOpError("region must terminate with omp.yield");
  }
  
  return success();
}

LogicalResult AtomicCaptureOp::verify() {
  // Verify region has exactly one block
  if (region().getBlocks().size() != 1) {
    return emitOpError("region must have exactly one block");
  }
  
  Block &block = region().front();
  
  // Must contain exactly two atomic operations (read + update or update + read)
  int atomicOps = 0;
  bool hasRead = false;
  bool hasUpdate = false;
  
  for (Operation &op : block.getOperations()) {
    if (isa<AtomicReadOp>(&op)) {
      hasRead = true;
      atomicOps++;
    } else if (isa<AtomicUpdateOp, AtomicWriteOp>(&op)) {
      hasUpdate = true;
      atomicOps++;
    } else if (!isa<TerminatorOp>(&op)) {
      return emitOpError("region can only contain atomic operations");
    }
  }
  
  if (atomicOps != 2 || !hasRead || !hasUpdate) {
    return emitOpError("ATOMIC CAPTURE must contain exactly one read and one update/write operation");
  }
  
  return success();
}
```

**Step 3.4: Create MLIR operation test**

File: `mlir/test/Dialect/OpenMP/atomic-ops.mlir`

```mlir
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// CHECK-LABEL: func @test_atomic_read
func.func @test_atomic_read(%x: !llvm.ptr<i32>) {
  // CHECK: %[[VAL:.*]] = omp.atomic.read %{{.*}} : !llvm.ptr<i32>, i32
  %v = omp.atomic.read %x : !llvm.ptr<i32>, i32
  
  // CHECK: %[[VAL2:.*]] = omp.atomic.read memory_order(acquire) %{{.*}} : !llvm.ptr<i32>, i32
  %v2 = omp.atomic.read memory_order(acquire) %x : !llvm.ptr<i32>, i32
  return
}

// CHECK-LABEL: func @test_atomic_write
func.func @test_atomic_write(%x: !llvm.ptr<i32>, %expr: i32) {
  // CHECK: omp.atomic.write %{{.*}} = %{{.*}} : !llvm.ptr<i32>, i32
  omp.atomic.write %x = %expr : !llvm.ptr<i32>, i32
  
  // CHECK: omp.atomic.write memory_order(release) %{{.*}} = %{{.*}} : !llvm.ptr<i32>, i32
  omp.atomic.write memory_order(release) %x = %expr : !llvm.ptr<i32>, i32
  return
}

// CHECK-LABEL: func @test_atomic_update
func.func @test_atomic_update(%x: !llvm.ptr<i32>, %expr: i32) {
  // CHECK: omp.atomic.update %{{.*}} : !llvm.ptr<i32>
  omp.atomic.update %x : !llvm.ptr<i32> {
  ^bb0(%xval: i32):
    %newval = arith.addi %xval, %expr : i32
    omp.yield(%newval : i32)
  }
  
  // CHECK: omp.atomic.update memory_order(seq_cst) %{{.*}} : !llvm.ptr<i32>
  %c2 = arith.constant 2 : i32
  omp.atomic.update memory_order(seq_cst) %x : !llvm.ptr<i32> {
  ^bb0(%xval: i32):
    %newval = arith.muli %xval, %c2 : i32
    omp.yield(%newval : i32)
  }
  return
}

// CHECK-LABEL: func @test_atomic_capture
func.func @test_atomic_capture(%x: !llvm.ptr<i32>, %v: !llvm.ptr<i32>) {
  %c1 = arith.constant 1 : i32
  
  // CHECK: omp.atomic.capture
  omp.atomic.capture {
    // Capture old value
    %old = omp.atomic.read %x : !llvm.ptr<i32>, i32
    llvm.store %old, %v : !llvm.ptr<i32>
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = arith.addi %xval, %c1 : i32
      omp.yield(%newval : i32)
    }
    omp.terminator
  }
  
  // CHECK: omp.atomic.capture memory_order(acq_rel)
  omp.atomic.capture memory_order(acq_rel) {
    // Capture new value
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = arith.addi %xval, %c1 : i32
      omp.yield(%newval : i32)
    }
    %new = omp.atomic.read %x : !llvm.ptr<i32>, i32
    llvm.store %new, %v : !llvm.ptr<i32>
    omp.terminator
  }
  return
}
```

---

### Stage 4: Lowering from Parse Tree to MLIR

**Files to modify:**
1. `flang/lib/Lower/OpenMP/OpenMP.cpp`
2. `flang/lib/Lower/OpenMP/Clauses.cpp`

**Step 4.1: Implement atomic directive lowering in OpenMP.cpp**

Location: `flang/lib/Lower/OpenMP/OpenMP.cpp` (around line 1500)

```cpp
static void genAtomicOp(lower::AbstractConverter &converter,
                        lower::pft::Evaluation &eval,
                        const parser::OmpAtomicDirective &atomicDir) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  
  const auto &clauses = std::get<std::list<parser::OmpClause>>(atomicDir.t);
  const auto &stmts = std::get<parser::OmpAtomicStatements>(atomicDir.t);
  
  // Determine atomic variant (READ/WRITE/UPDATE/CAPTURE)
  llvm::omp::Clause atomicVariant = llvm::omp::Clause::OMPC_update;  // default
  if (findClause<clause::Read>(clauses))
    atomicVariant = llvm::omp::Clause::OMPC_read;
  else if (findClause<clause::Write>(clauses))
    atomicVariant = llvm::omp::Clause::OMPC_write;
  else if (findClause<clause::Capture>(clauses))
    atomicVariant = llvm::omp::Clause::OMPC_capture;
  
  // Determine memory ordering
  mlir::omp::AtomicMemoryOrder memoryOrder = mlir::omp::AtomicMemoryOrder::OrderSeqCst;  // default
  if (findClause<clause::AcqRel>(clauses))
    memoryOrder = mlir::omp::AtomicMemoryOrder::OrderAcqRel;
  else if (findClause<clause::Acquire>(clauses))
    memoryOrder = mlir::omp::AtomicMemoryOrder::OrderAcquire;
  else if (findClause<clause::Release>(clauses))
    memoryOrder = mlir::omp::AtomicMemoryOrder::OrderRelease;
  else if (findClause<clause::Relaxed>(clauses))
    memoryOrder = mlir::omp::AtomicMemoryOrder::OrderRelaxed;
  
  // Get the assignment statement(s)
  const auto &firstStmt = std::get<parser::Statement<parser::AssignmentStmt>>(stmts.t);
  const auto &secondStmt = std::get<std::optional<parser::Statement<parser::AssignmentStmt>>>(stmts.t);
  
  // Lower based on atomic variant
  switch (atomicVariant) {
    case llvm::omp::Clause::OMPC_read:
      genAtomicReadOp(converter, builder, loc, firstStmt, memoryOrder);
      break;
    case llvm::omp::Clause::OMPC_write:
      genAtomicWriteOp(converter, builder, loc, firstStmt, memoryOrder);
      break;
    case llvm::omp::Clause::OMPC_update:
      genAtomicUpdateOp(converter, builder, loc, firstStmt, memoryOrder);
      break;
    case llvm::omp::Clause::OMPC_capture:
      genAtomicCaptureOp(converter, builder, loc, firstStmt, secondStmt, memoryOrder);
      break;
  }
}

// Helper: Lower ATOMIC READ
static void genAtomicReadOp(lower::AbstractConverter &converter,
                             fir::FirOpBuilder &builder,
                             mlir::Location loc,
                             const parser::Statement<parser::AssignmentStmt> &stmt,
                             mlir::omp::AtomicMemoryOrder memoryOrder) {
  // Parse: v = x
  const auto &assignment = stmt.statement;
  const auto &lhs = std::get<parser::Variable>(assignment.t);  // v
  const auto &rhs = std::get<parser::Expr>(assignment.t);      // x
  
  // Get address of x (RHS)
  mlir::Value xAddr = getAddressOfExpr(converter, rhs);
  
  // Get element type
  mlir::Type elemType = fir::unwrapRefType(xAddr.getType());
  
  // Create atomic read operation
  auto atomicReadOp = builder.create<mlir::omp::AtomicReadOp>(
      loc, elemType, xAddr);
  atomicReadOp.setMemoryOrderAttr(
      mlir::omp::AtomicMemoryOrderAttr::get(builder.getContext(), memoryOrder));
  
  // Store result to v (LHS)
  mlir::Value vAddr = getAddressOfVariable(converter, lhs);
  builder.create<fir::StoreOp>(loc, atomicReadOp.getResult(), vAddr);
}

// Helper: Lower ATOMIC WRITE
static void genAtomicWriteOp(lower::AbstractConverter &converter,
                              fir::FirOpBuilder &builder,
                              mlir::Location loc,
                              const parser::Statement<parser::AssignmentStmt> &stmt,
                              mlir::omp::AtomicMemoryOrder memoryOrder) {
  // Parse: x = expr
  const auto &assignment = stmt.statement;
  const auto &lhs = std::get<parser::Variable>(assignment.t);  // x
  const auto &rhs = std::get<parser::Expr>(assignment.t);      // expr
  
  // Get address of x (LHS)
  mlir::Value xAddr = getAddressOfVariable(converter, lhs);
  
  // Evaluate expr (RHS)
  mlir::Value exprValue = converter.genExprValue(rhs, loc);
  
  // Create atomic write operation
  auto atomicWriteOp = builder.create<mlir::omp::AtomicWriteOp>(
      loc, xAddr, exprValue);
  atomicWriteOp.setMemoryOrderAttr(
      mlir::omp::AtomicMemoryOrderAttr::get(builder.getContext(), memoryOrder));
}

// Helper: Lower ATOMIC UPDATE
static void genAtomicUpdateOp(lower::AbstractConverter &converter,
                               fir::FirOpBuilder &builder,
                               mlir::Location loc,
                               const parser::Statement<parser::AssignmentStmt> &stmt,
                               mlir::omp::AtomicMemoryOrder memoryOrder) {
  // Parse: x = x op expr  OR  x = expr op x  OR  x = intrinsic(x, expr)
  const auto &assignment = stmt.statement;
  const auto &lhs = std::get<parser::Variable>(assignment.t);  // x
  const auto &rhs = std::get<parser::Expr>(assignment.t);      // x op expr
  
  // Get address of x
  mlir::Value xAddr = getAddressOfVariable(converter, lhs);
  mlir::Type elemType = fir::unwrapRefType(xAddr.getType());
  
  // Create atomic update operation with region
  auto atomicUpdateOp = builder.create<mlir::omp::AtomicUpdateOp>(
      loc, xAddr);
  atomicUpdateOp.setMemoryOrderAttr(
      mlir::omp::AtomicMemoryOrderAttr::get(builder.getContext(), memoryOrder));
  
  // Build the update region
  mlir::Block *block = builder.createBlock(&atomicUpdateOp.getRegion());
  mlir::Value xVal = block->addArgument(elemType, loc);  // Current value of x
  
  builder.setInsertionPointToStart(block);
  
  // Analyze RHS to determine the operation
  // RHS is of form: x op expr, expr op x, or intrinsic(x, expr)
  mlir::Value newValue = genAtomicUpdateExpression(converter, builder, loc, rhs, xVal);
  
  // Yield the new value
  builder.create<mlir::omp::YieldOp>(loc, newValue);
}

// Helper: Analyze and generate atomic update expression
static mlir::Value genAtomicUpdateExpression(lower::AbstractConverter &converter,
                                              fir::FirOpBuilder &builder,
                                              mlir::Location loc,
                                              const parser::Expr &expr,
                                              mlir::Value currentVal) {
  // Traverse expression tree to find the operation
  // Handle cases: x + expr, expr + x, max(x, expr), etc.
  
  // Simplified version - handle binary operations
  if (auto *binOp = std::get_if<parser::Expr::BinaryOp>(&expr.u)) {
    const auto &leftExpr = std::get<0>(binOp->t);
    const auto &op = std::get<1>(binOp->t);
    const auto &rightExpr = std::get<2>(binOp->t);
    
    // Check if one side is the atomic variable (use currentVal)
    // and evaluate the other side
    bool leftIsX = isAtomicVariable(leftExpr);
    const parser::Expr &otherExpr = leftIsX ? rightExpr : leftExpr;
    mlir::Value otherValue = converter.genExprValue(otherExpr, loc);
    
    // Generate the appropriate operation
    switch (op) {
      case parser::Expr::Operator::Add:
        return builder.create<mlir::arith::AddIOp>(loc, currentVal, otherValue);
      case parser::Expr::Operator::Subtract:
        if (leftIsX)
          return builder.create<mlir::arith::SubIOp>(loc, currentVal, otherValue);
        else
          return builder.create<mlir::arith::SubIOp>(loc, otherValue, currentVal);
      case parser::Expr::Operator::Multiply:
        return builder.create<mlir::arith::MulIOp>(loc, currentVal, otherValue);
      // ... handle other operations
    }
  }
  
  // Handle intrinsic functions (MAX, MIN, etc.)
  if (auto *funcCall = std::get_if<parser::Expr::FunctionCall>(&expr.u)) {
    // Extract function name and arguments
    // Generate appropriate MLIR operation
  }
  
  llvm_unreachable("Unsupported atomic update expression");
}

// Helper: Lower ATOMIC CAPTURE
static void genAtomicCaptureOp(lower::AbstractConverter &converter,
                                fir::FirOpBuilder &builder,
                                mlir::Location loc,
                                const parser::Statement<parser::AssignmentStmt> &firstStmt,
                                const std::optional<parser::Statement<parser::AssignmentStmt>> &secondStmt,
                                mlir::omp::AtomicMemoryOrder memoryOrder) {
  // ATOMIC CAPTURE contains two statements:
  // Either: v = x; x = x op expr  (capture old value)
  // Or:     x = x op expr; v = x  (capture new value)
  
  // Create atomic capture operation
  auto atomicCaptureOp = builder.create<mlir::omp::AtomicCaptureOp>(loc);
  atomicCaptureOp.setMemoryOrderAttr(
      mlir::omp::AtomicMemoryOrderAttr::get(builder.getContext(), memoryOrder));
  
  // Build capture region
  mlir::Block *block = builder.createBlock(&atomicCaptureOp.getRegion());
  builder.setInsertionPointToStart(block);
  
  // Determine order: read-update or update-read
  bool readFirst = isAtomicRead(firstStmt);
  
  if (readFirst) {
    // v = x; x = x op expr (capture old value)
    genAtomicReadOp(converter, builder, loc, firstStmt, memoryOrder);
    if (secondStmt)
      genAtomicUpdateOp(converter, builder, loc, *secondStmt, memoryOrder);
  } else {
    // x = x op expr; v = x (capture new value)
    genAtomicUpdateOp(converter, builder, loc, firstStmt, memoryOrder);
    if (secondStmt)
      genAtomicReadOp(converter, builder, loc, *secondStmt, memoryOrder);
  }
  
  builder.create<mlir::omp::TerminatorOp>(loc);
}
```

**Step 4.2: Create lowering test**

File: `flang/test/Lower/OpenMP/atomic-variants.f90`

```fortran
! RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s

! CHECK-LABEL: func @test_atomic_read
subroutine test_atomic_read()
  integer :: x, v
  
  ! CHECK: omp.atomic.read %{{.*}} : !fir.ref<i32>, i32
  !$omp atomic read
  v = x
  
  ! CHECK: omp.atomic.read memory_order(acquire) %{{.*}} : !fir.ref<i32>, i32
  !$omp atomic read acquire
  v = x
end subroutine

! CHECK-LABEL: func @test_atomic_write
subroutine test_atomic_write()
  integer :: x
  
  ! CHECK: omp.atomic.write %{{.*}} = %{{.*}} : !fir.ref<i32>, i32
  !$omp atomic write
  x = 42
  
  ! CHECK: omp.atomic.write memory_order(release) %{{.*}} = %{{.*}} : !fir.ref<i32>, i32
  !$omp atomic write release
  x = 99
end subroutine

! CHECK-LABEL: func @test_atomic_update
subroutine test_atomic_update()
  integer :: x
  
  ! CHECK: omp.atomic.update %{{.*}} : !fir.ref<i32>
  ! CHECK: ^bb0(%[[VAL:.*]]: i32):
  ! CHECK:   %[[NEW:.*]] = arith.addi %[[VAL]], %{{.*}} : i32
  ! CHECK:   omp.yield(%[[NEW]] : i32)
  !$omp atomic
  x = x + 1
  
  ! CHECK: omp.atomic.update memory_order(seq_cst) %{{.*}} : !fir.ref<i32>
  ! CHECK: ^bb0(%[[VAL2:.*]]: i32):
  ! CHECK:   %[[NEW2:.*]] = arith.muli %[[VAL2]], %{{.*}} : i32
  ! CHECK:   omp.yield(%[[NEW2]] : i32)
  !$omp atomic update seq_cst
  x = x * 2
end subroutine

! CHECK-LABEL: func @test_atomic_capture
subroutine test_atomic_capture()
  integer :: x, v
  
  ! CHECK: omp.atomic.capture {
  ! CHECK:   %[[OLD:.*]] = omp.atomic.read %{{.*}} : !fir.ref<i32>, i32
  ! CHECK:   fir.store %[[OLD]], %{{.*}} : !fir.ref<i32>
  ! CHECK:   omp.atomic.update %{{.*}} : !fir.ref<i32>
  ! CHECK:   omp.terminator
  ! CHECK: }
  !$omp atomic capture
  v = x
  x = x + 1
  !$omp end atomic
end subroutine
```

---

### Stage 5: LLVM IR Translation

**Files to modify:**
1. `mlir/lib/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.cpp`

**Step 5.1: Implement atomic operation translation**

Location: `mlir/lib/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.cpp` (around line 800)

```cpp
// Convert OpenMP memory ordering to LLVM atomic ordering
static llvm::AtomicOrdering getLLVMAtomicOrdering(mlir::omp::AtomicMemoryOrder order) {
  switch (order) {
    case mlir::omp::AtomicMemoryOrder::OrderSeqCst:
      return llvm::AtomicOrdering::SequentiallyConsistent;
    case mlir::omp::AtomicMemoryOrder::OrderAcqRel:
      return llvm::AtomicOrdering::AcquireRelease;
    case mlir::omp::AtomicMemoryOrder::OrderAcquire:
      return llvm::AtomicOrdering::Acquire;
    case mlir::omp::AtomicMemoryOrder::OrderRelease:
      return llvm::AtomicOrdering::Release;
    case mlir::omp::AtomicMemoryOrder::OrderRelaxed:
      return llvm::AtomicOrdering::Monotonic;
  }
  llvm_unreachable("Unknown atomic memory ordering");
}

// Translate omp.atomic.read to LLVM IR
static LogicalResult convertOmpAtomicRead(
    Operation *op, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) {
  
  auto atomicReadOp = cast<mlir::omp::AtomicReadOp>(op);
  mlir::Location loc = atomicReadOp.getLoc();
  
  // Get address to read from
  llvm::Value *xAddr = moduleTranslation.lookupValue(atomicReadOp.getX());
  
  // Get memory ordering
  llvm::AtomicOrdering ordering = llvm::AtomicOrdering::SequentiallyConsistent;
  if (atomicReadOp.getMemoryOrder())
    ordering = getLLVMAtomicOrdering(*atomicReadOp.getMemoryOrder());
  
  // Create LLVM atomic load instruction
  llvm::Type *elemType = xAddr->getType()->getPointerElementType();
  llvm::LoadInst *loadInst = builder.CreateLoad(elemType, xAddr);
  loadInst->setAtomic(ordering);
  loadInst->setAlignment(llvm::Align(elemType->getPrimitiveSizeInBits() / 8));
  
  // Map the result value
  moduleTranslation.mapValue(atomicReadOp.getResult(), loadInst);
  
  return success();
}

// Translate omp.atomic.write to LLVM IR
static LogicalResult convertOmpAtomicWrite(
    Operation *op, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) {
  
  auto atomicWriteOp = cast<mlir::omp::AtomicWriteOp>(op);
  
  // Get address and value
  llvm::Value *xAddr = moduleTranslation.lookupValue(atomicWriteOp.getX());
  llvm::Value *expr = moduleTranslation.lookupValue(atomicWriteOp.getExpr());
  
  // Get memory ordering
  llvm::AtomicOrdering ordering = llvm::AtomicOrdering::SequentiallyConsistent;
  if (atomicWriteOp.getMemoryOrder())
    ordering = getLLVMAtomicOrdering(*atomicWriteOp.getMemoryOrder());
  
  // Create LLVM atomic store instruction
  llvm::StoreInst *storeInst = builder.CreateStore(expr, xAddr);
  storeInst->setAtomic(ordering);
  storeInst->setAlignment(llvm::Align(expr->getType()->getPrimitiveSizeInBits() / 8));
  
  return success();
}

// Translate omp.atomic.update to LLVM IR
static LogicalResult convertOmpAtomicUpdate(
    Operation *op, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) {
  
  auto atomicUpdateOp = cast<mlir::omp::AtomicUpdateOp>(op);
  mlir::Location loc = atomicUpdateOp.getLoc();
  
  // Get address
  llvm::Value *xAddr = moduleTranslation.lookupValue(atomicUpdateOp.getX());
  llvm::Type *elemType = xAddr->getType()->getPointerElementType();
  
  // Get memory ordering
  llvm::AtomicOrdering ordering = llvm::AtomicOrdering::SequentiallyConsistent;
  if (atomicUpdateOp.getMemoryOrder())
    ordering = getLLVMAtomicOrdering(*atomicUpdateOp.getMemoryOrder());
  
  // Check if the update region can be translated to a single atomicrmw instruction
  Block &block = atomicUpdateOp.getRegion().front();
  mlir::Value blockArg = block.getArgument(0);
  
  // Analyze the region to determine if it's a simple binary operation
  if (auto rmwOp = extractAtomicRMWOperation(block, blockArg)) {
    // Use LLVM atomicrmw instruction (lock-free on most platforms)
    llvm::AtomicRMWInst::BinOp binOp = rmwOp->first;
    llvm::Value *operand = rmwOp->second;
    
    llvm::AtomicRMWInst *rmwInst = builder.CreateAtomicRMW(
        binOp, xAddr, operand, llvm::MaybeAlign(), ordering);
    
    return success();
  }
  
  // Fall back to compare-and-swap loop for complex operations
  return convertAtomicUpdateWithCAS(atomicUpdateOp, xAddr, elemType, ordering,
                                     builder, moduleTranslation);
}

// Helper: Try to extract atomic RMW operation from region
static std::optional<std::pair<llvm::AtomicRMWInst::BinOp, llvm::Value*>>
extractAtomicRMWOperation(Block &block, mlir::Value currentVal) {
  // Look for simple pattern: yield(op(currentVal, otherVal))
  
  if (block.getOperations().size() != 2)  // Should have 1 op + yield
    return std::nullopt;
  
  Operation *computeOp = &block.front();
  Operation *yieldOp = &block.back();
  
  if (!isa<mlir::omp::YieldOp>(yieldOp))
    return std::nullopt;
  
  // Check if it's a binary arithmetic operation
  if (auto addOp = dyn_cast<mlir::arith::AddIOp>(computeOp)) {
    // Find the non-blockArg operand
    llvm::Value *otherVal = (addOp.getLhs() == currentVal) ?
        moduleTranslation.lookupValue(addOp.getRhs()) :
        moduleTranslation.lookupValue(addOp.getLhs());
    return std::make_pair(llvm::AtomicRMWInst::Add, otherVal);
  }
  
  if (auto subOp = dyn_cast<mlir::arith::SubIOp>(computeOp)) {
    if (subOp.getLhs() == currentVal) {
      llvm::Value *otherVal = moduleTranslation.lookupValue(subOp.getRhs());
      return std::make_pair(llvm::AtomicRMWInst::Sub, otherVal);
    }
  }
  
  if (auto mulOp = dyn_cast<mlir::arith::MulIOp>(computeOp)) {
    llvm::Value *otherVal = (mulOp.getLhs() == currentVal) ?
        moduleTranslation.lookupValue(mulOp.getRhs()) :
        moduleTranslation.lookupValue(mulOp.getLhs());
    // Note: LLVM doesn't have AtomicRMW for multiply
    return std::nullopt;  // Fall back to CAS
  }
  
  // Add more operation patterns: And, Or, Xor, Max, Min, etc.
  
  return std::nullopt;
}

// Helper: Fall back to CAS loop for complex atomic updates
static LogicalResult convertAtomicUpdateWithCAS(
    mlir::omp::AtomicUpdateOp atomicUpdateOp,
    llvm::Value *xAddr,
    llvm::Type *elemType,
    llvm::AtomicOrdering ordering,
    llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) {
  
  // Generate compare-and-swap loop:
  // do {
  //   old = atomic_load(x)
  //   new = compute(old)
  // } while (!atomic_compare_exchange(x, old, new))
  
  llvm::LLVMContext &ctx = builder.getContext();
  llvm::Function *func = builder.GetInsertBlock()->getParent();
  
  llvm::BasicBlock *entryBB = builder.GetInsertBlock();
  llvm::BasicBlock *loopBB = llvm::BasicBlock::Create(ctx, "atomic.loop", func);
  llvm::BasicBlock *exitBB = llvm::BasicBlock::Create(ctx, "atomic.exit", func);
  
  builder.CreateBr(loopBB);
  builder.SetInsertPoint(loopBB);
  
  // Load current value
  llvm::LoadInst *oldVal = builder.CreateLoad(elemType, xAddr);
  oldVal->setAtomic(llvm::AtomicOrdering::Monotonic);
  
  // Compute new value by translating the update region
  Block &block = atomicUpdateOp.getRegion().front();
  moduleTranslation.mapValue(block.getArgument(0), oldVal);
  
  for (Operation &op : block.without_terminator()) {
    if (failed(moduleTranslation.convertOperation(op, builder)))
      return failure();
  }
  
  auto yieldOp = cast<mlir::omp::YieldOp>(block.back());
  llvm::Value *newVal = moduleTranslation.lookupValue(yieldOp.getResults()[0]);
  
  // Compare-and-swap
  llvm::AtomicCmpXchgInst *casInst = builder.CreateAtomicCmpXchg(
      xAddr, oldVal, newVal, llvm::MaybeAlign(), ordering,
      llvm::AtomicOrdering::Monotonic);
  
  llvm::Value *success = builder.CreateExtractValue(casInst, 1);
  builder.CreateCondBr(success, exitBB, loopBB);
  
  builder.SetInsertPoint(exitBB);
  
  return LogicalResult::success();
}

// Translate omp.atomic.capture to LLVM IR
static LogicalResult convertOmpAtomicCapture(
    Operation *op, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) {
  
  auto atomicCaptureOp = cast<mlir::omp::AtomicCaptureOp>(op);
  
  // Translate each operation in the capture region
  Block &block = atomicCaptureOp.getRegion().front();
  
  for (Operation &innerOp : block.without_terminator()) {
    if (isa<mlir::omp::AtomicReadOp, mlir::omp::AtomicWriteOp, 
            mlir::omp::AtomicUpdateOp>(&innerOp)) {
      if (failed(moduleTranslation.convertOperation(innerOp, builder)))
        return failure();
    }
  }
  
  return success();
}

// Register atomic operation converters
void mlir::registerOpenMPAtomicTranslation() {
  TranslateFromMLIRRegistration registration(
      "atomic-to-llvmir", "Translate OpenMP atomic ops to LLVM IR",
      [](Operation *op, llvm::IRBuilderBase &builder,
         LLVM::ModuleTranslation &moduleTranslation) {
        return llvm::TypeSwitch<Operation *, LogicalResult>(op)
            .Case([&](mlir::omp::AtomicReadOp) {
              return convertOmpAtomicRead(op, builder, moduleTranslation);
            })
            .Case([&](mlir::omp::AtomicWriteOp) {
              return convertOmpAtomicWrite(op, builder, moduleTranslation);
            })
            .Case([&](mlir::omp::AtomicUpdateOp) {
              return convertOmpAtomicUpdate(op, builder, moduleTranslation);
            })
            .Case([&](mlir::omp::AtomicCaptureOp) {
              return convertOmpAtomicCapture(op, builder, moduleTranslation);
            })
            .Default([](Operation *) { return success(); });
      });
}
```

**Step 5.2: Create LLVM IR translation test**

File: `mlir/test/Target/LLVMIR/openmp-atomic.mlir`

```mlir
// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @test_atomic_read(%x: !llvm.ptr<i32>) {
  // CHECK: %[[VAL:.*]] = load atomic i32, ptr %{{.*}} acquire, align 4
  %v = omp.atomic.read memory_order(acquire) %x : !llvm.ptr<i32>, i32
  llvm.return
}

llvm.func @test_atomic_write(%x: !llvm.ptr<i32>) {
  %c42 = llvm.mlir.constant(42 : i32) : i32
  // CHECK: store atomic i32 42, ptr %{{.*}} release, align 4
  omp.atomic.write memory_order(release) %x = %c42 : !llvm.ptr<i32>, i32
  llvm.return
}

llvm.func @test_atomic_update_add(%x: !llvm.ptr<i32>) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[OLD:.*]] = atomicrmw add ptr %{{.*}}, i32 1 seq_cst, align 4
  omp.atomic.update %x : !llvm.ptr<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %c1 : i32
    omp.yield(%newval : i32)
  }
  llvm.return
}

llvm.func @test_atomic_update_complex(%x: !llvm.ptr<i32>) {
  %c2 = llvm.mlir.constant(2 : i32) : i32
  // CHECK: [[LOOP:.*]]:
  // CHECK:   %[[OLDVAL:.*]] = load atomic i32, ptr %{{.*}} monotonic, align 4
  // CHECK:   %[[NEWVAL:.*]] = mul i32 %[[OLDVAL]], 2
  // CHECK:   %[[CAS:.*]] = cmpxchg ptr %{{.*}}, i32 %[[OLDVAL]], i32 %[[NEWVAL]] seq_cst monotonic, align 4
  // CHECK:   %[[SUCCESS:.*]] = extractvalue { i32, i1 } %[[CAS]], 1
  // CHECK:   br i1 %[[SUCCESS]], label %{{.*}}, label %[[LOOP]]
  omp.atomic.update %x : !llvm.ptr<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.mul %xval, %c2 : i32
    omp.yield(%newval : i32)
  }
  llvm.return
}

llvm.func @test_atomic_capture(%x: !llvm.ptr<i32>, %v: !llvm.ptr<i32>) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[OLD:.*]] = atomicrmw add ptr %{{.*}}, i32 1 seq_cst, align 4
  // CHECK: store i32 %[[OLD]], ptr %{{.*}}, align 4
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %c1 : i32
      omp.yield(%newval : i32)
    }
    %old = omp.atomic.read %x : !llvm.ptr<i32>, i32
    llvm.store %old, %v : !llvm.ptr<i32>
    omp.terminator
  }
  llvm.return
}
```

---

### Summary: ATOMIC Implementation Key Points

| **Aspect** | **Implementation Details** |
|------------|---------------------------|
| **Parse Tree** | 4 variants (READ/WRITE/UPDATE/CAPTURE) + memory ordering clauses |
| **Semantic Validation** | Mutual exclusion checks, memory ordering compatibility, scalar variables only |
| **MLIR Operations** | 4 separate ops: AtomicReadOp, AtomicWriteOp, AtomicUpdateOp, AtomicCaptureOp |
| **Lowering Strategy** | Direct lowering to MLIR atomic ops (no runtime calls) |
| **LLVM Translation** | Maps to atomic load/store/rmw/cmpxchg instructions |
| **Lock-Free** | Uses CPU atomic instructions when possible (atomicrmw for simple ops, CAS loop for complex) |
| **Memory Ordering** | OpenMP clauses map to LLVM atomic orderings (seq_cst, acquire, release, etc.) |

### Unique Challenges Solved

1. **Multiple Variants**: Used separate MLIR operations for each variant instead of attributes
2. **Memory Ordering**: Created proper enum mapping from OpenMP to LLVM semantics
3. **Expression Analysis**: Analyzed update region to choose between atomicrmw vs CAS loop
4. **No Runtime Dependency**: Direct lowering to LLVM atomic instructions (unlike PARALLEL/TASK)
5. **Capture Semantics**: Handled two-statement atomic operations with proper ordering

### Files Modified for ATOMIC

| Stage | Files | Key Changes |
|-------|-------|-------------|
| Parser | `OMPKinds.def`, `OMP.td`, `parse-tree.h`, `openmp-parsers.cpp` | 4 variants + memory ordering |
| Semantics | `check-omp-structure.cpp` | Mutual exclusion, ordering compatibility |
| MLIR | `OpenMPOps.td`, `OpenMPOpsEnums.td`, `OpenMPDialect.cpp` | 4 atomic ops with regions |
| Lowering | `OpenMP.cpp`, `Clauses.cpp` | Expression analysis, region building |
| Translation | `OpenMPToLLVMIRTranslation.cpp` | atomicrmw, CAS loops, proper orderings |

---

## Common Pitfalls & How to Avoid Them

### Pitfall 1: Missing Clause Validation

**Problem:** Adding clause to MLIR without semantic checks  
**Consequence:** Invalid Fortran code compiles, runtime crashes  
**Solution:** Always add validation in `check-omp-structure.cpp` first

**Example:**
```fortran
! Invalid: PRIVATE on BARRIER (should be caught)
!$omp barrier private(x)  ! Semantic checker must reject this
```

### Pitfall 2: Incorrect Parse Tree Structure

**Problem:** Using wrong parse tree macro (`EMPTY_CLASS` vs `TUPLE_CLASS`)  
**Consequence:** Compilation errors, incorrect clause parsing  
**Solution:** Match macro to clause structure:
- No arguments → `EMPTY_CLASS`
- Single element → `WRAPPER_CLASS_BOILERPLATE`
- Multiple elements → `TUPLE_CLASS_BOILERPLATE`

### Pitfall 3: Forgetting Symbol Mapping

**Problem:** Not mapping block arguments to LLVM values  
**Consequence:** Undefined value errors during translation  
**Solution:** Always call `moduleTranslation.mapValue(blockArg, llvmValue)`

### Pitfall 4: Missing Test Coverage

**Problem:** Adding feature without negative tests  
**Consequence:** Regressions go unnoticed  
**Solution:** Write both positive (valid) and negative (error) tests:
- `flang/test/Parser/OpenMP/` - Parser tests
- `flang/test/Semantics/OpenMP/` - Semantic error tests (use `!ERROR:` comments)
- `flang/test/Lower/OpenMP/` - Lowering tests
- `mlir/test/Target/LLVMIR/` - Translation tests

### Pitfall 5: Hardcoding Runtime Behavior

**Problem:** Assuming specific OpenMP runtime implementation  
**Consequence:** Code breaks with different runtime libraries  
**Solution:** Use `OMPIRBuilder` abstraction layer, not direct runtime calls

### Pitfall 6: Ignoring OpenMP Specification

**Problem:** Implementing feature based on assumptions  
**Consequence:** Non-compliant behavior, spec violations  
**Solution:** Always reference OpenMP spec section numbers in code comments

**Example:**
```cpp
// OpenMP 5.2 [2.10.1] - CRITICAL construct
// Restriction: A thread cannot enter a critical region while holding a lock
```

---

## IMPLEMENTATION PATTERN: Loop-Based Constructs (SIMD Directive Example)

This section demonstrates the implementation pattern for **loop-based OpenMP constructs** using the `SIMD` directive as a reference. Loop-based constructs (SIMD, DO, DISTRIBUTE, TASKLOOP) share common patterns for handling iteration spaces, loop bounds, and vectorization/parallelization hints.

### Background: SIMD Directive

**OpenMP Specification**: OpenMP 5.2 Section 2.10.4  
**Purpose**: Declares that iterations of associated loops can be executed concurrently using SIMD (Single Instruction Multiple Data) instructions  
**Key Characteristics**:
- Operates on loop constructs (DO loops in Fortran)
- Requires loop iteration space analysis
- Supports vectorization clauses (SAFELEN, SIMDLEN, ALIGNED, LINEAR)
- Must handle canonical loop form

**Fortran Syntax**:
```fortran
!$omp simd simdlen(8) aligned(array:16) linear(i:1)
do i = 1, n
  array(i) = array(i) * 2.0
end do
!$omp end simd
```

---

### Stage 1: Parser - Loop Directive Recognition

**Key Concept**: Loop directives associate with Fortran DO constructs and require special parsing to capture both the directive and the loop structure.

**File**: `flang/lib/Parser/openmp-parsers.cpp` (around line 150)

```cpp
// Parser for loop-based directives
TYPE_PARSER(construct<OmpLoopDirective>(
    // Parse the directive keyword
    "SIMD" >> pure(llvm::omp::Directive::OMPD_simd) ||
    "DO" >> pure(llvm::omp::Directive::OMPD_do) ||
    "DISTRIBUTE" >> pure(llvm::omp::Directive::OMPD_distribute) ||
    "TASKLOOP" >> pure(llvm::omp::Directive::OMPD_taskloop)))

// Complete loop construct parser
TYPE_PARSER(construct<OmpBeginLoopDirective>(
    sourced(construct<OmpBeginLoopDirective>(
        // !$omp simd clause-list
        "!$OMP"_tok >> construct<OmpLoopDirective>(Parser<OmpLoopDirective>{}),
        Parser<OmpClauseList>{}))))

// Associate directive with DO loop
TYPE_PARSER(construct<OpenMPLoopConstruct>(
    // Begin directive: !$omp simd ...
    Parser<OmpBeginLoopDirective>{} / endOfStmt,
    // The DO loop itself
    Parser<DoConstruct>{},
    // Optional end directive: !$omp end simd
    maybe(Parser<OmpEndLoopDirective>{} / endOfStmt)))
```

**File**: `flang/include/flang/Parser/parse-tree.h` (around line 3600)

```cpp
// Loop directive structure
struct OpenMPLoopConstruct {
  TUPLE_CLASS_BOILERPLATE(OpenMPLoopConstruct);
  // (begin directive, DO construct, optional end directive)
  std::tuple<OmpBeginLoopDirective, 
             DoConstruct, 
             std::optional<OmpEndLoopDirective>> t;
};

// SIMD-specific clauses
struct OmpClause::Simdlen {
  WRAPPER_CLASS_BOILERPLATE(OmpClause::Simdlen, ScalarIntConstantExpr);
};

struct OmpClause::Safelen {
  WRAPPER_CLASS_BOILERPLATE(OmpClause::Safelen, ScalarIntConstantExpr);
};

struct OmpClause::Aligned {
  TUPLE_CLASS_BOILERPLATE(OmpClause::Aligned);
  // (variable list, optional alignment value)
  std::tuple<OmpObjectList, std::optional<ScalarIntConstantExpr>> t;
};

struct OmpClause::Linear {
  TUPLE_CLASS_BOILERPLATE(OmpClause::Linear);
  // (modifier, variable list, optional step)
  std::tuple<std::optional<OmpLinearModifier>,
             OmpObjectList,
             std::optional<ScalarIntExpr>> t;
};
```

**Test**: `flang/test/Parser/OpenMP/simd-directive.f90`

```fortran
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck %s

program test_simd_parsing
  integer :: i, n
  real :: array(100)
  
  n = 100
  
  ! CHECK: OpenMPLoopConstruct
  ! CHECK: OmpBeginLoopDirective
  ! CHECK: OmpLoopDirective -> llvm::omp::Directive = simd
  !$omp simd
  do i = 1, n
    array(i) = i * 2.0
  end do
  !$omp end simd
  
  ! CHECK: OmpClause -> Simdlen
  ! CHECK: ScalarIntConstantExpr
  !$omp simd simdlen(8)
  do i = 1, n
    array(i) = array(i) + 1.0
  end do
  
  ! CHECK: OmpClause -> Aligned
  ! CHECK: OmpObjectList
  !$omp simd aligned(array:16)
  do i = 1, n
    array(i) = array(i) * 2.0
  end do

end program
```

---

### Stage 2: Semantics - Loop Structure Validation

**Key Concept**: Loop directives must validate the loop structure is in canonical form and clauses don't violate restrictions.

**File**: `flang/lib/Semantics/check-omp-structure.cpp` (around line 2800)

```cpp
void OmpStructureChecker::Enter(const parser::OpenMPLoopConstruct &x) {
  const auto &beginDir{std::get<parser::OmpBeginLoopDirective>(x.t)};
  const auto &dir{std::get<parser::OmpLoopDirective>(beginDir.t)};
  
  PushContext(beginDir.source, dir.v);
  SetClauseSets(dir.v);
  
  // Validate loop structure is canonical
  CheckLoopCanonicalForm(x);
}

// Validate canonical loop form
void OmpStructureChecker::CheckLoopCanonicalForm(
    const parser::OpenMPLoopConstruct &x) {
  
  const auto &doConstruct{std::get<parser::DoConstruct>(x.t)};
  
  // Extract loop control
  if (const auto *loopControl = 
      GetLoopControl(doConstruct)) {
    
    // Check for integer loop variable
    if (const auto *bounds = 
        std::get_if<parser::LoopControl::Bounds>(&loopControl->u)) {
      
      const auto &name{bounds->name};
      if (const auto *symbol = name.thing.symbol) {
        
        // Must be integer type
        if (!symbol->GetType() || 
            symbol->GetType()->category() != TypeCategory::Integer) {
          context_.Say(name.thing.source,
              "Loop iteration variable must be of INTEGER type"_err_en_US);
        }
        
        // Cannot have PARAMETER attribute
        if (IsParameter(*symbol)) {
          context_.Say(name.thing.source,
              "Loop iteration variable cannot be PARAMETER"_err_en_US);
        }
        
        // Track as loop iteration variable
        AddToContextObjectWithDSA(*symbol, Symbol::Flag::OmpIterVar);
      }
    }
  }
  
  // Check for premature exits (not allowed in SIMD)
  if (GetContext().directive == llvm::omp::Directive::OMPD_simd) {
    CheckNoJumpOutOfConstruct(doConstruct);
  }
}

// SIMD-specific clause validation
void OmpStructureChecker::Enter(const parser::OmpClause::Simdlen &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_simdlen);
  
  // Must be positive constant
  RequiresPositiveParameter(llvm::omp::Clause::OMPC_simdlen, x.v);
  
  // Cannot appear with SAFELEN > SIMDLEN
  if (const auto *safelen = FindClause(llvm::omp::Clause::OMPC_safelen)) {
    // Validate safelen >= simdlen
    CheckSafelenSimdlenCompatibility(safelen, x);
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Aligned &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_aligned);
  
  const auto &[objList, alignment] = x.t;
  
  // Validate each variable
  for (const auto &obj : objList.v) {
    if (const auto *name = parser::Unwrap<parser::Name>(obj)) {
      if (const auto *symbol = name->symbol) {
        
        // Must be array or pointer
        if (!IsArray(*symbol) && !IsPointer(*symbol)) {
          context_.Say(name->source,
              "ALIGNED variable must be array or pointer"_err_en_US);
        }
        
        // Check for C_PTR type (interop)
        if (IsCPtr(symbol->GetType())) {
          // Valid for C interoperability
        }
      }
    }
  }
  
  // Alignment must be power of 2
  if (alignment) {
    RequiresPowerOfTwo(llvm::omp::Clause::OMPC_aligned, *alignment);
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Linear &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_linear);
  
  const auto &[modifier, objList, step] = x.t;
  
  for (const auto &obj : objList.v) {
    if (const auto *name = parser::Unwrap<parser::Name>(obj)) {
      if (const auto *symbol = name->symbol) {
        
        // Must be integer or pointer type
        const auto *type = symbol->GetType();
        if (type && 
            type->category() != TypeCategory::Integer &&
            !IsPointer(*symbol)) {
          context_.Say(name->source,
              "LINEAR variable must be INTEGER or POINTER"_err_en_US);
        }
        
        // Cannot be reduction variable
        if (HasDataSharingAttribute(*symbol, 
            llvm::omp::Clause::OMPC_reduction)) {
          context_.Say(name->source,
              "Variable cannot be both LINEAR and REDUCTION"_err_en_US);
        }
      }
    }
  }
}
```

**Test**: `flang/test/Semantics/OpenMP/simd-errors.f90`

```fortran
! RUN: %python %S/test_errors.py %s %flang_fc1 -fopenmp

program test_simd_errors
  integer :: i, n
  real :: array(100)
  real :: scalar
  integer, parameter :: const_var = 10
  
  ! Valid SIMD
  !$omp simd
  do i = 1, n
    array(i) = i
  end do
  
  ! ERROR: Non-canonical loop
  !ERROR: Loop iteration variable must be of INTEGER type
  !$omp simd
  do r = 1.0, 10.0
    array(i) = r
  end do
  
  ! ERROR: ALIGNED on non-array
  !ERROR: ALIGNED variable must be array or pointer
  !$omp simd aligned(scalar:16)
  do i = 1, n
    scalar = i
  end do
  
  ! ERROR: LINEAR conflict with REDUCTION
  !ERROR: Variable cannot be both LINEAR and REDUCTION
  !$omp simd linear(i:1) reduction(+:i)
  do i = 1, n
    array(i) = i
  end do
  
  ! ERROR: SIMDLEN not positive
  !ERROR: SIMDLEN must be a positive integer constant
  !$omp simd simdlen(0)
  do i = 1, n
    array(i) = i
  end do

end program
```

---

### Stage 3: MLIR Dialect - Loop Operation Definition

**Key Concept**: Loop operations in MLIR represent the iteration space and carry loop-specific attributes like step, bounds, and vectorization hints.

**File**: `mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td` (around line 600)

```tablegen
def SimdLoopOp : OpenMP_Op<"simd", [AttrSizedOperandSegments, RecursiveMemoryEffects]> {
  let summary = "simd loop directive";
  let description = [{
    The simd construct can be applied to a loop to indicate that the loop can be
    transformed into a SIMD loop (that is, multiple iterations of the loop can 
    be executed concurrently using SIMD instructions).
    
    OpenMP 5.2 Section 2.10.4
    
    Example:
      omp.simd simdlen(8) aligned(%ptr : memref<?xf32> : 16)
      for (%iv) : i32 = (%lb) to (%ub) step (%step) {
        // Loop body
        omp.yield
      }
  }];

  let arguments = (ins
    // Loop bounds
    AnyType:$lower_bound,
    AnyType:$upper_bound,
    AnyType:$step,
    
    // SIMD-specific clauses
    Optional<I64>:$simdlen,
    Optional<I64>:$safelen,
    
    // Data-sharing clauses
    Variadic<AnyType>:$private_vars,
    Variadic<AnyType>:$reduction_vars,
    Variadic<AnyType>:$linear_vars,
    Variadic<AnyType>:$aligned_vars,
    
    // Attributes for clause modifiers
    OptionalAttr<I64ArrayAttr>:$linear_step_vars,
    OptionalAttr<I64ArrayAttr>:$aligned_alignments,
    OptionalAttr<ArrayAttr>:$reduction_syms,
    
    // Loop control
    OptionalAttr<OrderKindAttr>:$order_val,
    OptionalAttr<UnitAttr>:$inclusive
  );

  let regions = (region AnyRegion:$region);
  
  let builders = [
    OpBuilder<(ins "const LoopBounds &":$bounds)>
  ];

  let assemblyFormat = [{
    oilist(
      `simdlen` `(` $simdlen `)`
      | `safelen` `(` $safelen `)`
      | `private` `(` $private_vars `:` type($private_vars) `)`
      | `reduction` custom<ReductionClause>($reduction_syms, $reduction_vars)
      | `linear` custom<LinearClause>($linear_vars, $linear_step_vars)
      | `aligned` custom<AlignedClause>($aligned_vars, $aligned_alignments)
      | `order` `(` $order_val `)`
    )
    `for` custom<LoopControl>($region, $lower_bound, $upper_bound, $step, 
                              type($lower_bound), type($upper_bound), type($step))
    attr-dict
  }];

  let hasVerifier = 1;
  
  let extraClassDeclaration = [{
    /// Get the loop induction variable
    mlir::Block::BlockArgListType getIVs() {
      return getRegion().getArguments();
    }
    
    /// Check if this is a perfectly nested SIMD loop
    bool isPerfectlyNested();
  }];
}

// Helper operation for loop yield (terminator)
def YieldOp : OpenMP_Op<"yield", [Pure, ReturnLike, Terminator,
                                   ParentOneOf<["SimdLoopOp", "WsLoopOp"]>]> {
  let summary = "loop yield and termination directive";
  let description = [{
    Terminates a loop region. Can optionally yield values for reductions.
  }];
  
  let arguments = (ins Variadic<AnyType>:$results);
  
  let assemblyFormat = [{ 
    ($results^ `:` type($results))? attr-dict 
  }];
}
```

**Verification**: `mlir/lib/Dialect/OpenMP/OpenMPDialect.cpp`

```cpp
LogicalResult SimdLoopOp::verify() {
  // Verify loop bounds are same type
  if (getLowerBound().getType() != getUpperBound().getType() ||
      getLowerBound().getType() != getStep().getType()) {
    return emitOpError("loop bounds and step must have the same type");
  }
  
  // Verify SIMDLEN is positive
  if (getSimdlen() && *getSimdlen() <= 0) {
    return emitOpError("SIMDLEN must be positive");
  }
  
  // Verify SAFELEN >= SIMDLEN
  if (getSafelen() && getSimdlen()) {
    if (*getSafelen() < *getSimdlen()) {
      return emitOpError("SAFELEN must be greater than or equal to SIMDLEN");
    }
  }
  
  // Verify region has exactly one block argument (loop IV)
  if (getRegion().getNumArguments() != 1) {
    return emitOpError("loop region must have exactly one argument");
  }
  
  // Verify loop IV type matches bounds
  auto ivType = getRegion().getArgument(0).getType();
  if (ivType != getLowerBound().getType()) {
    return emitOpError("loop induction variable type must match bounds type");
  }
  
  return success();
}
```

**Test**: `mlir/test/Dialect/OpenMP/simd-loop.mlir`

```mlir
// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @simd_simple
func.func @simd_simple(%lb: i32, %ub: i32, %step: i32) {
  // CHECK: omp.simd for (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) step (%{{.*}})
  omp.simd for (%iv) : i32 = (%lb) to (%ub) step (%step) {
    %c1 = arith.constant 1 : i32
    omp.yield
  }
  return
}

// CHECK-LABEL: func @simd_with_simdlen
func.func @simd_with_simdlen(%lb: i32, %ub: i32, %step: i32) {
  // CHECK: omp.simd simdlen(8) for (%{{.*}}) : i32
  omp.simd simdlen(8) for (%iv) : i32 = (%lb) to (%ub) step (%step) {
    omp.yield
  }
  return
}

// CHECK-LABEL: func @simd_aligned
func.func @simd_aligned(%lb: i32, %ub: i32, %step: i32, %ptr: memref<?xf32>) {
  // CHECK: omp.simd aligned(%{{.*}} : memref<?xf32> : 16)
  omp.simd aligned(%ptr : memref<?xf32> : 16) 
  for (%iv) : i32 = (%lb) to (%ub) step (%step) {
    %idx = arith.index_cast %iv : i32 to index
    %val = memref.load %ptr[%idx] : memref<?xf32>
    omp.yield
  }
  return
}

// CHECK-LABEL: func @simd_reduction
func.func @simd_reduction(%lb: i32, %ub: i32, %step: i32) {
  %sum = memref.alloca() : memref<i32>
  
  // CHECK: omp.simd reduction(@add_reduction %{{.*}} -> %{{.*}} : memref<i32>)
  omp.simd reduction(@add_reduction %sum -> %arg0 : memref<i32>)
  for (%iv) : i32 = (%lb) to (%ub) step (%step) {
    %current = memref.load %arg0 : memref<i32>
    %new = arith.addi %current, %iv : i32
    memref.store %new, %arg0 : memref<i32>
    omp.yield
  }
  return
}
```

---

### Stage 4: Lowering - Loop Bounds Extraction and MLIR Generation

**Key Concept**: Extract loop bounds from Fortran DO construct and create MLIR loop operation with proper iteration space.

**File**: `flang/lib/Lower/OpenMP/OpenMP.cpp` (around line 1500)

```cpp
static mlir::omp::SimdLoopOp genSimdLoopOp(
    lower::AbstractConverter &converter,
    lower::SymMap &symTable,
    semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval,
    mlir::Location loc,
    const parser::OmpClauseList &clauseList) {
  
  auto &firOpBuilder = converter.getFirOpBuilder();
  ClauseProcessor cp(converter, semaCtx, clauseList);
  
  // Extract loop bounds from DO construct
  // The DO construct is stored in the evaluation
  const auto *doConstruct = 
      eval.getIf<parser::DoConstruct>();
  assert(doConstruct && "SIMD directive must be associated with DO loop");
  
  // Get loop control information
  const auto *loopControl = 
      GetLoopControl(*doConstruct);
  assert(loopControl && "DO loop must have loop control");
  
  const auto *bounds = 
      std::get_if<parser::LoopControl::Bounds>(&loopControl->u);
  assert(bounds && "Loop must have bounds");
  
  // Extract loop variable
  const auto &loopVar = bounds->name.thing;
  const auto *symbol = loopVar.symbol;
  
  // Lower loop bounds to MLIR values
  mlir::Value lowerBound = 
      fir::getBase(converter.genExprValue(loc, bounds->lower, symTable));
  mlir::Value upperBound = 
      fir::getBase(converter.genExprValue(loc, bounds->upper, symTable));
  
  // Handle optional step (default is 1)
  mlir::Value step;
  if (bounds->step) {
    step = fir::getBase(converter.genExprValue(loc, *bounds->step, symTable));
  } else {
    // Default step is 1
    auto intType = lowerBound.getType();
    step = firOpBuilder.createIntegerConstant(loc, intType, 1);
  }
  
  // Process SIMD-specific clauses
  mlir::IntegerAttr simdlenAttr, safelenAttr;
  cp.processSimdlen(simdlenAttr);
  cp.processSafelen(safelenAttr);
  
  // Process data-sharing clauses
  llvm::SmallVector<mlir::Value> privateVars;
  llvm::SmallVector<mlir::Value> reductionVars;
  llvm::SmallVector<mlir::Attribute> reductionSyms;
  llvm::SmallVector<mlir::Value> linearVars;
  llvm::SmallVector<mlir::Value> alignedVars;
  llvm::SmallVector<mlir::Attribute> alignedAlignments;
  
  cp.processPrivate(privateVars);
  cp.processReduction(loc, reductionVars, reductionSyms);
  cp.processLinear(linearVars);
  cp.processAligned(alignedVars, alignedAlignments);
  
  // Create SIMD loop operation
  auto simdLoopOp = firOpBuilder.create<mlir::omp::SimdLoopOp>(
      loc,
      lowerBound,
      upperBound,
      step,
      simdlenAttr ? simdlenAttr.getInt() : mlir::IntegerAttr(),
      safelenAttr ? safelenAttr.getInt() : mlir::IntegerAttr(),
      privateVars,
      reductionVars,
      linearVars,
      alignedVars,
      /*linear_step_vars=*/nullptr,
      alignedAlignments.empty() ? nullptr : 
          firOpBuilder.getArrayAttr(alignedAlignments),
      reductionSyms.empty() ? nullptr : 
          firOpBuilder.getArrayAttr(reductionSyms),
      /*order_val=*/nullptr,
      /*inclusive=*/nullptr);
  
  // Create loop region with induction variable
  auto &loopRegion = simdLoopOp.getRegion();
  auto *loopBlock = firOpBuilder.createBlock(&loopRegion);
  
  // Add loop induction variable as block argument
  auto ivType = lowerBound.getType();
  auto ivArg = loopBlock->addArgument(ivType, loc);
  
  // Bind loop variable to induction variable
  converter.bindSymbol(*symbol, ivArg);
  
  // Lower loop body
  firOpBuilder.setInsertionPointToStart(loopBlock);
  
  // Generate loop body statements
  auto *doBody = 
      &std::get<parser::Block>(doConstruct->t).value();
  for (auto &stmt : doBody->list) {
    genFIR(converter, eval, stmt);
  }
  
  // Insert loop terminator
  firOpBuilder.create<mlir::omp::YieldOp>(loc);
  
  return simdLoopOp;
}

// Helper: Process SIMDLEN clause
void ClauseProcessor::processSimdlen(mlir::IntegerAttr &result) const {
  const parser::CharBlock *source = nullptr;
  if (auto *clause = findClause<parser::OmpClause::Simdlen>(source)) {
    auto &expr = clause->v;
    
    // Evaluate constant expression
    if (auto constantValue = evaluate::ToInt64(expr)) {
      result = converter.getFirOpBuilder().getI64IntegerAttr(*constantValue);
    }
  }
}

// Helper: Process ALIGNED clause
void ClauseProcessor::processAligned(
    llvm::SmallVectorImpl<mlir::Value> &alignedVars,
    llvm::SmallVectorImpl<mlir::Attribute> &alignments) const {
  
  const parser::CharBlock *source = nullptr;
  auto alignedClauses = 
      findRepeatableClause<parser::OmpClause::Aligned>(source);
  
  for (const auto *clause : alignedClauses) {
    const auto &[objList, alignment] = clause->t;
    
    for (const auto &ompObject : objList.v) {
      if (const auto *name = parser::Unwrap<parser::Name>(ompObject)) {
        if (auto *symbol = name->symbol) {
          mlir::Value var = converter.getSymbolAddress(*symbol);
          alignedVars.push_back(var);
          
          // Get alignment value (default is type-dependent)
          if (alignment) {
            if (auto alignVal = evaluate::ToInt64(*alignment)) {
              alignments.push_back(
                  converter.getFirOpBuilder().getI64IntegerAttr(*alignVal));
            }
          } else {
            // Default alignment
            alignments.push_back(
                converter.getFirOpBuilder().getI64IntegerAttr(0));
          }
        }
      }
    }
  }
}
```

**Test**: `flang/test/Lower/OpenMP/simd-lowering.f90`

```fortran
! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s

program test_simd_lowering
  integer :: i, n
  real :: array(100), sum
  
  n = 100
  sum = 0.0
  
  ! CHECK-LABEL: func @_QQmain
  
  ! Simple SIMD loop
  ! CHECK: %[[LB:.*]] = arith.constant 1 : i32
  ! CHECK: %[[UB:.*]] = fir.load %{{.*}} : !fir.ref<i32>
  ! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK: omp.simd for (%[[IV:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) step (%[[STEP]])
  !$omp simd
  do i = 1, n
    array(i) = i * 2.0
  end do
  !$omp end simd
  
  ! SIMD with SIMDLEN
  ! CHECK: omp.simd simdlen(8) for (%{{.*}}) : i32
  !$omp simd simdlen(8)
  do i = 1, n
    array(i) = array(i) + 1.0
  end do
  
  ! SIMD with ALIGNED
  ! CHECK: omp.simd aligned(%{{.*}} : !fir.ref<!fir.array<100xf32>> : 16)
  !$omp simd aligned(array:16)
  do i = 1, n
    array(i) = array(i) * 2.0
  end do
  
  ! SIMD with REDUCTION
  ! CHECK: omp.reduction.declare @{{.*}} : f32
  ! CHECK: omp.simd reduction(@{{.*}} %{{.*}} -> %{{.*}} : !fir.ref<f32>)
  !$omp simd reduction(+:sum)
  do i = 1, n
    sum = sum + array(i)
  end do

end program
```

---

### Stage 5: LLVM IR Translation - Vectorization Metadata

**Key Concept**: SIMD loops translate to regular loops with LLVM vectorization metadata hints.

**File**: `mlir/lib/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.cpp` (around line 800)

```cpp
static LogicalResult convertOmpSimdLoop(
    omp::SimdLoopOp simdOp,
    llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) {
  
  llvm::LLVMContext &llvmContext = builder.getContext();
  llvm::Function *currentFunction = builder.GetInsertBlock()->getParent();
  
  // Translate loop bounds
  llvm::Value *lowerBound = moduleTranslation.lookupValue(simdOp.getLowerBound());
  llvm::Value *upperBound = moduleTranslation.lookupValue(simdOp.getUpperBound());
  llvm::Value *step = moduleTranslation.lookupValue(simdOp.getStep());
  
  // Create loop structure
  llvm::BasicBlock *loopPreheader = builder.GetInsertBlock();
  llvm::BasicBlock *loopHeader = 
      llvm::BasicBlock::Create(llvmContext, "simd.header", currentFunction);
  llvm::BasicBlock *loopBody = 
      llvm::BasicBlock::Create(llvmContext, "simd.body", currentFunction);
  llvm::BasicBlock *loopLatch = 
      llvm::BasicBlock::Create(llvmContext, "simd.latch", currentFunction);
  llvm::BasicBlock *loopExit = 
      llvm::BasicBlock::Create(llvmContext, "simd.exit", currentFunction);
  
  // Branch to loop header
  builder.CreateBr(loopHeader);
  
  // Loop header: PHI node for induction variable
  builder.SetInsertPoint(loopHeader);
  llvm::PHINode *iv = builder.CreatePHI(
      lowerBound->getType(), 2, "simd.iv");
  iv->addIncoming(lowerBound, loopPreheader);
  
  // Loop condition: iv <= upperBound
  llvm::Value *cond = builder.CreateICmpSLE(iv, upperBound, "simd.cond");
  builder.CreateCondBr(cond, loopBody, loopExit);
  
  // Loop body
  builder.SetInsertPoint(loopBody);
  
  // Map induction variable
  mlir::Block &simdBlock = simdOp.getRegion().front();
  moduleTranslation.mapValue(simdBlock.getArgument(0), iv);
  
  // Translate loop body operations
  for (auto &op : simdBlock.without_terminator()) {
    if (failed(moduleTranslation.convertOperation(op, builder))) {
      return failure();
    }
  }
  
  // Branch to latch
  builder.CreateBr(loopLatch);
  
  // Loop latch: increment induction variable
  builder.SetInsertPoint(loopLatch);
  llvm::Value *nextIV = builder.CreateAdd(iv, step, "simd.iv.next");
  iv->addIncoming(nextIV, loopLatch);
  builder.CreateBr(loopHeader);
  
  // Attach vectorization metadata
  llvm::MDNode *simdMD = createSimdMetadata(simdOp, llvmContext);
  loopLatch->getTerminator()->setMetadata(
      llvm::LLVMContext::MD_loop, simdMD);
  
  // Continue after loop
  builder.SetInsertPoint(loopExit);
  
  return success();
}

// Create SIMD vectorization metadata
static llvm::MDNode* createSimdMetadata(
    omp::SimdLoopOp simdOp,
    llvm::LLVMContext &ctx) {
  
  llvm::SmallVector<llvm::Metadata*> loopMD;
  
  // Self-referential loop ID
  llvm::TempMDTuple tempNode = llvm::MDNode::getTemporary(ctx, std::nullopt);
  loopMD.push_back(tempNode.get());
  
  // Add vectorization hints
  // llvm.loop.vectorize.enable = true
  loopMD.push_back(llvm::MDNode::get(ctx, {
      llvm::MDString::get(ctx, "llvm.loop.vectorize.enable"),
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt1Ty(ctx), 1))
  }));
  
  // Add SIMDLEN if specified
  if (simdOp.getSimdlen()) {
    uint64_t simdlen = *simdOp.getSimdlen();
    loopMD.push_back(llvm::MDNode::get(ctx, {
        llvm::MDString::get(ctx, "llvm.loop.vectorize.width"),
        llvm::ConstantAsMetadata::get(
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), simdlen))
    }));
  }
  
  // Add SAFELEN if specified
  if (simdOp.getSafelen()) {
    uint64_t safelen = *simdOp.getSafelen();
    loopMD.push_back(llvm::MDNode::get(ctx, {
        llvm::MDString::get(ctx, "llvm.loop.vectorize.safelen"),
        llvm::ConstantAsMetadata::get(
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), safelen))
    }));
  }
  
  // Finalize loop metadata
  llvm::MDNode *loopID = llvm::MDNode::get(ctx, loopMD);
  loopID->replaceOperandWith(0, loopID);  // Self-reference
  
  return loopID;
}

// Handle ALIGNED clause - add alignment assumptions
static void addAlignmentAssumptions(
    omp::SimdLoopOp simdOp,
    llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) {
  
  if (simdOp.getAlignedVars().empty()) return;
  
  auto alignedVars = simdOp.getAlignedVars();
  auto alignments = simdOp.getAlignedAlignments();
  
  for (auto [var, alignAttr] : llvm::zip(alignedVars, 
                                         alignments.getAsRange<mlir::IntegerAttr>())) {
    llvm::Value *varLLVM = moduleTranslation.lookupValue(var);
    uint64_t alignment = alignAttr.getInt();
    
    if (alignment > 0) {
      // Add alignment assumption using llvm.assume
      llvm::Value *ptr = builder.CreatePtrToInt(
          varLLVM, builder.getInt64Ty());
      llvm::Value *mask = builder.getInt64(alignment - 1);
      llvm::Value *aligned = builder.CreateAnd(ptr, mask);
      llvm::Value *isAligned = builder.CreateICmpEQ(
          aligned, builder.getInt64(0));
      
      // Call llvm.assume(isAligned)
      llvm::Function *assumeFn = llvm::Intrinsic::getDeclaration(
          moduleTranslation.getLLVMModule(), 
          llvm::Intrinsic::assume);
      builder.CreateCall(assumeFn, {isAligned});
    }
  }
}
```

**Test**: `mlir/test/Target/LLVMIR/openmp-simd.mlir`

```mlir
// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @simd_simple
llvm.func @simd_simple(%lb: i32, %ub: i32, %step: i32) {
  // CHECK: br label %simd.header
  omp.simd for (%iv) : i32 = (%lb) to (%ub) step (%step) {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    omp.yield
  }
  // CHECK: simd.header:
  // CHECK: %[[IV:.*]] = phi i32
  // CHECK: %[[COND:.*]] = icmp sle i32 %[[IV]], %{{.*}}
  // CHECK: br i1 %[[COND]], label %simd.body, label %simd.exit
  
  // CHECK: simd.body:
  // CHECK: br label %simd.latch
  
  // CHECK: simd.latch:
  // CHECK: %[[NEXT:.*]] = add i32 %[[IV]], %{{.*}}
  // CHECK: br label %simd.header, !llvm.loop [[LOOP_MD:!.*]]
  
  llvm.return
}

// CHECK-LABEL: define void @simd_with_simdlen
llvm.func @simd_with_simdlen(%lb: i32, %ub: i32, %step: i32) {
  // CHECK: br label %simd.header
  omp.simd simdlen(8) for (%iv) : i32 = (%lb) to (%ub) step (%step) {
    omp.yield
  }
  // CHECK: br label %simd.header, !llvm.loop [[LOOP_SIMDLEN:!.*]]
  llvm.return
}

// Verify loop metadata
// CHECK: [[LOOP_MD]] = distinct !{[[LOOP_MD]], [[VEC_ENABLE:!.*]]}
// CHECK: [[VEC_ENABLE]] = !{!"llvm.loop.vectorize.enable", i1 true}

// CHECK: [[LOOP_SIMDLEN]] = distinct !{[[LOOP_SIMDLEN]], [[VEC_ENABLE]], [[VEC_WIDTH:!.*]]}
// CHECK: [[VEC_WIDTH]] = !{!"llvm.loop.vectorize.width", i32 8}
```

---

### Key Patterns for Loop-Based Constructs

**1. Loop Structure Recognition**:
- Parse loop directive + DO construct together
- Extract loop control (variable, bounds, step)
- Validate canonical form

**2. Iteration Space Handling**:
- Convert Fortran bounds to MLIR values
- Handle default step (1 if not specified)
- Create block argument for induction variable

**3. Clause Processing**:
- SIMDLEN/SAFELEN: Scalar integer constants
- ALIGNED: Variable list + optional alignment values
- LINEAR: Variable list + optional step expression
- Standard data-sharing clauses apply

**4. MLIR Representation**:
- Loop operation with bounds operands
- Single block with IV as argument
- YieldOp terminator

**5. LLVM Translation**:
- Generate PHI node for induction variable
- Attach vectorization metadata
- Add alignment assumptions for ALIGNED clause

**Common Loop Directives Using This Pattern**:
- `SIMD` - Vectorization hints
- `DO` - Worksharing loop parallelization
- `DISTRIBUTE` - Loop distribution across teams
- `TASKLOOP` - Loop iterations as tasks
- Combined constructs: `DO SIMD`, `DISTRIBUTE SIMD`, etc.

---

## IMPLEMENTATION WALKTHROUGH: COPYIN Clause for PARALLEL Directive

This section provides a **detailed walkthrough** of implementing the `COPYIN` clause with actual file paths, line-by-line code additions, and complete examples. Follow this guide to understand the exact workflow for adding a data-transfer clause.

### Overview: What is COPYIN?

**OpenMP Specification**: OpenMP 5.2 Section 2.21.4.2  
**Purpose**: Copy the value of a THREADPRIVATE variable from the master thread to all other threads in a team  
**Applicable to**: PARALLEL and combined parallel constructs  
**Restriction**: Variables must have THREADPRIVATE attribute  

**Fortran Example**:
```fortran
module shared_data
  integer :: global_counter
  !$omp threadprivate(global_counter)
end module

program main
  use shared_data
  
  global_counter = 100  ! Set in master thread
  
  !$omp parallel copyin(global_counter)
    ! All threads now have global_counter = 100
    print *, "Thread counter:", global_counter
  !$omp end parallel
end program
```

**Why COPYIN is needed**: Threadprivate variables maintain separate storage per thread. Without COPYIN, each thread's copy has undefined initial value. COPYIN broadcasts the master thread's value to all threads.

---

### Step 1: Add Clause to Parser

**File**: `llvm-project/llvm/include/llvm/Frontend/OpenMP/OMPKinds.def`  
**Location**: Around line 480-520 (clause definitions section)

**Add this line**:
```cpp
__OMP_CLAUSE(copyin, OMPC_copyin)
```

**Context** (existing code around it):
```cpp
__OMP_CLAUSE(copyprivate, OMPC_copyprivate)
__OMP_CLAUSE(copyin, OMPC_copyin)        // ← ADD THIS LINE
__OMP_CLAUSE(default, OMPC_default)
```

---

**File**: `llvm-project/flang/include/flang/Parser/parse-tree.h`  
**Location**: Around line 3700-3800 (OmpClause definitions)

**Add this structure**:
```cpp
// COPYIN clause - takes a list of threadprivate variables
struct OmpClause::Copyin {
  WRAPPER_CLASS_BOILERPLATE(OmpClause::Copyin, OmpObjectList);
};
```

**Context** (surrounding code):
```cpp
struct OmpClause::Copyprivate {
  WRAPPER_CLASS_BOILERPLATE(OmpClause::Copyprivate, OmpObjectList);
};

struct OmpClause::Copyin {                          // ← ADD THIS
  WRAPPER_CLASS_BOILERPLATE(OmpClause::Copyin, OmpObjectList);
};

struct OmpClause::Default {
  WRAPPER_CLASS_BOILERPLATE(OmpClause::Default, OmpDefaultClause);
};
```

**Explanation**: `WRAPPER_CLASS_BOILERPLATE` creates a simple wrapper around `OmpObjectList` since COPYIN just takes a variable list with no modifiers.

---

**File**: `llvm-project/flang/lib/Parser/openmp-parsers.cpp`  
**Location**: Around line 450-500 (clause parsers section)

**Add this parser**:
```cpp
// Parse COPYIN clause: COPYIN(list)
TYPE_PARSER(construct<OmpClause::Copyin>(
    "COPYIN" >> parenthesized(Parser<OmpObjectList>{})))
```

**Then add to main clause parser** (around line 650):
```cpp
TYPE_PARSER(
    construct<OmpClause>(Parser<OmpClause::Allocate>{}) ||
    // ... other clauses ...
    construct<OmpClause>(Parser<OmpClause::Copyin>{}) ||     // ← ADD THIS LINE
    construct<OmpClause>(Parser<OmpClause::Copyprivate>{}) ||
    // ... more clauses ...
)
```

**Context**:
```cpp
TYPE_PARSER(
    construct<OmpClause>(Parser<OmpClause::Collapse>{}) ||
    construct<OmpClause>(Parser<OmpClause::Copyin>{}) ||        // ← ADD THIS
    construct<OmpClause>(Parser<OmpClause::Copyprivate>{}) ||
    construct<OmpClause>(Parser<OmpClause::Default>{}) ||
)
```

---

**File**: `llvm-project/flang/test/Parser/OpenMP/copyin-clause.f90`  
**Action**: Create new test file

**Complete test code**:
```fortran
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck %s
! Test COPYIN clause parsing

module test_mod
  integer :: shared_var
  real :: shared_array(100)
  !$omp threadprivate(shared_var, shared_array)
end module

program test_copyin_parsing
  use test_mod
  implicit none
  integer :: i
  
  shared_var = 42
  
  ! CHECK: OpenMPBlockConstruct
  ! CHECK: OmpBeginBlockDirective
  ! CHECK: OmpBlockDirective -> llvm::omp::Directive = parallel
  ! CHECK: OmpClauseList -> OmpClause -> Copyin
  ! CHECK: OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'shared_var'
  !$omp parallel copyin(shared_var)
    print *, "Var:", shared_var
  !$omp end parallel
  
  ! Test multiple variables
  ! CHECK: OmpClause -> Copyin
  ! CHECK: OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'shared_var'
  ! CHECK: OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'shared_array'
  !$omp parallel copyin(shared_var, shared_array)
    do i = 1, 10
      shared_array(i) = shared_var + i
    end do
  !$omp end parallel

end program
```

**Run test**:
```bash
cd llvm-project/build
./bin/llvm-lit -v ../flang/test/Parser/OpenMP/copyin-clause.f90
```

**Expected output**: `PASS: Flang :: Parser/OpenMP/copyin-clause.f90`

---

### Step 2: Add Semantic Validation

**File**: `llvm-project/flang/lib/Semantics/check-omp-structure.cpp`  
**Location**: Around line 1850-1900 (SetClauseSets function)

**Add COPYIN to PARALLEL's allowed clauses**:
```cpp
void OmpStructureChecker::SetClauseSets(llvm::omp::Directive dir) {
  switch (dir) {
    case llvm::omp::Directive::OMPD_parallel:
      allowedClauses_ = {
        llvm::omp::Clause::OMPC_default,
        llvm::omp::Clause::OMPC_private,
        llvm::omp::Clause::OMPC_firstprivate,
        llvm::omp::Clause::OMPC_shared,
        llvm::omp::Clause::OMPC_copyin,           // ← ADD THIS LINE
        llvm::omp::Clause::OMPC_reduction,
        llvm::omp::Clause::OMPC_if,
        llvm::omp::Clause::OMPC_num_threads,
        llvm::omp::Clause::OMPC_proc_bind,
      };
      break;
    // ... other directives ...
  }
}
```

---

**File**: `llvm-project/flang/lib/Semantics/check-omp-structure.cpp`  
**Location**: Around line 2600-2700 (clause validation section)

**Add COPYIN validation function**:
```cpp
void OmpStructureChecker::Enter(const parser::OmpClause::Copyin &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_copyin);
  
  // COPYIN only allowed on PARALLEL and combined parallel constructs
  const auto currentDir = GetContext().directive;
  if (currentDir != llvm::omp::Directive::OMPD_parallel &&
      currentDir != llvm::omp::Directive::OMPD_parallel_do &&
      currentDir != llvm::omp::Directive::OMPD_parallel_sections &&
      currentDir != llvm::omp::Directive::OMPD_parallel_workshare) {
    context_.Say(GetContext().clauseSource,
        "COPYIN clause is only allowed on PARALLEL constructs"_err_en_US);
  }
  
  // Validate each variable in the list
  for (const auto &ompObject : x.v.v) {
    if (const auto *name = parser::Unwrap<parser::Name>(ompObject)) {
      if (const auto *symbol = name->symbol) {
        
        // Must have THREADPRIVATE attribute
        if (!IsThreadprivate(*symbol)) {
          context_.Say(name->source,
              "Variable '%s' in COPYIN clause must be THREADPRIVATE"_err_en_US,
              name->ToString());
          continue;
        }
        
        // Cannot be POINTER
        if (IsPointer(*symbol)) {
          context_.Say(name->source,
              "POINTER variable '%s' is not allowed in COPYIN clause"_err_en_US,
              name->ToString());
        }
        
        // Cannot be ALLOCATABLE
        if (IsAllocatable(*symbol)) {
          context_.Say(name->source,
              "ALLOCATABLE variable '%s' is not allowed in COPYIN clause"_err_en_US,
              name->ToString());
        }
        
        // Check for conflicts with other data-sharing clauses
        // COPYIN cannot appear with PRIVATE, FIRSTPRIVATE for same variable
        if (HasDataSharingAttribute(*symbol, llvm::omp::Clause::OMPC_private)) {
          context_.Say(name->source,
              "Variable '%s' cannot appear in both COPYIN and PRIVATE clauses"_err_en_US,
              name->ToString());
        }
        
        if (HasDataSharingAttribute(*symbol, llvm::omp::Clause::OMPC_firstprivate)) {
          context_.Say(name->source,
              "Variable '%s' cannot appear in both COPYIN and FIRSTPRIVATE clauses"_err_en_US,
              name->ToString());
        }
        
        // Track that this symbol has COPYIN attribute
        SetDataSharingAttribute(*symbol, llvm::omp::Clause::OMPC_copyin);
      }
    }
  }
}
```

**Helper function** (may need to add):
```cpp
// Check if symbol has THREADPRIVATE attribute
bool OmpStructureChecker::IsThreadprivate(const Symbol &symbol) const {
  // Check if symbol has threadprivate common block or directive
  if (const auto *commonBlock = FindCommonBlockContaining(symbol)) {
    return HasOmpThreadprivate(*commonBlock);
  }
  
  // Check for explicit threadprivate directive
  return symbol.test(Symbol::Flag::OmpThreadprivate);
}
```

---

**File**: `llvm-project/flang/test/Semantics/OpenMP/copyin-errors.f90`  
**Action**: Create new test file

**Complete test code**:
```fortran
! RUN: %python %S/test_errors.py %s %flang_fc1 -fopenmp
! Test semantic errors for COPYIN clause

module copyin_test_mod
  integer :: tpvar
  integer :: normal_var
  integer, pointer :: ptr_var
  integer, allocatable :: alloc_var
  
  !$omp threadprivate(tpvar)
  ! normal_var is NOT threadprivate
end module

program test_copyin_errors
  use copyin_test_mod
  implicit none
  
  tpvar = 100
  normal_var = 200
  
  ! Valid: COPYIN with threadprivate variable
  !$omp parallel copyin(tpvar)
    print *, tpvar
  !$omp end parallel
  
  ! ERROR: Variable not threadprivate
  !ERROR: Variable 'normal_var' in COPYIN clause must be THREADPRIVATE
  !$omp parallel copyin(normal_var)
    print *, normal_var
  !$omp end parallel
  
  ! ERROR: COPYIN on DO directive (only allowed on PARALLEL)
  !ERROR: COPYIN clause is only allowed on PARALLEL constructs
  !$omp do copyin(tpvar)
  do i = 1, 10
    print *, i
  end do
  
  ! ERROR: Conflict with PRIVATE
  !ERROR: Variable 'tpvar' cannot appear in both COPYIN and PRIVATE clauses
  !$omp parallel copyin(tpvar) private(tpvar)
    tpvar = 1
  !$omp end parallel
  
  ! ERROR: Conflict with FIRSTPRIVATE
  !ERROR: Variable 'tpvar' cannot appear in both COPYIN and FIRSTPRIVATE clauses
  !$omp parallel copyin(tpvar) firstprivate(tpvar)
    tpvar = 2
  !$omp end parallel
  
  ! Valid: Multiple threadprivate variables
  !$omp threadprivate(ptr_var, alloc_var)
  
  ! ERROR: POINTER not allowed
  !ERROR: POINTER variable 'ptr_var' is not allowed in COPYIN clause
  !$omp parallel copyin(ptr_var)
    print *, ptr_var
  !$omp end parallel
  
  ! ERROR: ALLOCATABLE not allowed
  !ERROR: ALLOCATABLE variable 'alloc_var' is not allowed in COPYIN clause
  !$omp parallel copyin(alloc_var)
    print *, alloc_var
  !$omp end parallel

end program
```

**Run test**:
```bash
./bin/llvm-lit -v ../flang/test/Semantics/OpenMP/copyin-errors.f90
```

---

### Step 3: Add MLIR Dialect Support

**File**: `llvm-project/mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td`  
**Location**: Around line 400-500 (ParallelOp definition)

**Add copyin_vars operand**:
```tablegen
def ParallelOp : OpenMP_Op<"parallel", [AttrSizedOperandSegments]> {
  let summary = "parallel construct";
  let description = [{
    The parallel construct creates a team of threads that execute the region.
    
    OpenMP 5.2 Section 2.6
  }];

  let arguments = (ins
    Optional<I1>:$if_expr,
    Optional<I32>:$num_threads,
    Variadic<AnyType>:$private_vars,
    Variadic<AnyType>:$firstprivate_vars,
    Variadic<AnyType>:$shared_vars,
    Variadic<AnyType>:$copyin_vars,          // ← ADD THIS LINE
    Variadic<AnyType>:$reduction_vars,
    OptionalAttr<ProcBindKindAttr>:$proc_bind,
    OptionalAttr<ArrayAttr>:$reduction_syms
  );

  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    oilist(
      `if` `(` $if_expr `)`
      | `num_threads` `(` $num_threads `)`
      | `private` `(` $private_vars `:` type($private_vars) `)`
      | `firstprivate` `(` $firstprivate_vars `:` type($firstprivate_vars) `)`
      | `shared` `(` $shared_vars `:` type($shared_vars) `)`
      | `copyin` `(` $copyin_vars `:` type($copyin_vars) `)`    // ← ADD THIS LINE
      | `reduction` custom<ReductionClause>($reduction_syms, $reduction_vars)
      | `proc_bind` `(` $proc_bind `)`
    )
    $region attr-dict
  }];

  let hasVerifier = 1;
}
```

**Note**: The `AttrSizedOperandSegments` trait automatically handles the variable number of operands.

---

**File**: `llvm-project/mlir/test/Dialect/OpenMP/copyin.mlir`  
**Action**: Create new test file

**Complete test code**:
```mlir
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Test COPYIN clause in MLIR OpenMP dialect

// CHECK-LABEL: func @parallel_copyin_simple
func.func @parallel_copyin_simple() {
  %tpvar = fir.address_of(@_QFEtpvar) : !fir.ref<i32>
  
  // CHECK: omp.parallel copyin(%{{.*}} : !fir.ref<i32>)
  omp.parallel copyin(%tpvar : !fir.ref<i32>) {
    %val = fir.load %tpvar : !fir.ref<i32>
    omp.terminator
  }
  return
}

// CHECK-LABEL: func @parallel_copyin_multiple
func.func @parallel_copyin_multiple() {
  %tpvar1 = fir.address_of(@_QFEtpvar1) : !fir.ref<i32>
  %tpvar2 = fir.address_of(@_QFEtpvar2) : !fir.ref<f32>
  
  // CHECK: omp.parallel copyin(%{{.*}}, %{{.*}} : !fir.ref<i32>, !fir.ref<f32>)
  omp.parallel copyin(%tpvar1, %tpvar2 : !fir.ref<i32>, !fir.ref<f32>) {
    %val1 = fir.load %tpvar1 : !fir.ref<i32>
    %val2 = fir.load %tpvar2 : !fir.ref<f32>
    omp.terminator
  }
  return
}

// CHECK-LABEL: func @parallel_copyin_with_private
func.func @parallel_copyin_with_private() {
  %tpvar = fir.address_of(@_QFEtpvar) : !fir.ref<i32>
  %priv = fir.address_of(@_QFEpriv) : !fir.ref<i32>
  
  // CHECK: omp.parallel private(%{{.*}} : !fir.ref<i32>) copyin(%{{.*}} : !fir.ref<i32>)
  omp.parallel private(%priv : !fir.ref<i32>) copyin(%tpvar : !fir.ref<i32>) {
    omp.terminator
  }
  return
}

// Declare threadprivate global variables
fir.global @_QFEtpvar : i32 {
  %c0 = arith.constant 0 : i32
  fir.has_value %c0 : i32
}

fir.global @_QFEtpvar1 : i32 {
  %c0 = arith.constant 0 : i32
  fir.has_value %c0 : i32
}

fir.global @_QFEtpvar2 : f32 {
  %c0 = arith.constant 0.0 : f32
  fir.has_value %c0 : f32
}

fir.global @_QFEpriv : i32 {
  %c0 = arith.constant 0 : i32
  fir.has_value %c0 : i32
}
```

**Run test**:
```bash
./bin/llvm-lit -v ../mlir/test/Dialect/OpenMP/copyin.mlir
```

---

### Step 4: Implement Lowering

**File**: `llvm-project/flang/lib/Lower/OpenMP/Clauses.h`  
**Location**: Around line 150-200 (ClauseProcessor class)

**Add method declaration**:
```cpp
class ClauseProcessor {
public:
  // ... existing methods ...
  
  void processCopyin(
      llvm::SmallVectorImpl<mlir::Value> &copyinVars) const;
  
  // ... more methods ...
};
```

---

**File**: `llvm-project/flang/lib/Lower/OpenMP/Clauses.cpp`  
**Location**: Around line 1200-1300 (clause processing implementations)

**Add implementation**:
```cpp
void ClauseProcessor::processCopyin(
    llvm::SmallVectorImpl<mlir::Value> &copyinVars) const {
  
  const parser::CharBlock *source = nullptr;
  auto copyinClauses = 
      findRepeatableClause<parser::OmpClause::Copyin>(source);
  
  for (const auto *clause : copyinClauses) {
    // Get the object list
    const auto &objectList = clause->v;
    
    for (const auto &ompObject : objectList.v) {
      if (const auto *name = parser::Unwrap<parser::Name>(ompObject)) {
        if (auto *symbol = name->symbol) {
          
          // Get the address of the threadprivate variable
          // For threadprivate, use special addressing
          mlir::Value copyinVar;
          
          if (IsThreadprivate(*symbol)) {
            // Get threadprivate address (may involve TLS lookup)
            copyinVar = converter.getSymbolAddress(*symbol);
          } else {
            // This shouldn't happen if semantic checks passed
            continue;
          }
          
          copyinVars.push_back(copyinVar);
        }
      }
    }
  }
}

// Helper to check threadprivate attribute
bool ClauseProcessor::IsThreadprivate(const semantics::Symbol &symbol) const {
  return symbol.test(semantics::Symbol::Flag::OmpThreadprivate);
}
```

---

**File**: `llvm-project/flang/lib/Lower/OpenMP/OpenMP.cpp`  
**Location**: Around line 2000-2100 (genParallelOp function)

**Add COPYIN processing**:
```cpp
static mlir::omp::ParallelOp genParallelOp(
    lower::AbstractConverter &converter,
    lower::SymMap &symTable,
    semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval,
    mlir::Location loc,
    const ConstructQueue &queue,
    const parser::OmpClauseList &clauseList) {
  
  ClauseProcessor cp(converter, semaCtx, clauseList);
  
  // Process all clauses
  mlir::Value ifClauseOperand, numThreadsClauseOperand;
  llvm::SmallVector<mlir::Value> privateVars, firstprivateVars;
  llvm::SmallVector<mlir::Value> sharedVars, copyinVars;      // ← ADD copyinVars
  llvm::SmallVector<mlir::Value> reductionVars;
  llvm::SmallVector<mlir::Attribute> reductionSyms;
  mlir::omp::ClauseProcBindKindAttr procBindAttr;
  
  cp.processIf(llvm::omp::Directive::OMPD_parallel, ifClauseOperand);
  cp.processNumThreads(numThreadsClauseOperand);
  cp.processPrivate(privateVars);
  cp.processFirstprivate(firstprivateVars);
  cp.processShared(sharedVars);
  cp.processCopyin(copyinVars);                               // ← ADD THIS
  cp.processReduction(loc, reductionVars, reductionSyms);
  cp.processProcBind(procBindAttr);
  
  // Create parallel operation
  auto &builder = converter.getFirOpBuilder();
  auto parallelOp = builder.create<mlir::omp::ParallelOp>(
      loc,
      ifClauseOperand,
      numThreadsClauseOperand,
      privateVars,
      firstprivateVars,
      sharedVars,
      copyinVars,                                             // ← ADD THIS
      reductionVars,
      procBindAttr,
      reductionSyms.empty() ? nullptr : builder.getArrayAttr(reductionSyms)
  );
  
  // Create region and lower body
  auto &block = parallelOp.getRegion().emplaceBlock();
  auto insertPt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(&block);
  
  genNestedEvaluations(converter, eval);
  
  builder.create<mlir::omp::TerminatorOp>(loc);
  builder.restoreInsertionPoint(insertPt);
  
  return parallelOp;
}
```

---

**File**: `llvm-project/flang/test/Lower/OpenMP/copyin-lowering.f90`  
**Action**: Create new test file

**Complete test code**:
```fortran
! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s
! Test lowering of COPYIN clause

module copyin_mod
  integer :: shared_counter
  real :: shared_value
  !$omp threadprivate(shared_counter, shared_value)
end module

program test_copyin_lowering
  use copyin_mod
  implicit none
  
  shared_counter = 100
  shared_value = 3.14
  
  ! CHECK-LABEL: func @_QQmain
  
  ! Simple COPYIN
  ! CHECK: %[[COUNTER_ADDR:.*]] = fir.address_of(@_QMcopyin_modEshared_counter)
  ! CHECK: omp.parallel copyin(%[[COUNTER_ADDR]] : !fir.ref<i32>)
  !$omp parallel copyin(shared_counter)
    print *, "Counter:", shared_counter
  !$omp end parallel
  
  ! Multiple COPYIN variables
  ! CHECK: %[[COUNTER:.*]] = fir.address_of(@_QMcopyin_modEshared_counter)
  ! CHECK: %[[VALUE:.*]] = fir.address_of(@_QMcopyin_modEshared_value)
  ! CHECK: omp.parallel copyin(%[[COUNTER]], %[[VALUE]] : !fir.ref<i32>, !fir.ref<f32>)
  !$omp parallel copyin(shared_counter, shared_value)
    shared_counter = shared_counter + 1
    shared_value = shared_value * 2.0
  !$omp end parallel
  
  ! COPYIN with other clauses
  ! CHECK: %[[PRIV:.*]] = fir.alloca i32
  ! CHECK: omp.parallel private(%[[PRIV]] : !fir.ref<i32>) copyin(%[[COUNTER]] : !fir.ref<i32>)
  !$omp parallel private(i) copyin(shared_counter)
    do i = 1, 10
      print *, shared_counter + i
    end do
  !$omp end parallel

end program

! CHECK: fir.global @_QMcopyin_modEshared_counter : i32
! CHECK: fir.global @_QMcopyin_modEshared_value : f32
```

**Run test**:
```bash
./bin/llvm-lit -v ../flang/test/Lower/OpenMP/copyin-lowering.f90
```

---

### Step 5: LLVM IR Translation

**File**: `llvm-project/mlir/lib/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.cpp`  
**Location**: Around line 900-1000 (convertOmpParallelOp function)

**Add COPYIN handling**:
```cpp
static LogicalResult convertOmpParallelOp(
    omp::ParallelOp parallelOp,
    llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) {
  
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  
  // Extract clause operands
  llvm::Value *ifCondition = nullptr;
  if (parallelOp.getIfExpr()) {
    ifCondition = moduleTranslation.lookupValue(parallelOp.getIfExpr());
  }
  
  llvm::Value *numThreads = nullptr;
  if (parallelOp.getNumThreads()) {
    numThreads = moduleTranslation.lookupValue(parallelOp.getNumThreads());
  }
  
  // Process COPYIN variables
  SmallVector<llvm::Value*> copyinVars;
  for (auto copyinVar : parallelOp.getCopyinVars()) {
    llvm::Value *copyinVarLLVM = moduleTranslation.lookupValue(copyinVar);
    copyinVars.push_back(copyinVarLLVM);
  }
  
  // Create parallel body callback
  auto bodyCB = [&](InsertPointTy allocaIP, InsertPointTy codegenIP) {
    builder.restoreIP(codegenIP);
    
    // Generate COPYIN initialization code
    if (!copyinVars.empty()) {
      generateCopyinCode(copyinVars, builder, ompBuilder, moduleTranslation);
    }
    
    // Translate parallel region body
    auto &region = parallelOp.getRegion();
    auto &block = region.front();
    
    for (auto &op : block.without_terminator()) {
      if (failed(moduleTranslation.convertOperation(op, builder)))
        return llvm::Error::success();
    }
    
    return llvm::Error::success();
  };
  
  // Create parallel region
  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);
  
  auto result = ompBuilder->createParallel(
      builder.saveIP(),
      allocaIP,
      bodyCB,
      /*PrivCB=*/nullptr,
      /*FiniCB=*/nullptr,
      ifCondition,
      numThreads,
      llvm::omp::ProcBindKind::OMP_PROC_BIND_default,
      /*IsCancellable=*/false);
  
  builder.restoreIP(result);
  return success();
}

// Helper: Generate COPYIN runtime calls
static void generateCopyinCode(
    ArrayRef<llvm::Value*> copyinVars,
    llvm::IRBuilderBase &builder,
    llvm::OpenMPIRBuilder *ompBuilder,
    LLVM::ModuleTranslation &moduleTranslation) {
  
  if (copyinVars.empty()) return;
  
  // Get runtime location and thread ID
  llvm::Value *loc = ompBuilder->getOrCreateIdent(
      builder.getCurrentDebugLocation());
  llvm::Value *globalTid = ompBuilder->getOrCreateThreadID(loc);
  
  // Call __kmpc_copyprivate for each variable
  // Runtime signature:
  // void __kmpc_copyprivate(ident_t *loc, int32_t gtid, size_t cpy_size,
  //                         void *cpy_data, void (*cpy_func)(void*, void*),
  //                         int32_t didit);
  
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::FunctionCallee copyinFn = 
      ompBuilder->getOrCreateRuntimeFunction(
          module, llvm::omp::OMPRTL___kmpc_copyprivate);
  
  for (llvm::Value *copyinVar : copyinVars) {
    // Get size of the variable type
    llvm::Type *varType = copyinVar->getType()->getPointerElementType();
    llvm::DataLayout DL(module);
    uint64_t varSize = DL.getTypeStoreSize(varType);
    
    // Create copy function (simple memcpy wrapper)
    llvm::Function *copyFunc = createCopyFunction(varType, module, builder);
    
    // Master thread: didit = 1, others: didit = 0
    // Master thread broadcasts value to all threads
    llvm::Value *isMaster = builder.CreateICmpEQ(
        globalTid, 
        builder.getInt32(0),
        "is_master");
    llvm::Value *didit = builder.CreateZExt(isMaster, builder.getInt32Ty());
    
    // Call copyprivate runtime
    builder.CreateCall(copyinFn, {
        loc,
        globalTid,
        builder.getInt64(varSize),
        builder.CreateBitCast(copyinVar, builder.getInt8PtrTy()),
        copyFunc,
        didit
    });
  }
}

// Create simple copy function for copyprivate
static llvm::Function* createCopyFunction(
    llvm::Type *varType,
    llvm::Module *module,
    llvm::IRBuilderBase &builder) {
  
  llvm::LLVMContext &ctx = module->getContext();
  
  // Function signature: void copy_func(void* dst, void* src)
  llvm::Type *voidPtrTy = builder.getInt8PtrTy();
  llvm::FunctionType *copyFnTy = llvm::FunctionType::get(
      builder.getVoidTy(),
      {voidPtrTy, voidPtrTy},
      /*isVarArg=*/false);
  
  llvm::Function *copyFn = llvm::Function::Create(
      copyFnTy,
      llvm::GlobalValue::InternalLinkage,
      ".omp.copyin.func",
      module);
  
  // Create function body
  llvm::BasicBlock *entryBB = 
      llvm::BasicBlock::Create(ctx, "entry", copyFn);
  llvm::IRBuilder<> funcBuilder(entryBB);
  
  llvm::Value *dst = copyFn->getArg(0);
  llvm::Value *src = copyFn->getArg(1);
  
  // Cast to proper type and perform copy
  llvm::Type *varPtrTy = varType->getPointerTo();
  llvm::Value *typedDst = funcBuilder.CreateBitCast(dst, varPtrTy);
  llvm::Value *typedSrc = funcBuilder.CreateBitCast(src, varPtrTy);
  
  llvm::Value *value = funcBuilder.CreateLoad(varType, typedSrc);
  funcBuilder.CreateStore(value, typedDst);
  
  funcBuilder.CreateRetVoid();
  
  return copyFn;
}
```

---

**File**: `llvm-project/mlir/test/Target/LLVMIR/openmp-copyin.mlir`  
**Action**: Create new test file

**Complete test code**:
```mlir
// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @test_copyin() {
  %tpvar = llvm.mlir.addressof @tpvar : !llvm.ptr<i32>
  
  // CHECK: call void @__kmpc_fork_call
  // CHECK-SAME: @[[OUTLINED:.*]]
  omp.parallel copyin(%tpvar : !llvm.ptr<i32>) {
    %val = llvm.load %tpvar : !llvm.ptr<i32>
    omp.terminator
  }
  
  llvm.return
}

// CHECK: define internal void @[[OUTLINED]]
// CHECK: %[[LOC:.*]] = call {{.*}} @__kmpc_global_thread_num
// CHECK: %[[IS_MASTER:.*]] = icmp eq i32 %[[LOC]], 0
// CHECK: %[[DIDIT:.*]] = zext i1 %[[IS_MASTER]] to i32
// CHECK: call void @__kmpc_copyprivate(
// CHECK-SAME: i32 %[[LOC]],
// CHECK-SAME: i64 4,
// CHECK-SAME: i8* {{.*}},
// CHECK-SAME: void (i8*, i8*)* @.omp.copyin.func,
// CHECK-SAME: i32 %[[DIDIT]])

// CHECK: define internal void @.omp.copyin.func(i8* %{{.*}}, i8* %{{.*}})
// CHECK: %[[SRC:.*]] = bitcast i8* %{{.*}} to i32*
// CHECK: %[[DST:.*]] = bitcast i8* %{{.*}} to i32*
// CHECK: %[[VAL:.*]] = load i32, i32* %[[SRC]]
// CHECK: store i32 %[[VAL]], i32* %[[DST]]

llvm.mlir.global internal @tpvar() : i32 {
  %c0 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %c0 : i32
}
```

**Run test**:
```bash
./bin/llvm-lit -v ../mlir/test/Target/LLVMIR/openmp-copyin.mlir
```

---

### Step 6: Build and Test

**Complete build process**:

```bash
# Navigate to LLVM build directory
cd llvm-project/build

# Rebuild affected components
ninja flang-new

# Run all COPYIN tests
./bin/llvm-lit -v ../flang/test/Parser/OpenMP/copyin-clause.f90
./bin/llvm-lit -v ../flang/test/Semantics/OpenMP/copyin-errors.f90
./bin/llvm-lit -v ../flang/test/Lower/OpenMP/copyin-lowering.f90
./bin/llvm-lit -v ../mlir/test/Dialect/OpenMP/copyin.mlir
./bin/llvm-lit -v ../mlir/test/Target/LLVMIR/openmp-copyin.mlir

# Or run all OpenMP tests
./bin/llvm-lit -v ../flang/test/Parser/OpenMP/
./bin/llvm-lit -v ../flang/test/Semantics/OpenMP/
./bin/llvm-lit -v ../flang/test/Lower/OpenMP/
```

---

### Step 7: End-to-End Testing

**File**: `test_copyin_complete.f90` (create locally)

```fortran
! Complete test program
module global_data
  integer :: thread_id
  real :: computation_result(10)
  !$omp threadprivate(thread_id, computation_result)
end module

program test_copyin_complete
  use global_data
  use omp_lib
  implicit none
  integer :: i
  
  ! Initialize in master thread
  thread_id = 999
  computation_result = 0.0
  
  print *, "Before parallel region:"
  print *, "  thread_id =", thread_id
  
  !$omp parallel copyin(thread_id, computation_result)
    ! All threads now have thread_id = 999
    ! Update with actual thread number
    thread_id = omp_get_thread_num()
    
    ! Each thread computes
    do i = 1, 10
      computation_result(i) = thread_id * i
    end do
    
    !$omp critical
    print *, "Thread", thread_id, "result(5) =", computation_result(5)
    !$omp end critical
  !$omp end parallel
  
  print *, "After parallel region:"
  print *, "  master thread_id =", thread_id

end program
```

**Compile and run**:
```bash
# Compile with flang
./bin/flang-new -fopenmp test_copyin_complete.f90 -o test_copyin

# Run
./test_copyin
```

**Expected output**:
```
Before parallel region:
  thread_id = 999
Thread 0 result(5) = 0.0
Thread 1 result(5) = 5.0
Thread 2 result(5) = 10.0
Thread 3 result(5) = 15.0
After parallel region:
  master thread_id = 0
```

---

### Summary: Files Modified for COPYIN

| **Stage** | **File Path** | **Lines Modified** | **Change Description** |
|-----------|---------------|-------------------|------------------------|
| **1. Parser** | `llvm/include/llvm/Frontend/OpenMP/OMPKinds.def` | ~490 | Add `__OMP_CLAUSE(copyin, OMPC_copyin)` |
| | `flang/include/flang/Parser/parse-tree.h` | ~3750 | Add `OmpClause::Copyin` structure |
| | `flang/lib/Parser/openmp-parsers.cpp` | ~480, ~660 | Add COPYIN parser, add to clause list |
| | `flang/test/Parser/OpenMP/copyin-clause.f90` | New file | Parser test cases |
| **2. Semantics** | `flang/lib/Semantics/check-omp-structure.cpp` | ~1870, ~2650 | Add to allowed clauses, add validation |
| | `flang/test/Semantics/OpenMP/copyin-errors.f90` | New file | Semantic error tests |
| **3. MLIR** | `mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td` | ~450 | Add copyin_vars operand to ParallelOp |
| | `mlir/test/Dialect/OpenMP/copyin.mlir` | New file | MLIR dialect tests |
| **4. Lowering** | `flang/lib/Lower/OpenMP/Clauses.h` | ~170 | Add processCopyin declaration |
| | `flang/lib/Lower/OpenMP/Clauses.cpp` | ~1250 | Implement processCopyin |
| | `flang/lib/Lower/OpenMP/OpenMP.cpp` | ~2050 | Add COPYIN to genParallelOp |
| | `flang/test/Lower/OpenMP/copyin-lowering.f90` | New file | Lowering tests |
| **5. Translation** | `mlir/lib/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.cpp` | ~950 | Add generateCopyinCode, runtime calls |
| | `mlir/test/Target/LLVMIR/openmp-copyin.mlir` | New file | LLVM IR translation tests |

**Total**: 10 files modified + 5 new test files = **15 files**

---

### Key Implementation Insights

**1. THREADPRIVATE Dependency**:
- COPYIN only works with threadprivate variables
- Requires checking `Symbol::Flag::OmpThreadprivate` attribute
- Each thread has separate storage for threadprivate data

**2. Runtime Implementation**:
- Uses `__kmpc_copyprivate` runtime function
- Master thread broadcasts value to all threads
- Requires creating a copy function for each variable type

**3. Clause Processing Pattern**:
- Similar to PRIVATE/SHARED (variable list clause)
- Simpler than REDUCTION (no operator, no combiner)
- Must validate threadprivate attribute in semantics

**4. Testing Strategy**:
- Parser: Verify clause recognition and object list parsing
- Semantics: Test restrictions (threadprivate requirement, conflicts)
- Lowering: Check MLIR operands are correctly populated
- Translation: Verify runtime calls are generated

**5. Common Pitfalls**:
- Forgetting to add clause to `allowedClauses_` for PARALLEL
- Not handling threadprivate addressing correctly in lowering
- Missing COPYIN in `AttrSizedOperandSegments` count

---

## TASKLOOP CLAUSES IMPLEMENTATION NOTES

This section provides detailed implementation guidance for TASKLOOP-specific clauses: **GRAINSIZE**, **NUM_TASKS**, and **NOGROUP**. These clauses control how loop iterations are divided into tasks and how task synchronization is handled.

### Overview: TASKLOOP and Its Clauses

**OpenMP Specification**: OpenMP 5.2 Section 2.12.2  
**TASKLOOP Directive**: Creates tasks to execute loop iterations

**Key Clauses**:
- **GRAINSIZE**: Specifies the number of loop iterations per task
- **NUM_TASKS**: Specifies the number of tasks to create
- **NOGROUP**: Removes the implicit taskgroup around the taskloop

**Mutual Exclusivity**: GRAINSIZE and NUM_TASKS are mutually exclusive

**Fortran Example**:
```fortran
! Using GRAINSIZE - each task processes 10 iterations
!$omp taskloop grainsize(10)
do i = 1, 100
  call process(i)
end do

! Using NUM_TASKS - create exactly 4 tasks
!$omp taskloop num_tasks(4)
do i = 1, 100
  call process(i)
end do

! Using NOGROUP - no implicit taskgroup synchronization
!$omp taskloop nogroup
do i = 1, 100
  call process(i)
end do
```

---

### Clause 1: GRAINSIZE Implementation

**Purpose**: Controls task granularity by specifying how many loop iterations each task should execute.

**OpenMP Semantics**:
- Takes a scalar integer expression
- Can have modifiers: `strict` (OpenMP 5.1+)
- Runtime may adjust grainsize for load balancing (unless `strict` modifier specified)
- Mutually exclusive with NUM_TASKS

#### Parser Implementation

**File**: `flang/include/flang/Parser/parse-tree.h` (~3800)

```cpp
// GRAINSIZE modifier (OpenMP 5.1+)
struct OmpGrainsizeModifier {
  ENUM_CLASS(Type, Strict)
  WRAPPER_CLASS_BOILERPLATE(OmpGrainsizeModifier, Type);
};

// GRAINSIZE clause structure
struct OmpClause::Grainsize {
  TUPLE_CLASS_BOILERPLATE(OmpClause::Grainsize);
  // (optional modifier, scalar integer expression)
  std::tuple<std::optional<OmpGrainsizeModifier>, ScalarIntExpr> t;
};
```

**File**: `flang/lib/Parser/openmp-parsers.cpp` (~500)

```cpp
// Parse GRAINSIZE modifier
TYPE_PARSER(construct<OmpGrainsizeModifier>(
    construct<OmpGrainsizeModifier::Type>(
        "STRICT" >> pure(OmpGrainsizeModifier::Type::Strict))))

// Parse GRAINSIZE clause
TYPE_PARSER(construct<OmpClause::Grainsize>(
    "GRAINSIZE" >> parenthesized(
        // Optional modifier followed by colon
        maybe(Parser<OmpGrainsizeModifier>{} / ":"_tok),
        // Scalar integer expression
        scalarIntExpr)))
```

**Test**: `flang/test/Parser/OpenMP/taskloop-grainsize.f90`

```fortran
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck %s

program test_grainsize
  integer :: i, n
  
  n = 100
  
  ! CHECK: OmpClause -> Grainsize
  ! CHECK: ScalarIntExpr
  !$omp taskloop grainsize(10)
  do i = 1, n
    print *, i
  end do
  
  ! CHECK: OmpClause -> Grainsize
  ! CHECK: OmpGrainsizeModifier -> Type = Strict
  ! CHECK: ScalarIntExpr
  !$omp taskloop grainsize(strict:20)
  do i = 1, n
    print *, i
  end do
  
  ! Variable grainsize
  ! CHECK: OmpClause -> Grainsize
  !$omp taskloop grainsize(n/10)
  do i = 1, n
    print *, i
  end do

end program
```

#### Semantic Validation

**File**: `flang/lib/Semantics/check-omp-structure.cpp` (~2700)

```cpp
void OmpStructureChecker::Enter(const parser::OmpClause::Grainsize &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_grainsize);
  
  const auto &[modifier, expr] = x.t;
  
  // GRAINSIZE only allowed on TASKLOOP
  const auto currentDir = GetContext().directive;
  if (currentDir != llvm::omp::Directive::OMPD_taskloop &&
      currentDir != llvm::omp::Directive::OMPD_taskloop_simd) {
    context_.Say(GetContext().clauseSource,
        "GRAINSIZE clause is only allowed on TASKLOOP constructs"_err_en_US);
  }
  
  // Check for mutual exclusivity with NUM_TASKS
  if (FindClause(llvm::omp::Clause::OMPC_num_tasks)) {
    context_.Say(GetContext().clauseSource,
        "GRAINSIZE and NUM_TASKS clauses are mutually exclusive"_err_en_US);
  }
  
  // Expression must be positive
  RequiresPositiveParameter(llvm::omp::Clause::OMPC_grainsize, expr);
  
  // Validate modifier (if present)
  if (modifier) {
    // STRICT modifier is OpenMP 5.1+
    // May want to check OpenMP version here
  }
}
```

#### MLIR Dialect

**File**: `mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td` (~850)

```tablegen
def TaskLoopOp : OpenMP_Op<"taskloop", [AttrSizedOperandSegments]> {
  let summary = "taskloop directive";
  let description = [{
    The taskloop construct specifies that iterations of loops will be 
    executed in parallel using explicit tasks.
    
    OpenMP 5.2 Section 2.12.2
  }];

  let arguments = (ins
    // Loop bounds
    AnyType:$lower_bound,
    AnyType:$upper_bound,
    AnyType:$step,
    
    // GRAINSIZE/NUM_TASKS (mutually exclusive)
    Optional<I64>:$grainsize,
    Optional<I64>:$num_tasks,
    
    // Other clauses
    Optional<I1>:$if_expr,
    Optional<I1>:$final_expr,
    Optional<I32>:$priority,
    Optional<UnitAttr>:$nogroup,
    Optional<UnitAttr>:$grainsize_strict,  // Modifier flag
    
    // Data-sharing
    Variadic<AnyType>:$private_vars,
    Variadic<AnyType>:$firstprivate_vars,
    Variadic<AnyType>:$reduction_vars,
    
    OptionalAttr<ArrayAttr>:$reduction_syms
  );

  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    oilist(
      `grainsize` `(` custom<GrainsizeClause>($grainsize, $grainsize_strict) `)`
      | `num_tasks` `(` $num_tasks `)`
      | `if` `(` $if_expr `)`
      | `final` `(` $final_expr `)`
      | `priority` `(` $priority `)`
      | `nogroup` $nogroup
      | `private` `(` $private_vars `:` type($private_vars) `)`
      | `reduction` custom<ReductionClause>($reduction_syms, $reduction_vars)
    )
    `for` custom<LoopControl>($region, $lower_bound, $upper_bound, $step,
                              type($lower_bound), type($upper_bound), type($step))
    attr-dict
  }];

  let hasVerifier = 1;
}
```

**Verifier**: `mlir/lib/Dialect/OpenMP/OpenMPDialect.cpp`

```cpp
LogicalResult TaskLoopOp::verify() {
  // Verify GRAINSIZE and NUM_TASKS are mutually exclusive
  if (getGrainsize() && getNumTasks()) {
    return emitOpError("GRAINSIZE and NUM_TASKS are mutually exclusive");
  }
  
  // Verify GRAINSIZE is positive
  if (getGrainsize() && *getGrainsize() <= 0) {
    return emitOpError("GRAINSIZE must be positive");
  }
  
  // Verify loop structure
  if (getRegion().getNumArguments() != 1) {
    return emitOpError("taskloop must have exactly one induction variable");
  }
  
  return success();
}
```

#### Lowering

**File**: `flang/lib/Lower/OpenMP/Clauses.cpp` (~1400)

```cpp
void ClauseProcessor::processGrainsize(
    mlir::Value &grainsizeOperand,
    mlir::UnitAttr &strictAttr) const {
  
  const parser::CharBlock *source = nullptr;
  if (auto *clause = findClause<parser::OmpClause::Grainsize>(source)) {
    const auto &[modifier, expr] = clause->t;
    
    // Evaluate the grainsize expression
    auto &firOpBuilder = converter.getFirOpBuilder();
    mlir::Location loc = converter.getCurrentLocation();
    
    mlir::Value grainsizeVal = 
        fir::getBase(converter.genExprValue(loc, expr, symTable));
    
    // Convert to i64 if needed
    auto i64Type = firOpBuilder.getI64Type();
    if (grainsizeVal.getType() != i64Type) {
      grainsizeOperand = firOpBuilder.createConvert(loc, i64Type, grainsizeVal);
    } else {
      grainsizeOperand = grainsizeVal;
    }
    
    // Handle STRICT modifier
    if (modifier && modifier->v == parser::OmpGrainsizeModifier::Type::Strict) {
      strictAttr = firOpBuilder.getUnitAttr();
    }
  }
}
```

**File**: `flang/lib/Lower/OpenMP/OpenMP.cpp` (~2500)

```cpp
static mlir::omp::TaskLoopOp genTaskLoopOp(
    lower::AbstractConverter &converter,
    const parser::OmpClauseList &clauseList,
    mlir::Location loc) {
  
  ClauseProcessor cp(converter, clauseList);
  auto &firOpBuilder = converter.getFirOpBuilder();
  
  // Extract loop bounds (similar to SIMD)
  mlir::Value lowerBound, upperBound, step;
  // ... extract from DO construct ...
  
  // Process GRAINSIZE
  mlir::Value grainsizeOperand;
  mlir::UnitAttr grainsizeStrict;
  cp.processGrainsize(grainsizeOperand, grainsizeStrict);
  
  // Process other clauses
  mlir::Value numTasksOperand, ifOperand, finalOperand, priorityOperand;
  mlir::UnitAttr nogroupAttr;
  cp.processNumTasks(numTasksOperand);
  cp.processNogroup(nogroupAttr);
  
  // Create taskloop operation
  auto taskloopOp = firOpBuilder.create<mlir::omp::TaskLoopOp>(
      loc,
      lowerBound,
      upperBound,
      step,
      grainsizeOperand,       // May be null
      numTasksOperand,        // May be null
      ifOperand,
      finalOperand,
      priorityOperand,
      nogroupAttr,
      grainsizeStrict,
      /*private_vars=*/llvm::SmallVector<mlir::Value>(),
      /*firstprivate_vars=*/llvm::SmallVector<mlir::Value>(),
      /*reduction_vars=*/llvm::SmallVector<mlir::Value>(),
      /*reduction_syms=*/nullptr
  );
  
  return taskloopOp;
}
```

#### LLVM IR Translation

**File**: `mlir/lib/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.cpp` (~1800)

```cpp
static LogicalResult convertOmpTaskLoopOp(
    omp::TaskLoopOp taskloopOp,
    llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) {
  
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  
  // Get loop bounds
  llvm::Value *lowerBound = moduleTranslation.lookupValue(taskloopOp.getLowerBound());
  llvm::Value *upperBound = moduleTranslation.lookupValue(taskloopOp.getUpperBound());
  llvm::Value *step = moduleTranslation.lookupValue(taskloopOp.getStep());
  
  // Get GRAINSIZE or NUM_TASKS
  llvm::Value *grainsizeVal = nullptr;
  llvm::Value *numTasksVal = nullptr;
  bool isStrict = false;
  
  if (taskloopOp.getGrainsize()) {
    grainsizeVal = moduleTranslation.lookupValue(taskloopOp.getGrainsize());
    isStrict = taskloopOp.getGrainsizeStrict().has_value();
  } else if (taskloopOp.getNumTasks()) {
    numTasksVal = moduleTranslation.lookupValue(taskloopOp.getNumTasks());
  }
  
  // Create task loop body
  auto bodyCB = [&](InsertPointTy allocaIP, InsertPointTy codegenIP, 
                    llvm::Value *iv) {
    // Map induction variable and translate body
    return llvm::Error::success();
  };
  
  // Call OMPIRBuilder to create taskloop
  // Runtime function: __kmpc_taskloop(...)
  llvm::OpenMPIRBuilder::InsertPointTy allocaIP = 
      findAllocaInsertPoint(builder, moduleTranslation);
  
  auto taskloopInfo = ompBuilder->createTaskloop(
      builder.saveIP(),
      allocaIP,
      lowerBound,
      upperBound,
      step,
      grainsizeVal,
      numTasksVal,
      isStrict,
      bodyCB);
  
  return success();
}
```

**Runtime Call Generated**:
```cpp
// __kmpc_taskloop signature
void __kmpc_taskloop(
    ident_t *loc,              // Source location
    int32_t gtid,              // Global thread ID
    kmp_task_t *task,          // Task descriptor
    int if_val,                // IF clause value
    uint64_t *lb,              // Loop lower bound
    uint64_t *ub,              // Loop upper bound
    int64_t st,                // Loop step
    int nogroup,               // NOGROUP flag
    int sched,                 // Schedule type (grainsize=1, num_tasks=2)
    uint64_t grainsize,        // Grainsize or num_tasks value
    void *task_dup             // Task duplication function
);
```

---

### Clause 2: NUM_TASKS Implementation

**Purpose**: Specifies the exact number of tasks to create for the taskloop.

**OpenMP Semantics**:
- Takes a scalar integer expression
- Can have modifiers: `strict` (OpenMP 5.1+)
- Runtime may adjust num_tasks for system constraints (unless `strict`)
- Mutually exclusive with GRAINSIZE

#### Key Differences from GRAINSIZE

**GRAINSIZE**: "How many iterations per task?" → Runtime calculates number of tasks  
**NUM_TASKS**: "How many tasks total?" → Runtime calculates iterations per task

#### Parser Implementation

**File**: `flang/include/flang/Parser/parse-tree.h` (~3820)

```cpp
// NUM_TASKS modifier (OpenMP 5.1+)
struct OmpNumTasksModifier {
  ENUM_CLASS(Type, Strict)
  WRAPPER_CLASS_BOILERPLATE(OmpNumTasksModifier, Type);
};

// NUM_TASKS clause structure
struct OmpClause::NumTasks {
  TUPLE_CLASS_BOILERPLATE(OmpClause::NumTasks);
  // (optional modifier, scalar integer expression)
  std::tuple<std::optional<OmpNumTasksModifier>, ScalarIntExpr> t;
};
```

**File**: `flang/lib/Parser/openmp-parsers.cpp` (~520)

```cpp
// Parse NUM_TASKS modifier
TYPE_PARSER(construct<OmpNumTasksModifier>(
    construct<OmpNumTasksModifier::Type>(
        "STRICT" >> pure(OmpNumTasksModifier::Type::Strict))))

// Parse NUM_TASKS clause
TYPE_PARSER(construct<OmpClause::NumTasks>(
    "NUM_TASKS" >> parenthesized(
        maybe(Parser<OmpNumTasksModifier>{} / ":"_tok),
        scalarIntExpr)))
```

#### Semantic Validation

**File**: `flang/lib/Semantics/check-omp-structure.cpp` (~2750)

```cpp
void OmpStructureChecker::Enter(const parser::OmpClause::NumTasks &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_num_tasks);
  
  const auto &[modifier, expr] = x.t;
  
  // NUM_TASKS only allowed on TASKLOOP
  const auto currentDir = GetContext().directive;
  if (currentDir != llvm::omp::Directive::OMPD_taskloop &&
      currentDir != llvm::omp::Directive::OMPD_taskloop_simd) {
    context_.Say(GetContext().clauseSource,
        "NUM_TASKS clause is only allowed on TASKLOOP constructs"_err_en_US);
  }
  
  // Check for mutual exclusivity with GRAINSIZE
  if (FindClause(llvm::omp::Clause::OMPC_grainsize)) {
    context_.Say(GetContext().clauseSource,
        "NUM_TASKS and GRAINSIZE clauses are mutually exclusive"_err_en_US);
  }
  
  // Expression must be positive
  RequiresPositiveParameter(llvm::omp::Clause::OMPC_num_tasks, expr);
}
```

#### Lowering

**File**: `flang/lib/Lower/OpenMP/Clauses.cpp` (~1430)

```cpp
void ClauseProcessor::processNumTasks(
    mlir::Value &numTasksOperand,
    mlir::UnitAttr &strictAttr) const {
  
  const parser::CharBlock *source = nullptr;
  if (auto *clause = findClause<parser::OmpClause::NumTasks>(source)) {
    const auto &[modifier, expr] = clause->t;
    
    auto &firOpBuilder = converter.getFirOpBuilder();
    mlir::Location loc = converter.getCurrentLocation();
    
    mlir::Value numTasksVal = 
        fir::getBase(converter.genExprValue(loc, expr, symTable));
    
    // Convert to i64
    auto i64Type = firOpBuilder.getI64Type();
    if (numTasksVal.getType() != i64Type) {
      numTasksOperand = firOpBuilder.createConvert(loc, i64Type, numTasksVal);
    } else {
      numTasksOperand = numTasksVal;
    }
    
    // Handle STRICT modifier
    if (modifier && modifier->v == parser::OmpNumTasksModifier::Type::Strict) {
      strictAttr = firOpBuilder.getUnitAttr();
    }
  }
}
```

**Integration**: Same as GRAINSIZE in `genTaskLoopOp`, but sets `numTasksOperand` instead of `grainsizeOperand`.

**Test**: `flang/test/Semantics/OpenMP/taskloop-num-tasks-grainsize.f90`

```fortran
! RUN: %python %S/test_errors.py %s %flang_fc1 -fopenmp

program test_mutual_exclusivity
  integer :: i, n
  
  n = 100
  
  ! Valid: GRAINSIZE alone
  !$omp taskloop grainsize(10)
  do i = 1, n
    print *, i
  end do
  
  ! Valid: NUM_TASKS alone
  !$omp taskloop num_tasks(4)
  do i = 1, n
    print *, i
  end do
  
  ! ERROR: Both GRAINSIZE and NUM_TASKS
  !ERROR: NUM_TASKS and GRAINSIZE clauses are mutually exclusive
  !$omp taskloop grainsize(10) num_tasks(4)
  do i = 1, n
    print *, i
  end do
  
  ! ERROR: NUM_TASKS must be positive
  !ERROR: NUM_TASKS must be a positive integer
  !$omp taskloop num_tasks(0)
  do i = 1, n
    print *, i
  end do
  
  ! Valid: STRICT modifier
  !$omp taskloop num_tasks(strict:8)
  do i = 1, n
    print *, i
  end do

end program
```

---

### Clause 3: NOGROUP Implementation

**Purpose**: Removes the implicit taskgroup synchronization that normally surrounds a TASKLOOP construct.

**OpenMP Semantics**:
- Boolean clause (no arguments)
- By default, TASKLOOP has implicit taskgroup (waits for all tasks to complete)
- NOGROUP removes this implicit synchronization
- Tasks may outlive the taskloop region

**Performance Impact**:
- **Without NOGROUP**: Parent thread waits for all taskloop tasks to complete
- **With NOGROUP**: Parent thread continues immediately, tasks complete asynchronously

#### Parser Implementation

**File**: `flang/include/flang/Parser/parse-tree.h` (~3840)

```cpp
// NOGROUP clause - simple boolean (no arguments)
struct OmpClause::Nogroup {
  EMPTY_CLASS(Nogroup);
};
```

**File**: `flang/lib/Parser/openmp-parsers.cpp` (~540)

```cpp
// Parse NOGROUP clause
TYPE_PARSER(construct<OmpClause::Nogroup>("NOGROUP"_tok))

// Add to clause list
TYPE_PARSER(
    construct<OmpClause>(Parser<OmpClause::NumTasks>{}) ||
    construct<OmpClause>(Parser<OmpClause::Nogroup>{}) ||    // ← ADD
    construct<OmpClause>(Parser<OmpClause::Nowait>{}) ||
    // ... more clauses ...
)
```

#### Semantic Validation

**File**: `flang/lib/Semantics/check-omp-structure.cpp` (~2780)

```cpp
void OmpStructureChecker::Enter(const parser::OmpClause::Nogroup &) {
  CheckAllowed(llvm::omp::Clause::OMPC_nogroup);
  
  // NOGROUP only allowed on TASKLOOP
  const auto currentDir = GetContext().directive;
  if (currentDir != llvm::omp::Directive::OMPD_taskloop &&
      currentDir != llvm::omp::Directive::OMPD_taskloop_simd) {
    context_.Say(GetContext().clauseSource,
        "NOGROUP clause is only allowed on TASKLOOP constructs"_err_en_US);
  }
  
  // No other restrictions - simple boolean clause
}
```

#### Lowering

**File**: `flang/lib/Lower/OpenMP/Clauses.cpp` (~1460)

```cpp
void ClauseProcessor::processNogroup(mlir::UnitAttr &result) const {
  const parser::CharBlock *source = nullptr;
  auto &context = converter.getFirOpBuilder().getContext();
  
  if (findClause<parser::OmpClause::Nogroup>(source)) {
    result = mlir::UnitAttr::get(context);
  }
}
```

**Integration in genTaskLoopOp**: Same as UNTIED clause pattern - pass `UnitAttr` to operation.

#### LLVM IR Translation

**Key Implementation**: NOGROUP is passed as a flag to `__kmpc_taskloop`:

```cpp
// In convertOmpTaskLoopOp:
bool nogroup = taskloopOp.getNogroup().has_value();

// Pass to runtime (8th parameter)
ompBuilder->createTaskloop(
    // ... other parameters ...
    nogroup,  // int nogroup parameter
    // ... remaining parameters ...
);
```

**Runtime Behavior**:
```cpp
// Without NOGROUP (nogroup = 0):
__kmpc_taskgroup(loc, gtid);        // Start implicit taskgroup
__kmpc_taskloop(..., 0, ...);       // Create tasks
__kmpc_end_taskgroup(loc, gtid);    // Wait for tasks

// With NOGROUP (nogroup = 1):
__kmpc_taskloop(..., 1, ...);       // Create tasks, no wait
// Parent continues immediately
```

---

### Combined Example: All Three Clauses

**Test**: `flang/test/Lower/OpenMP/taskloop-complete.f90`

```fortran
! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s

program test_taskloop_complete
  integer :: i, n, chunk_size, num_chunks
  
  n = 1000
  chunk_size = 50
  num_chunks = 20
  
  ! CHECK-LABEL: func @_QQmain
  
  ! GRAINSIZE example
  ! CHECK: %[[GRAIN:.*]] = fir.load %{{.*}} : !fir.ref<i32>
  ! CHECK: %[[GRAIN_I64:.*]] = fir.convert %[[GRAIN]] : (i32) -> i64
  ! CHECK: omp.taskloop grainsize(%[[GRAIN_I64]] : i64)
  !$omp taskloop grainsize(chunk_size)
  do i = 1, n
    call process_chunk(i)
  end do
  
  ! NUM_TASKS example
  ! CHECK: %[[NTASKS:.*]] = fir.load %{{.*}} : !fir.ref<i32>
  ! CHECK: %[[NTASKS_I64:.*]] = fir.convert %[[NTASKS]] : (i32) -> i64
  ! CHECK: omp.taskloop num_tasks(%[[NTASKS_I64]] : i64)
  !$omp taskloop num_tasks(num_chunks)
  do i = 1, n
    call process_parallel(i)
  end do
  
  ! NOGROUP example
  ! CHECK: omp.taskloop nogroup
  !$omp taskloop nogroup
  do i = 1, n
    call async_process(i)
  end do
  
  ! Combined: GRAINSIZE + NOGROUP
  ! CHECK: omp.taskloop grainsize(%{{.*}} : i64) nogroup
  !$omp taskloop grainsize(100) nogroup
  do i = 1, n
    call fast_process(i)
  end do
  
  ! STRICT modifier (OpenMP 5.1)
  ! CHECK: omp.taskloop grainsize(%{{.*}} : i64) {grainsize_strict}
  !$omp taskloop grainsize(strict:50)
  do i = 1, n
    call strict_process(i)
  end do

end program
```

---

### Implementation Comparison Table

| **Aspect** | **GRAINSIZE** | **NUM_TASKS** | **NOGROUP** |
|------------|---------------|---------------|-------------|
| **Type** | Scalar int expression | Scalar int expression | Boolean (no args) |
| **Parse Tree** | `TUPLE_CLASS` (modifier, expr) | `TUPLE_CLASS` (modifier, expr) | `EMPTY_CLASS` |
| **Modifiers** | `strict` (5.1+) | `strict` (5.1+) | None |
| **Restrictions** | Mutually exclusive with NUM_TASKS | Mutually exclusive with GRAINSIZE | None |
| **MLIR Operand** | `Optional<I64>` | `Optional<I64>` | `Optional<UnitAttr>` |
| **Runtime Impact** | Controls task size | Controls task count | Removes taskgroup barrier |
| **Default Behavior** | Runtime-determined grainsize | Runtime-determined num_tasks | Implicit taskgroup added |

---

### Semantic Restriction Summary

**GRAINSIZE Restrictions**:
1. Only on TASKLOOP/TASKLOOP SIMD
2. Must be positive integer
3. Cannot coexist with NUM_TASKS
4. Expression evaluated at taskloop entry

**NUM_TASKS Restrictions**:
1. Only on TASKLOOP/TASKLOOP SIMD
2. Must be positive integer
3. Cannot coexist with GRAINSIZE
4. Expression evaluated at taskloop entry

**NOGROUP Restrictions**:
1. Only on TASKLOOP/TASKLOOP SIMD
2. No expression restrictions (boolean)
3. Can coexist with GRAINSIZE or NUM_TASKS

**Common Validation Pattern**:
```cpp
// In check-omp-structure.cpp
void OmpStructureChecker::Leave(const parser::OmpClauseList &) {
  // Check mutual exclusivity
  bool hasGrainsize = FindClause(llvm::omp::Clause::OMPC_grainsize) != nullptr;
  bool hasNumTasks = FindClause(llvm::omp::Clause::OMPC_num_tasks) != nullptr;
  
  if (hasGrainsize && hasNumTasks) {
    context_.Say(GetContext().clauseSource,
        "GRAINSIZE and NUM_TASKS are mutually exclusive on TASKLOOP"_err_en_US);
  }
}
```

---

### Runtime Function Signatures

**TASKLOOP Runtime Call**:
```cpp
// OpenMP runtime function
void __kmpc_taskloop(
    ident_t *loc,              // Source location
    int32_t gtid,              // Global thread ID
    kmp_task_t *task,          // Task structure
    int if_val,                // IF clause condition
    uint64_t *lb,              // Loop lower bound pointer
    uint64_t *ub,              // Loop upper bound pointer
    int64_t st,                // Loop step
    int nogroup,               // 1 if NOGROUP present, 0 otherwise
    int sched,                 // 1 for GRAINSIZE, 2 for NUM_TASKS
    uint64_t grainsize_or_num, // GRAINSIZE value OR NUM_TASKS value
    void *task_dup             // Task duplication callback
);
```

**Schedule Type (sched parameter)**:
```cpp
enum kmp_taskloop_schedule_t {
  kmp_taskloop_sched_grainsize = 0,     // Use grainsize
  kmp_taskloop_sched_num_tasks = 1,     // Use num_tasks
  kmp_taskloop_sched_runtime = 2        // Neither specified, runtime decides
};
```

---

### Performance Considerations

**GRAINSIZE Selection**:
```fortran
! Too small: High task creation overhead
!$omp taskloop grainsize(1)  ! Bad: 1000 tasks for 1000 iterations

! Too large: Poor load balancing
!$omp taskloop grainsize(1000)  ! Bad: Only 1 task

! Good rule of thumb: iterations / (threads * 4)
!$omp taskloop grainsize(n / (omp_get_max_threads() * 4))
```

**NUM_TASKS Selection**:
```fortran
! Typically: multiple of thread count for load balancing
num_tasks = omp_get_max_threads() * 2

!$omp taskloop num_tasks(num_tasks)
do i = 1, n
  call process(i)
end do
```

**NOGROUP Usage**:
```fortran
! Use NOGROUP when tasks don't need to complete before continuing
!$omp taskloop nogroup
do i = 1, n
  call log_async(i)  ! Fire and forget
end do

! Continue immediately
print *, "Logging initiated"

! Don't use NOGROUP when results needed immediately
!$omp taskloop  ! Implicit taskgroup ensures completion
do i = 1, n
  results(i) = compute(i)
end do
! Results guaranteed available here
```

---

### Debugging and Testing Tips

**1. Verify Mutual Exclusivity**:
```bash
# Should fail semantic check
echo '!$omp taskloop grainsize(10) num_tasks(4)' | \
  flang-new -fsyntax-only -fopenmp
```

**2. Check Runtime Calls**:
```bash
# Verify correct schedule type passed
flang-new -S -emit-llvm -fopenmp taskloop.f90 -o - | \
  grep __kmpc_taskloop | grep -o 'i32 [0-2]'
# Should show: i32 0 (grainsize) or i32 1 (num_tasks)
```

**3. Test NOGROUP Behavior**:
```fortran
! Add timing to verify asynchronous execution
real :: start, finish

call cpu_time(start)
!$omp taskloop nogroup
do i = 1, 1000000
  call slow_work(i)
end do
call cpu_time(finish)

print *, "Time:", finish - start  ! Should be very small with NOGROUP
```

---

### Files Modified Summary

| **Clause** | **Files** | **Key Changes** |
|------------|-----------|-----------------|
| **GRAINSIZE** | `parse-tree.h` | Add `OmpGrainsizeModifier`, `OmpClause::Grainsize` |
| | `openmp-parsers.cpp` | Parse modifier and expression |
| | `check-omp-structure.cpp` | Validate positive, check NUM_TASKS exclusivity |
| | `OpenMPOps.td` | Add `grainsize`, `grainsize_strict` to TaskLoopOp |
| | `Clauses.cpp` | Implement `processGrainsize()` |
| | `OpenMPToLLVMIRTranslation.cpp` | Pass grainsize to runtime |
| **NUM_TASKS** | `parse-tree.h` | Add `OmpNumTasksModifier`, `OmpClause::NumTasks` |
| | `openmp-parsers.cpp` | Parse modifier and expression |
| | `check-omp-structure.cpp` | Validate positive, check GRAINSIZE exclusivity |
| | `OpenMPOps.td` | Add `num_tasks`, `num_tasks_strict` to TaskLoopOp |
| | `Clauses.cpp` | Implement `processNumTasks()` |
| | `OpenMPToLLVMIRTranslation.cpp` | Pass num_tasks to runtime |
| **NOGROUP** | `parse-tree.h` | Add `OmpClause::Nogroup` (EMPTY_CLASS) |
| | `openmp-parsers.cpp` | Parse keyword |
| | `check-omp-structure.cpp` | Validate on TASKLOOP only |
| | `OpenMPOps.td` | Add `nogroup` UnitAttr to TaskLoopOp |
| | `Clauses.cpp` | Implement `processNogroup()` |
| | `OpenMPToLLVMIRTranslation.cpp` | Pass nogroup flag to runtime |

---

## DEPEND CLAUSE REFERENCE AND LOWERING PATTERNS

This section provides a comprehensive reference for all DEPEND clause types used in task-based OpenMP constructs. DEPEND clauses establish task dependencies, ensuring that tasks execute in a specified order based on data dependencies.

### Overview: Task Dependencies in OpenMP

**OpenMP Specification**: OpenMP 5.2 Section 2.19.11  
**Applicable Directives**: TASK, TASKLOOP, TARGET, TARGET UPDATE, TARGET ENTER DATA, TARGET EXIT DATA

**Purpose**: 
- Specify ordering constraints between sibling tasks
- Enable task scheduling based on data dependencies
- Prevent data races in task-parallel code
- Allow runtime to optimize task execution order

**Fortran Syntax**:
```fortran
!$omp task depend(depend-type: variable-list)
!$omp task depend(in: x, y) depend(out: z)
!$omp taskloop depend(inout: array)
```

---

### Dependency Types Reference

#### 1. IN (Input Dependency)

**Semantics**: Task reads from specified variables; must wait for previous tasks with OUT/INOUT on same variables to complete.

**OpenMP Spec**: Section 2.19.11.1  
**Use Case**: Read-only access to shared data

**Fortran Example**:
```fortran
integer :: shared_data

!$omp task depend(out: shared_data)
  shared_data = compute_value()
!$omp end task

!$omp task depend(in: shared_data)
  call process(shared_data)  ! Waits for previous task
!$omp end task
```

**Dependency Semantics**:
- **IN** task waits for: Tasks with OUT or INOUT on same variable
- **IN** task allows: Other IN tasks to execute concurrently
- **Memory behavior**: Read-only, multiple readers allowed

---

#### 2. OUT (Output Dependency)

**Semantics**: Task writes to specified variables; must wait for all previous IN, OUT, and INOUT tasks on same variables to complete.

**OpenMP Spec**: Section 2.19.11.1  
**Use Case**: Write-only access, overwrites data completely

**Fortran Example**:
```fortran
integer :: result

!$omp task depend(out: result)
  result = 100
!$omp end task

!$omp task depend(out: result)
  result = 200  ! Waits for previous task, overwrites
!$omp end task

!$omp task depend(in: result)
  print *, result  ! Will see 200
!$omp end task
```

**Dependency Semantics**:
- **OUT** task waits for: All previous IN, OUT, INOUT tasks on same variable
- **OUT** task blocks: All subsequent tasks on same variable
- **Memory behavior**: Write-only, serializes all access

---

#### 3. INOUT (Input-Output Dependency)

**Semantics**: Task both reads and writes; waits for all previous tasks on same variables, blocks all subsequent tasks.

**OpenMP Spec**: Section 2.19.11.1  
**Use Case**: Read-modify-write operations

**Fortran Example**:
```fortran
real :: temperature

!$omp task depend(out: temperature)
  temperature = initial_temp()
!$omp end task

!$omp task depend(inout: temperature)
  temperature = temperature * 1.1  ! Read, modify, write
!$omp end task

!$omp task depend(inout: temperature)
  temperature = temperature + 273.15  ! Another RMW
!$omp end task
```

**Dependency Semantics**:
- **INOUT** task waits for: All previous IN, OUT, INOUT tasks
- **INOUT** task blocks: All subsequent tasks
- **Memory behavior**: Read-modify-write, full serialization

---

#### 4. MUTEXINOUTSET (Mutual Exclusion Inout Set)

**Semantics**: Task has exclusive access but allows concurrent execution of other MUTEXINOUTSET tasks on the same variable if they don't actually conflict at runtime.

**OpenMP Spec**: OpenMP 5.0+, Section 2.19.11.1  
**Use Case**: Fine-grained concurrent access (e.g., different array elements)

**Fortran Example**:
```fortran
real :: array(1000)

do i = 1, 1000, 10
  !$omp task depend(mutexinoutset: array)
    ! Each task accesses different array elements
    call process_chunk(array(i:i+9))
  !$omp end task
end do
! Runtime can execute these concurrently if proven safe
```

**Dependency Semantics**:
- **MUTEXINOUTSET** task waits for: Previous IN, OUT, INOUT tasks
- **MUTEXINOUTSET** tasks: May execute concurrently with each other (runtime decision)
- **Memory behavior**: Potentially concurrent if non-overlapping access

**Difference from INOUT**:
- **INOUT**: Always serializes
- **MUTEXINOUTSET**: Allows concurrent execution if runtime determines safety

---

#### 5. DEPOBJ (Dependency Object)

**Semantics**: Uses a pre-initialized dependency object containing stored dependency information.

**OpenMP Spec**: OpenMP 5.0+, Section 2.19.11.3  
**Use Case**: Reusable dependency patterns, dynamic dependencies

**Fortran Example**:
```fortran
use omp_lib
integer(omp_depend_kind) :: dep_obj
integer :: x

! Initialize dependency object
!$omp depobj(dep_obj) depend(inout: x)

! Use in multiple tasks
!$omp task depend(depobj: dep_obj)
  call task_work_1(x)
!$omp end task

!$omp task depend(depobj: dep_obj)
  call task_work_2(x)
!$omp end task

! Update dependency object
!$omp depobj(dep_obj) update(in)

! Destroy when done
!$omp depobj(dep_obj) destroy
```

**Dependency Semantics**:
- **DEPOBJ** inherits: Dependency type from stored object
- **Runtime behavior**: Same as if explicit dependency specified
- **Memory behavior**: Determined by stored dependency type

**Benefits**:
- Reuse complex dependency patterns
- Dynamic dependency creation
- Separate dependency specification from task creation

---

#### 6. SOURCE (Deprecated in OpenMP 5.2)

**Semantics**: Task is a source point for ordered dependencies (replaced by DOACROSS in OpenMP 5.2).

**Note**: Use DOACROSS(source:) for new code  
**Use Case**: Legacy ordered dependencies in loops

---

#### 7. SINK (Deprecated in OpenMP 5.2)

**Semantics**: Task waits for source task at specific iteration (replaced by DOACROSS in OpenMP 5.2).

**Note**: Use DOACROSS(sink:) for new code  
**Use Case**: Legacy loop-carried dependencies

---

### Implementation Architecture

#### Parse Tree Structures

**File**: `flang/include/flang/Parser/parse-tree.h` (~3900)

```cpp
// Dependency type modifiers
struct OmpDependenceType {
  ENUM_CLASS(Type, In, Out, Inout, Mutexinoutset, Depobj, Source, Sink)
  WRAPPER_CLASS_BOILERPLATE(OmpDependenceType, Type);
};

// Dependency sink iterator
struct OmpDependSinkVec {
  WRAPPER_CLASS_BOILERPLATE(OmpDependSinkVec, std::list<OmpDependSinkVecLength>);
};

struct OmpDependSinkVecLength {
  TUPLE_CLASS_BOILERPLATE(OmpDependSinkVecLength);
  std::tuple<Name, std::optional<ScalarIntConstantExpr>> t;
};

// DEPEND clause structure
struct OmpClause::Depend {
  UNION_CLASS_BOILERPLATE(OmpClause::Depend);
  
  // Union of different depend forms
  std::variant<
    // depend(in/out/inout/mutexinoutset: list)
    std::tuple<OmpDependenceType, std::list<Designator>>,
    // depend(source)
    OmpDependenceType::Type,
    // depend(sink: vec)
    OmpDependSinkVec
  > u;
};

// DEPOBJ directive for dependency objects
struct OmpDepobj {
  TUPLE_CLASS_BOILERPLATE(OmpDepobj);
  std::tuple<
    Designator,                          // dependency object variable
    std::variant<
      OmpClause::Depend,                 // depend clause
      OmpClause::Destroy,                // destroy action
      OmpClause::Update                  // update action
    >
  > t;
};
```

#### Parser Implementation

**File**: `flang/lib/Parser/openmp-parsers.cpp` (~600)

```cpp
// Parse dependency type
TYPE_PARSER(construct<OmpDependenceType>(
    construct<OmpDependenceType::Type>(
        "IN"_tok >> pure(OmpDependenceType::Type::In) ||
        "OUT"_tok >> pure(OmpDependenceType::Type::Out) ||
        "INOUT"_tok >> pure(OmpDependenceType::Type::Inout) ||
        "MUTEXINOUTSET"_tok >> pure(OmpDependenceType::Type::Mutexinoutset) ||
        "DEPOBJ"_tok >> pure(OmpDependenceType::Type::Depobj) ||
        "SOURCE"_tok >> pure(OmpDependenceType::Type::Source) ||
        "SINK"_tok >> pure(OmpDependenceType::Type::Sink))))

// Parse depend sink vector
TYPE_PARSER(construct<OmpDependSinkVecLength>(
    name, maybe("+"_tok >> scalarIntConstantExpr ||
                "-"_tok >> applyFunction(negateScalarIntConstantExpr,
                                        scalarIntConstantExpr))))

TYPE_PARSER(construct<OmpDependSinkVec>(
    Parser<OmpDependSinkVecLength>{} / ","_tok))

// Parse DEPEND clause (three variants)
TYPE_PARSER(construct<OmpClause::Depend>(
    "DEPEND" >> parenthesized(
        // Variant 1: depend(type: list)
        construct<OmpClause::Depend>(
            Parser<OmpDependenceType>{},
            ":" >> nonemptyList(designator)) ||
        // Variant 2: depend(source)
        construct<OmpClause::Depend>(
            "SOURCE"_tok >> pure(OmpDependenceType::Type::Source)) ||
        // Variant 3: depend(sink: vec)
        construct<OmpClause::Depend>(
            "SINK" >> ":" >> Parser<OmpDependSinkVec>{}))))

// Parse DEPOBJ directive
TYPE_PARSER(construct<OpenMPDepobjConstruct>(
    Parser<OmpDepobj>{}))

TYPE_PARSER(construct<OmpDepobj>(
    designator,
    parenthesized(
        Parser<OmpClause::Depend>{} ||
        Parser<OmpClause::Destroy>{} ||
        Parser<OmpClause::Update>{})))
```

**Test**: `flang/test/Parser/OpenMP/depend-clause.f90`

```fortran
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck %s

program test_depend
  use omp_lib
  integer :: x, y, z
  integer :: array(100)
  integer(omp_depend_kind) :: dep_obj
  
  ! CHECK: OmpClause -> Depend
  ! CHECK: OmpDependenceType -> Type = In
  !$omp task depend(in: x)
    y = x + 1
  !$omp end task
  
  ! CHECK: OmpClause -> Depend
  ! CHECK: OmpDependenceType -> Type = Out
  !$omp task depend(out: z)
    z = compute()
  !$omp end task
  
  ! CHECK: OmpClause -> Depend
  ! CHECK: OmpDependenceType -> Type = Inout
  !$omp task depend(inout: x, y)
    x = x + y
  !$omp end task
  
  ! CHECK: OmpClause -> Depend
  ! CHECK: OmpDependenceType -> Type = Mutexinoutset
  !$omp task depend(mutexinoutset: array)
    call process(array)
  !$omp end task
  
  ! Multiple dependencies
  ! CHECK: OmpClause -> Depend
  ! CHECK: OmpDependenceType -> Type = In
  ! CHECK: OmpClause -> Depend
  ! CHECK: OmpDependenceType -> Type = Out
  !$omp task depend(in: x, y) depend(out: z)
    z = x + y
  !$omp end task
  
  ! DEPOBJ usage
  ! CHECK: OmpDepobj
  !$omp depobj(dep_obj) depend(inout: x)
  
  ! CHECK: OmpClause -> Depend
  ! CHECK: OmpDependenceType -> Type = Depobj
  !$omp task depend(depobj: dep_obj)
    call work(x)
  !$omp end task

end program
```

---

### Semantic Validation

**File**: `flang/lib/Semantics/check-omp-structure.cpp` (~2900)

```cpp
void OmpStructureChecker::Enter(const parser::OmpClause::Depend &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_depend);
  
  const auto currentDir = GetContext().directive;
  
  // DEPEND allowed on task constructs
  bool isTaskConstruct = 
      currentDir == llvm::omp::Directive::OMPD_task ||
      currentDir == llvm::omp::Directive::OMPD_taskloop ||
      currentDir == llvm::omp::Directive::OMPD_taskloop_simd;
  
  bool isTargetConstruct =
      currentDir == llvm::omp::Directive::OMPD_target ||
      currentDir == llvm::omp::Directive::OMPD_target_update ||
      currentDir == llvm::omp::Directive::OMPD_target_enter_data ||
      currentDir == llvm::omp::Directive::OMPD_target_exit_data;
  
  if (!isTaskConstruct && !isTargetConstruct) {
    context_.Say(GetContext().clauseSource,
        "DEPEND clause is only allowed on task or target constructs"_err_en_US);
  }
  
  // Visit the dependency type and variables
  std::visit(
    common::visitors{
      [&](const std::tuple<parser::OmpDependenceType, 
                           std::list<parser::Designator>> &tuple) {
        const auto &[depType, objectList] = tuple;
        
        // Validate each variable in dependency list
        for (const auto &designator : objectList) {
          // Check variable exists and is valid
          if (const auto *name = std::get_if<parser::Name>(&designator.u)) {
            const auto *symbol = ResolveOmp(*name, SymbolFlag::ObjectName, context_);
            if (symbol) {
              // Check variable is not threadprivate
              if (symbol->test(Symbol::Flag::OmpThreadprivate)) {
                context_.Say(name->source,
                    "Threadprivate variables cannot appear in DEPEND clause"_err_en_US);
              }
            }
          }
        }
        
        // SOURCE and SINK are deprecated in OpenMP 5.2
        if (depType.v == parser::OmpDependenceType::Type::Source ||
            depType.v == parser::OmpDependenceType::Type::Sink) {
          context_.Say(GetContext().clauseSource,
              "SOURCE and SINK are deprecated in OpenMP 5.2; use DOACROSS instead"_warn_en_US);
        }
      },
      [&](const parser::OmpDependenceType::Type &source) {
        // depend(source) form
        if (currentDir != llvm::omp::Directive::OMPD_ordered) {
          context_.Say(GetContext().clauseSource,
              "DEPEND(SOURCE) is only allowed in ORDERED construct"_err_en_US);
        }
      },
      [&](const parser::OmpDependSinkVec &sinkVec) {
        // depend(sink: vec) form
        if (currentDir != llvm::omp::Directive::OMPD_ordered) {
          context_.Say(GetContext().clauseSource,
              "DEPEND(SINK:...) is only allowed in ORDERED construct"_err_en_US);
        }
      }
    },
    x.u
  );
}

// Validate DEPOBJ directive
void OmpStructureChecker::Enter(const parser::OmpDepobj &x) {
  const auto &[depObj, action] = x.t;
  
  // Dependency object must be integer(omp_depend_kind)
  if (const auto *name = std::get_if<parser::Name>(&depObj.u)) {
    const auto *symbol = ResolveOmp(*name, SymbolFlag::ObjectName, context_);
    if (symbol) {
      const auto &type = symbol->GetType();
      // Check for integer(omp_depend_kind)
      if (!type || !type->IsInteger() || 
          type->kind() != OmpDependKind) {
        context_.Say(name->source,
            "DEPOBJ variable must be of type INTEGER(OMP_DEPEND_KIND)"_err_en_US);
      }
    }
  }
  
  // Validate action (depend/destroy/update)
  std::visit(
    common::visitors{
      [&](const parser::OmpClause::Depend &dep) {
        // Validate depend clause
      },
      [&](const parser::OmpClause::Destroy &) {
        // No additional validation
      },
      [&](const parser::OmpClause::Update &update) {
        // Update must specify new dependency type
      }
    },
    action
  );
}

// Check for duplicate variables across dependencies
void OmpStructureChecker::CheckDependenceVariables(
    const parser::OmpClauseList &clauses) {
  
  std::map<const Symbol*, std::vector<parser::OmpDependenceType::Type>> 
      varDependencies;
  
  for (const auto &clause : clauses) {
    if (const auto *depend = std::get_if<parser::OmpClause::Depend>(&clause.u)) {
      // Extract variables and dependency types
      // Check for conflicting dependency types on same variable
    }
  }
}
```

**Test**: `flang/test/Semantics/OpenMP/depend-errors.f90`

```fortran
! RUN: %python %S/test_errors.py %s %flang_fc1 -fopenmp

program depend_errors
  use omp_lib
  integer :: x, y
  integer :: arr(100)
  real :: wrong_type
  
  ! ERROR: DEPEND only on task constructs
  !ERROR: DEPEND clause is only allowed on task or target constructs
  !$omp parallel depend(in: x)
    y = x
  !$omp end parallel
  
  ! ERROR: Threadprivate variable in DEPEND
  !$omp threadprivate(x)
  !ERROR: Threadprivate variables cannot appear in DEPEND clause
  !$omp task depend(in: x)
    y = x + 1
  !$omp end task
  
  ! Valid: Multiple dependencies
  !$omp task depend(in: x, y) depend(out: arr)
    arr(1) = x + y
  !$omp end task
  
  ! ERROR: Wrong type for DEPOBJ
  !ERROR: DEPOBJ variable must be of type INTEGER(OMP_DEPEND_KIND)
  !$omp depobj(wrong_type) depend(in: x)

end program
```

---

### MLIR Dialect Representation

**File**: `mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td` (~950)

```tablegen
// Dependency type enumeration
def DependTypeIn : I32EnumAttrCase<"in", 0>;
def DependTypeOut : I32EnumAttrCase<"out", 1>;
def DependTypeInout : I32EnumAttrCase<"inout", 2>;
def DependTypeMutexinoutset : I32EnumAttrCase<"mutexinoutset", 3>;
def DependTypeDepobj : I32EnumAttrCase<"depobj", 4>;

def DependType : I32EnumAttr<"DependType",
    "OpenMP task dependency types",
    [DependTypeIn, DependTypeOut, DependTypeInout, 
     DependTypeMutexinoutset, DependTypeDepobj]> {
  let cppNamespace = "::mlir::omp";
}

// Task operation with dependencies
def TaskOp : OpenMP_Op<"task", [AttrSizedOperandSegments]> {
  let summary = "task directive";
  let description = [{
    The task construct defines an explicit task.
    
    Dependencies can be specified using the depend clause to establish
    ordering constraints between sibling tasks.
    
    OpenMP 5.2 Section 2.12.1
  }];

  let arguments = (ins
    // Task control
    Optional<I1>:$if_expr,
    Optional<I1>:$final_expr,
    Optional<I32>:$priority,
    
    // Flags
    Optional<UnitAttr>:$untied,
    Optional<UnitAttr>:$mergeable,
    
    // Data-sharing
    Variadic<AnyType>:$private_vars,
    Variadic<AnyType>:$firstprivate_vars,
    
    // Dependencies
    Variadic<AnyType>:$depend_vars,
    OptionalAttr<ArrayAttr>:$depend_types,  // Array of DependType
    
    // Allocator
    Optional<AnyType>:$allocate_vars
  );

  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    oilist(
      `if` `(` $if_expr `)`
      | `final` `(` $final_expr `)`
      | `priority` `(` $priority `)`
      | `untied` $untied
      | `mergeable` $mergeable
      | `private` `(` $private_vars `:` type($private_vars) `)`
      | `firstprivate` `(` $firstprivate_vars `:` type($firstprivate_vars) `)`
      | `depend` custom<DependClause>($depend_types, $depend_vars)
      | `allocate` `(` $allocate_vars `:` type($allocate_vars) `)`
    )
    $region attr-dict
  }];

  let hasVerifier = 1;
}

// Dependency object operation
def DepobjOp : OpenMP_Op<"depobj"> {
  let summary = "depobj directive";
  let description = [{
    The depobj construct initializes, updates, or destroys a dependency object.
    
    OpenMP 5.2 Section 2.19.12
  }];

  let arguments = (ins
    AnyType:$depobj,
    OptionalAttr<DependType>:$depend_type,
    Variadic<AnyType>:$depend_vars,
    Optional<UnitAttr>:$destroy,
    Optional<UnitAttr>:$update
  );

  let assemblyFormat = [{
    `depobj` `(` $depobj `:` type($depobj) `)`
    oilist(
      `depend` `(` $depend_type `:` $depend_vars `:` type($depend_vars) `)`
      | `destroy` $destroy
      | `update` $update
    )
    attr-dict
  }];
}
```

**Custom Printer/Parser for Dependencies**:

**File**: `mlir/lib/Dialect/OpenMP/OpenMPDialect.cpp` (~800)

```cpp
// Print depend clause: depend(in: %x, %y) depend(out: %z)
static void printDependClause(
    OpAsmPrinter &p,
    Operation *op,
    ArrayAttr dependTypes,
    OperandRange dependVars) {
  
  if (!dependVars.empty()) {
    // Group variables by dependency type
    std::map<omp::DependType, SmallVector<Value>> grouped;
    
    for (auto [var, typeAttr] : llvm::zip(dependVars, dependTypes)) {
      auto type = cast<omp::DependTypeAttr>(typeAttr).getValue();
      grouped[type].push_back(var);
    }
    
    // Print each group: depend(type: vars)
    bool first = true;
    for (auto &[type, vars] : grouped) {
      if (!first) p << " ";
      p << "depend(";
      
      switch (type) {
        case omp::DependType::in: p << "in"; break;
        case omp::DependType::out: p << "out"; break;
        case omp::DependType::inout: p << "inout"; break;
        case omp::DependType::mutexinoutset: p << "mutexinoutset"; break;
        case omp::DependType::depobj: p << "depobj"; break;
      }
      
      p << ": ";
      p.printOperands(vars);
      p << ")";
      first = false;
    }
  }
}

// Parse depend clause
static ParseResult parseDependClause(
    OpAsmParser &parser,
    ArrayAttr &dependTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dependVars) {
  
  SmallVector<Attribute> types;
  
  // Parse multiple depend clauses
  while (succeeded(parser.parseOptionalKeyword("depend"))) {
    if (parser.parseLParen())
      return failure();
    
    // Parse dependency type
    omp::DependType depType;
    StringRef typeStr;
    if (parser.parseKeyword(&typeStr))
      return failure();
    
    if (typeStr == "in") depType = omp::DependType::in;
    else if (typeStr == "out") depType = omp::DependType::out;
    else if (typeStr == "inout") depType = omp::DependType::inout;
    else if (typeStr == "mutexinoutset") depType = omp::DependType::mutexinoutset;
    else if (typeStr == "depobj") depType = omp::DependType::depobj;
    else return parser.emitError(parser.getNameLoc(), "unknown dependency type");
    
    if (parser.parseColon())
      return failure();
    
    // Parse variable list
    SmallVector<OpAsmParser::UnresolvedOperand> vars;
    if (parser.parseOperandList(vars))
      return failure();
    
    // Store type for each variable
    for (auto &var : vars) {
      types.push_back(omp::DependTypeAttr::get(parser.getContext(), depType));
      dependVars.push_back(var);
    }
    
    if (parser.parseRParen())
      return failure();
  }
  
  dependTypes = ArrayAttr::get(parser.getContext(), types);
  return success();
}
```

---

### Lowering Implementation

**File**: `flang/lib/Lower/OpenMP/Clauses.cpp` (~1500)

```cpp
// Dependency information structure
struct DependInfo {
  omp::DependType type;
  mlir::Value var;
};

void ClauseProcessor::processDepend(
    llvm::SmallVectorImpl<mlir::Value> &dependVars,
    llvm::SmallVectorImpl<mlir::Attribute> &dependTypes) const {
  
  auto &firOpBuilder = converter.getFirOpBuilder();
  auto context = firOpBuilder.getContext();
  
  const parser::CharBlock *source = nullptr;
  
  // Process all DEPEND clauses (can have multiple)
  for (const auto &clause : clauses) {
    if (const auto *depend = 
            std::get_if<parser::OmpClause::Depend>(&clause.u)) {
      
      std::visit(
        common::visitors{
          // depend(type: var-list)
          [&](const std::tuple<parser::OmpDependenceType, 
                               std::list<parser::Designator>> &tuple) {
            const auto &[depTypeNode, designators] = tuple;
            
            // Convert parser dependency type to MLIR enum
            omp::DependType depType;
            switch (depTypeNode.v) {
              case parser::OmpDependenceType::Type::In:
                depType = omp::DependType::in;
                break;
              case parser::OmpDependenceType::Type::Out:
                depType = omp::DependType::out;
                break;
              case parser::OmpDependenceType::Type::Inout:
                depType = omp::DependType::inout;
                break;
              case parser::OmpDependenceType::Type::Mutexinoutset:
                depType = omp::DependType::mutexinoutset;
                break;
              case parser::OmpDependenceType::Type::Depobj:
                depType = omp::DependType::depobj;
                break;
              default:
                llvm_unreachable("Unsupported dependency type");
            }
            
            // Process each variable in dependency list
            for (const auto &designator : designators) {
              mlir::Location loc = converter.getCurrentLocation();
              
              // Generate address of the variable
              mlir::Value addr;
              
              if (const auto *name = 
                      std::get_if<parser::Name>(&designator.u)) {
                // Simple variable
                const auto *symbol = 
                    converter.getSymbolAddress(*name->symbol);
                addr = fir::getBase(symbol);
                
              } else if (const auto *arrayElem = 
                             std::get_if<parser::ArrayElement>(&designator.u)) {
                // Array element or section
                addr = converter.genArrayElementAddr(loc, *arrayElem);
                
              } else if (const auto *structComp = 
                             std::get_if<parser::StructureComponent>(&designator.u)) {
                // Derived type component
                addr = converter.genStructComponentAddr(loc, *structComp);
              }
              
              // Add to lists
              dependVars.push_back(addr);
              dependTypes.push_back(
                  omp::DependTypeAttr::get(context, depType));
            }
          },
          
          // depend(source) - legacy
          [&](const parser::OmpDependenceType::Type &source) {
            // Handle source dependency (deprecated)
            // Typically converted to barrier or ignored
          },
          
          // depend(sink: vec) - legacy
          [&](const parser::OmpDependSinkVec &sinkVec) {
            // Handle sink dependencies (deprecated)
            // Requires iteration space tracking
          }
        },
        depend->u
      );
    }
  }
}

// Process DEPOBJ directive
static void genDepobjOp(
    lower::AbstractConverter &converter,
    const parser::OmpDepobj &depobj,
    mlir::Location loc) {
  
  auto &firOpBuilder = converter.getFirOpBuilder();
  
  const auto &[depobjDesignator, action] = depobj.t;
  
  // Get depobj variable (must be integer(omp_depend_kind))
  mlir::Value depobjVar = 
      converter.genDesignatorAddr(loc, depobjDesignator);
  
  std::visit(
    common::visitors{
      // DEPEND clause - initialize depobj
      [&](const parser::OmpClause::Depend &depend) {
        llvm::SmallVector<mlir::Value> dependVars;
        llvm::SmallVector<mlir::Attribute> dependTypes;
        
        // Process depend clause to extract type and variables
        // ... (similar to processDepend) ...
        
        auto depobjOp = firOpBuilder.create<mlir::omp::DepobjOp>(
            loc,
            depobjVar,
            dependTypes[0].cast<omp::DependTypeAttr>().getValue(),
            dependVars,
            /*destroy=*/nullptr,
            /*update=*/nullptr);
      },
      
      // DESTROY - destroy depobj
      [&](const parser::OmpClause::Destroy &) {
        auto depobjOp = firOpBuilder.create<mlir::omp::DepobjOp>(
            loc,
            depobjVar,
            /*depend_type=*/nullptr,
            /*depend_vars=*/{},
            /*destroy=*/firOpBuilder.getUnitAttr(),
            /*update=*/nullptr);
      },
      
      // UPDATE - update depobj dependency type
      [&](const parser::OmpClause::Update &update) {
        // Extract new dependency type from update clause
        omp::DependType newType = extractUpdateType(update);
        
        auto depobjOp = firOpBuilder.create<mlir::omp::DepobjOp>(
            loc,
            depobjVar,
            newType,
            /*depend_vars=*/{},
            /*destroy=*/nullptr,
            /*update=*/firOpBuilder.getUnitAttr());
      }
    },
    action
  );
}
```

**File**: `flang/lib/Lower/OpenMP/OpenMP.cpp` (~2700)

```cpp
static mlir::omp::TaskOp genTaskOp(
    lower::AbstractConverter &converter,
    const parser::OmpClauseList &clauseList,
    mlir::Location loc) {
  
  ClauseProcessor cp(converter, clauseList);
  auto &firOpBuilder = converter.getFirOpBuilder();
  
  // Process task clauses
  mlir::Value ifOperand, finalOperand, priorityOperand;
  mlir::UnitAttr untied, mergeable;
  
  cp.processIf(ifOperand, llvm::omp::Directive::OMPD_task);
  cp.processFinal(finalOperand);
  cp.processPriority(priorityOperand);
  cp.processUntied(untied);
  cp.processMergeable(mergeable);
  
  // Process dependencies
  llvm::SmallVector<mlir::Value> dependVars;
  llvm::SmallVector<mlir::Attribute> dependTypes;
  cp.processDepend(dependVars, dependTypes);
  
  // Process data-sharing
  llvm::SmallVector<mlir::Value> privateVars, firstprivateVars;
  cp.processPrivate(privateVars);
  cp.processFirstprivate(firstprivateVars);
  
  // Create task operation
  auto taskOp = firOpBuilder.create<mlir::omp::TaskOp>(
      loc,
      ifOperand,
      finalOperand,
      priorityOperand,
      untied,
      mergeable,
      privateVars,
      firstprivateVars,
      dependVars,
      firOpBuilder.getArrayAttr(dependTypes),  // Depend types array
      /*allocate_vars=*/llvm::SmallVector<mlir::Value>());
  
  return taskOp;
}
```

**Test**: `flang/test/Lower/OpenMP/task-depend.f90`

```fortran
! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s

program task_depend
  integer :: x, y, z
  integer :: array(100)
  
  ! CHECK-LABEL: func @_QQmain
  
  ! Simple IN dependency
  ! CHECK: %[[X_ADDR:.*]] = fir.alloca i32
  ! CHECK: omp.task depend(in: %[[X_ADDR]] : !fir.ref<i32>)
  !$omp task depend(in: x)
    y = x + 1
  !$omp end task
  
  ! Simple OUT dependency
  ! CHECK: %[[Z_ADDR:.*]] = fir.alloca i32
  ! CHECK: omp.task depend(out: %[[Z_ADDR]] : !fir.ref<i32>)
  !$omp task depend(out: z)
    z = compute()
  !$omp end task
  
  ! INOUT dependency
  ! CHECK: omp.task depend(inout: %[[X_ADDR]] : !fir.ref<i32>)
  !$omp task depend(inout: x)
    x = x * 2
  !$omp end task
  
  ! Multiple dependencies
  ! CHECK: omp.task depend(in: %[[X_ADDR]] : !fir.ref<i32>) 
  ! CHECK-SAME: depend(in: %[[Y_ADDR:.*]] : !fir.ref<i32>)
  ! CHECK-SAME: depend(out: %[[Z_ADDR]] : !fir.ref<i32>)
  !$omp task depend(in: x, y) depend(out: z)
    z = x + y
  !$omp end task
  
  ! MUTEXINOUTSET dependency
  ! CHECK: %[[ARR_ADDR:.*]] = fir.alloca !fir.array<100xi32>
  ! CHECK: omp.task depend(mutexinoutset: %[[ARR_ADDR]] : !fir.ref<!fir.array<100xi32>>)
  !$omp task depend(mutexinoutset: array)
    call process_array(array)
  !$omp end task
  
  ! Array section dependency
  ! CHECK: omp.task depend(in: %{{.*}} : !fir.box<!fir.array<?xi32>>)
  !$omp task depend(in: array(1:50))
    call process_section(array(1:50))
  !$omp end task

end program
```

---

### LLVM IR Translation

**File**: `mlir/lib/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.cpp` (~2100)

```cpp
static LogicalResult convertOmpTaskOp(
    omp::TaskOp taskOp,
    llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) {
  
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  llvm::Module &module = *builder.GetInsertBlock()->getModule();
  
  // Outline task body into separate function
  llvm::Function *taskFn = 
      outlineTaskRegion(taskOp, moduleTranslation, "task_body");
  
  // Get task operands
  llvm::Value *ifCond = nullptr;
  if (taskOp.getIfExpr()) {
    ifCond = moduleTranslation.lookupValue(taskOp.getIfExpr());
  }
  
  // Process dependencies
  llvm::SmallVector<llvm::OpenMPIRBuilder::DependData> dependencies;
  
  if (!taskOp.getDependVars().empty()) {
    auto dependVars = taskOp.getDependVars();
    auto dependTypeAttrs = taskOp.getDependTypes();
    
    for (auto [var, typeAttr] : llvm::zip(dependVars, *dependTypeAttrs)) {
      llvm::Value *depAddr = moduleTranslation.lookupValue(var);
      
      auto depTypeEnum = 
          cast<omp::DependTypeAttr>(typeAttr).getValue();
      
      // Convert to OpenMPIRBuilder dependency type
      llvm::OpenMPIRBuilder::DependData::DependKind kind;
      switch (depTypeEnum) {
        case omp::DependType::in:
          kind = llvm::OpenMPIRBuilder::DependData::DepKindIn;
          break;
        case omp::DependType::out:
          kind = llvm::OpenMPIRBuilder::DependData::DepKindOut;
          break;
        case omp::DependType::inout:
          kind = llvm::OpenMPIRBuilder::DependData::DepKindInout;
          break;
        case omp::DependType::mutexinoutset:
          kind = llvm::OpenMPIRBuilder::DependData::DepKindMutexinoutset;
          break;
        case omp::DependType::depobj:
          kind = llvm::OpenMPIRBuilder::DependData::DepKindDepobj;
          break;
      }
      
      dependencies.push_back({kind, depAddr});
    }
  }
  
  // Create task using OMPIRBuilder
  auto allocaIP = findAllocaInsertPoint(builder, moduleTranslation);
  
  auto taskData = ompBuilder->createTask(
      builder.saveIP(),
      allocaIP,
      taskFn,
      ifCond,
      dependencies,
      /*final=*/nullptr,
      /*priority=*/nullptr,
      /*is_untied=*/taskOp.getUntied().has_value());
  
  return success();
}
```

**Runtime Calls Generated**:

```cpp
// Task allocation with dependencies
kmp_task_t *task = __kmpc_omp_task_alloc(
    &loc,                    // Source location
    gtid,                    // Global thread ID
    flags,                   // Task flags (untied, final, etc.)
    sizeof(kmp_task_t),      // Task struct size
    sizeof(shareds),         // Shared data size
    task_entry);             // Task function

// Add dependencies to task
for (auto &dep : dependencies) {
  __kmpc_omp_task_with_deps(
      &loc,                  // Source location
      gtid,                  // Global thread ID
      task,                  // Task descriptor
      ndeps,                 // Number of dependencies
      dep_list,              // Array of dependency addresses
      ndeps_noalias,         // Number of no-alias dependencies
      noalias_dep_list);     // No-alias dependencies
}

// Enqueue task for execution
__kmpc_omp_task(&loc, gtid, task);
```

**Dependency List Structure**:
```cpp
struct kmp_depend_info {
  kmp_intptr_t base_addr;    // Variable address
  size_t len;                // Variable size
  union {
    struct {
      bool in:1;             // Input dependency
      bool out:1;            // Output dependency
      bool mtx:1;            // Mutexinoutset
    } flags;
    uint8_t flag_bits;
  };
};
```

---

### Dependency Resolution and Task Scheduling

#### Runtime Dependency Tracking

**Data Structure**: Dependency hash table per task team

```cpp
// Conceptual runtime structure
struct DependencyGraph {
  std::unordered_map<void*, std::vector<Task*>> address_to_tasks;
  
  void add_dependency(Task *task, void *addr, DependKind kind) {
    // Track which tasks access which addresses
    address_to_tasks[addr].push_back(task);
  }
  
  bool can_execute(Task *task) {
    // Check if all dependencies satisfied
    for (auto &dep : task->dependencies) {
      if (has_conflicting_task(dep.address, dep.kind)) {
        return false;  // Must wait
      }
    }
    return true;  // Ready to execute
  }
};
```

#### Dependency Conflict Rules

| **New Task** | **Existing Task** | **Conflict?** | **New Task Waits?** |
|--------------|-------------------|---------------|---------------------|
| IN           | IN                | No            | No (concurrent)     |
| IN           | OUT               | Yes           | Yes                 |
| IN           | INOUT             | Yes           | Yes                 |
| OUT          | IN                | Yes           | Yes                 |
| OUT          | OUT               | Yes           | Yes                 |
| OUT          | INOUT             | Yes           | Yes                 |
| INOUT        | IN                | Yes           | Yes                 |
| INOUT        | OUT               | Yes           | Yes                 |
| INOUT        | INOUT             | Yes           | Yes                 |
| MUTEXINOUTSET| IN                | Yes           | Yes                 |
| MUTEXINOUTSET| OUT               | Yes           | Yes                 |
| MUTEXINOUTSET| INOUT             | Yes           | Yes                 |
| MUTEXINOUTSET| MUTEXINOUTSET     | Maybe         | Runtime decision    |

**Key Insight**: Only multiple IN dependencies can execute concurrently.

---

### Complete Example: Dependency Chain

```fortran
! Demonstration of all dependency types
program depend_complete
  use omp_lib
  integer :: a, b, c, d
  integer :: result(4)
  integer(omp_depend_kind) :: dep_obj
  
  a = 1
  
  ! Task 1: Initialize b (OUT)
  !$omp task depend(out: b)
    b = a * 2  ! b = 2
  !$omp end task
  
  ! Task 2: Read b, write c (IN b, OUT c)
  ! Waits for Task 1
  !$omp task depend(in: b) depend(out: c)
    c = b + 10  ! c = 12
  !$omp end task
  
  ! Task 3: Read b concurrently (IN b)
  ! Waits for Task 1, can run with Task 2
  !$omp task depend(in: b)
    result(1) = b * 3  ! result(1) = 6
  !$omp end task
  
  ! Task 4: Update c (INOUT c)
  ! Waits for Task 2
  !$omp task depend(inout: c)
    c = c * 2  ! c = 24
  !$omp end task
  
  ! Task 5: Read c, read b (IN b, c)
  ! Waits for Task 4 (for c) and Task 1 (for b)
  !$omp task depend(in: b, c)
    result(2) = b + c  ! result(2) = 26
  !$omp end task
  
  ! Create reusable dependency object
  !$omp depobj(dep_obj) depend(inout: d)
  
  ! Task 6: Use dependency object
  !$omp task depend(depobj: dep_obj)
    d = 100
  !$omp end task
  
  ! Task 7: Reuse same dependency
  !$omp task depend(depobj: dep_obj)
    result(3) = d + 50  ! Waits for Task 6
  !$omp end task
  
  ! Wait for all tasks
  !$omp taskwait
  
  ! Destroy dependency object
  !$omp depobj(dep_obj) destroy
  
  print *, "Results:", result(1:3)
  ! Output: Results: 6 26 150

end program
```

**Execution Order Analysis**:
```
Time  | Executing Tasks           | Waiting Tasks
------|---------------------------|------------------
T0    | Task 1 (OUT b)            | 2,3 (wait for b)
T1    | Task 2 (IN b, OUT c)      | 4 (wait for c)
      | Task 3 (IN b)             |
T2    | Task 4 (INOUT c)          | 5 (wait for c)
T3    | Task 5 (IN b, c)          |
T4    | Task 6 (INOUT d)          | 7 (wait for d)
T5    | Task 7 (INOUT d)          |
```

---

### DEPOBJ Complete Walkthrough

**Use Case**: Dynamic dependency creation and reuse

```fortran
subroutine process_pipeline(data, n)
  use omp_lib
  integer, intent(inout) :: data(n)
  integer, intent(in) :: n
  integer(omp_depend_kind) :: stage_deps(10)
  integer :: i
  
  ! Create dependency objects for pipeline stages
  do i = 1, 10
    !$omp depobj(stage_deps(i)) depend(inout: data)
  end do
  
  ! Execute pipeline with dependencies
  do i = 1, 10
    !$omp task depend(depobj: stage_deps(i))
      call pipeline_stage(data, i)
    !$omp end task
  end do
  
  !$omp taskwait
  
  ! Destroy dependency objects
  do i = 1, 10
    !$omp depobj(stage_deps(i)) destroy
  end do
  
end subroutine
```

**Lowering**:
```mlir
// Initialize depobj
omp.depobj depobj(%stage_dep : !fir.ref<i64>) 
    depend(inout: %data : !fir.ref<!fir.array<?xi32>>)

// Use depobj in task
omp.task depend(depobj: %stage_dep : !fir.ref<i64>) {
  // task body
}

// Destroy depobj
omp.depobj depobj(%stage_dep : !fir.ref<i64>) destroy
```

---

### Performance and Optimization Notes

**1. Dependency Granularity**:
```fortran
! Fine-grained (better parallelism, more overhead)
do i = 1, 100
  !$omp task depend(inout: array(i))
    array(i) = process(array(i))
  !$omp end task
end do

! Coarse-grained (less overhead, less parallelism)
!$omp task depend(inout: array)
  do i = 1, 100
    array(i) = process(array(i))
  end do
!$omp end task
```

**2. Minimize False Dependencies**:
```fortran
! Bad: Entire array dependency
!$omp task depend(inout: array)
  call process_first_half(array(1:50))
!$omp end task
!$omp task depend(inout: array)  ! Waits unnecessarily
  call process_second_half(array(51:100))
!$omp end task

! Good: Section dependencies
!$omp task depend(inout: array(1:50))
  call process_first_half(array(1:50))
!$omp end task
!$omp task depend(inout: array(51:100))  ! Can run concurrently
  call process_second_half(array(51:100))
!$omp end task
```

**3. Use MUTEXINOUTSET for Potential Concurrency**:
```fortran
! Runtime can execute concurrently if safe
do i = 1, num_chunks
  !$omp task depend(mutexinoutset: sparse_matrix)
    call update_chunk(sparse_matrix, i)
  !$omp end task
end do
```

**4. Reuse DEPOBJ for Common Patterns**:
```fortran
! Initialize once
!$omp depobj(read_dep) depend(in: shared_data)

! Reuse many times
do i = 1, 1000
  !$omp task depend(depobj: read_dep)
    call read_task(shared_data, i)
  !$omp end task
end do

! Cleanup
!$omp depobj(read_dep) destroy
```

---

### Common Pitfalls and Solutions

**Pitfall 1: Forgetting to handle array sections in lowering**

```cpp
// WRONG: Only handle simple variables
if (const auto *name = std::get_if<parser::Name>(&designator.u)) {
  addr = getSymbolAddress(*name->symbol);
}
// BUG: Array sections not handled!

// CORRECT: Handle all designator forms
if (const auto *name = std::get_if<parser::Name>(&designator.u)) {
  addr = getSymbolAddress(*name->symbol);
} else if (const auto *arrayElem = std::get_if<parser::ArrayElement>(&designator.u)) {
  addr = genArrayElementAddr(*arrayElem);
} else if (const auto *structComp = std::get_if<parser::StructureComponent>(&designator.u)) {
  addr = genStructComponentAddr(*structComp);
}
```

**Pitfall 2: Not pairing dependency types correctly**

```fortran
! WRONG: Variable appears in both IN and OUT
!$omp task depend(in: x) depend(out: x)  ! Conflict!
```

Should be:
```fortran
! CORRECT: Use INOUT for read-modify-write
!$omp task depend(inout: x)
```

**Pitfall 3: Incorrect DEPOBJ type**

```fortran
! WRONG: Regular integer
integer :: dep_obj  ! Should be integer(omp_depend_kind)

! CORRECT:
use omp_lib
integer(omp_depend_kind) :: dep_obj
```

**Pitfall 4: Not handling DEPOBJ in MLIR translation**

```cpp
// Must special-case DEPOBJ type in lowering
if (depType == omp::DependType::depobj) {
  // Variable is already a dependency object, not a data address
  // Handle differently from IN/OUT/INOUT
}
```

---

### Testing Checklist

**Parser Tests** (`flang/test/Parser/OpenMP/`):
- [ ] All dependency types (IN, OUT, INOUT, MUTEXINOUTSET, DEPOBJ)
- [ ] Multiple dependencies on one task
- [ ] Array sections as dependencies
- [ ] Structure components as dependencies
- [ ] DEPOBJ directive (depend, destroy, update)

**Semantic Tests** (`flang/test/Semantics/OpenMP/`):
- [ ] Error: DEPEND on non-task construct
- [ ] Error: Threadprivate variable in DEPEND
- [ ] Error: Wrong type for DEPOBJ variable
- [ ] Error: SOURCE/SINK deprecation warnings
- [ ] Valid: Multiple tasks with dependencies

**Lowering Tests** (`flang/test/Lower/OpenMP/`):
- [ ] Correct MLIR dependency type attributes
- [ ] Proper variable address generation
- [ ] Array section dependency lowering
- [ ] DEPOBJ operation generation

**LLVM IR Tests** (`flang/test/Lower/OpenMP/`):
- [ ] Calls to `__kmpc_omp_task_with_deps`
- [ ] Correct dependency list structure
- [ ] Proper dependency kind encoding

---

### Summary Table: DEPEND Types

| **Type** | **Waits For** | **Blocks** | **Concurrent?** | **Use Case** |
|----------|---------------|------------|-----------------|--------------|
| **IN** | OUT, INOUT on same var | (none) | Yes (multiple IN) | Read-only access |
| **OUT** | IN, OUT, INOUT on same var | All subsequent | No | Write-only, overwrite |
| **INOUT** | IN, OUT, INOUT on same var | All subsequent | No | Read-modify-write |
| **MUTEXINOUTSET** | IN, OUT, INOUT on same var | (runtime decides) | Maybe | Potential non-overlap |
| **DEPOBJ** | (depends on stored type) | (depends on stored type) | (depends on stored type) | Reusable dependencies |

---

### Files Modified Summary

| **Component** | **Files** | **Key Implementation** |
|---------------|-----------|------------------------|
| **Parser** | `parse-tree.h` | `OmpClause::Depend`, `OmpDependenceType`, `OmpDepobj` |
| | `openmp-parsers.cpp` | Parse dependency types, variable lists, DEPOBJ directive |
| **Semantics** | `check-omp-structure.cpp` | Validate dependency types, check allowed constructs, verify DEPOBJ type |
| **MLIR Dialect** | `OpenMPOps.td` | `DependType` enum, `depend_vars`/`depend_types` in TaskOp, `DepobjOp` |
| | `OpenMPDialect.cpp` | Custom printer/parser for depend clause |
| **Lowering** | `Clauses.cpp` | `processDepend()` - convert parse tree to MLIR |
| | `OpenMP.cpp` | `genTaskOp()`, `genDepobjOp()` - create MLIR operations |
| **LLVM Translation** | `OpenMPToLLVMIRTranslation.cpp` | `convertOmpTaskOp()` - generate `__kmpc_omp_task_with_deps` calls |

---

## Asking for Help

### Where to Get Help

**LLVM Discourse (Recommended):**
- [Flang Discourse](https://discourse.llvm.org/c/subprojects/flang/35)
- Tag posts with `flang` and `openmp`
- Include minimal reproducible example

**GitHub Issues:**
- [LLVM Project Issues](https://github.com/llvm/llvm-project/issues)
- Search existing issues before creating new ones
- Use labels: `flang`, `openmp`, `bug`, `feature-request`

**IRC/Discord:**
- `#flang` on LLVM Discord server
- Active during US/EU business hours

### How to Ask Good Questions

**Bad Question:**
> "My OpenMP code doesn't work. Help!"

**Good Question:**
> "I'm implementing the AFFINITY clause for TASK directive. The parser accepts the syntax, but I'm getting an error during lowering: 'unhandled clause in genTaskClauses'. I've added the clause to the parse tree (parse-tree.h:1234) and semantic checker (check-omp-structure.cpp:5678). What am I missing in the lowering stage? Here's the minimal test case: [code snippet]"

**Include in your question:**
1. What you're trying to implement
2. What you've already tried
3. Specific error messages with file/line numbers
4. Minimal reproducible test case (Fortran source)
5. LLVM/Flang version or commit hash

---

## Quick Reference: File Navigation

### "I need to..."

**"...add a new OpenMP clause to the parser"**
→ `flang/include/flang/Parser/parse-tree.h` + `flang/lib/Parser/openmp-parsers.cpp`

**"...add semantic validation for a clause"**
→ `flang/lib/Semantics/check-omp-structure.cpp`

**"...implement lowering for a directive"**
→ `flang/lib/Lower/OpenMP/OpenMP.cpp` + `flang/lib/Lower/OpenMP/Clauses.cpp`

**"...add a new MLIR operation"**
→ `mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td`

**"...implement MLIR → LLVM IR translation"**
→ `mlir/lib/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.cpp`

**"...understand how a directive is currently implemented"**
→ Search for directive name (e.g., `OMPD_parallel`) across codebase

**"...find OpenMP runtime function definitions"**
→ `openmp/runtime/src/` in LLVM project

---

# PART 1: COMPILER STACK OVERVIEW

## OpenMP Implementation Pipeline

```
Fortran Source
    ↓
[1] Parser (flang/lib/Parser/)
    → Recognizes OpenMP directives and clauses
    → Builds parse tree
    ↓
[2] Semantics (flang/lib/Semantics/)
    → Validates OpenMP usage
    → Checks clause restrictions
    → Resolves symbols in clauses
    ↓
[3] MLIR Dialect (mlir/include/mlir/Dialect/OpenMP/)
    → Defines operations for OpenMP constructs
    → Attributes for clauses
    ↓
[4] Lowering (flang/lib/Lower/OpenMP/)
    → Converts parse tree to MLIR ops
    → Maps clauses to MLIR attributes
    ↓
[5] LLVM Translation (mlir/lib/Target/LLVMIR/Dialect/OpenMP/)
    → Converts MLIR ops to LLVM IR
    → Generates OpenMP runtime calls
    ↓
LLVM IR → Codegen
```

## Key Files by Stage

**Parser**:
- `flang/lib/Parser/openmp-parsers.cpp` - Grammar definitions
- `flang/include/flang/Parser/parse-tree.h` - Parse tree nodes

**Semantics**:
- `flang/lib/Semantics/check-omp-structure.cpp` - Structural validation
- `flang/lib/Semantics/resolve-directives.cpp` - Symbol resolution
- `flang/lib/Semantics/resolve-names.cpp` - Name binding in clauses

**MLIR Dialect**:
- `mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td` - Operation definitions
- `mlir/include/mlir/Dialect/OpenMP/OpenMPClauseOperands.h` - Clause operands

**Lowering**:
- `flang/lib/Lower/OpenMP/OpenMP.cpp` - Main lowering logic
- `flang/lib/Lower/OpenMP/Clauses.cpp` - Clause processing
- `flang/lib/Lower/OpenMP/DataSharingProcessor.cpp` - Variable handling

**LLVM Translation**:
- `mlir/lib/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.cpp`
- `llvm/include/llvm/Frontend/OpenMP/OMPConstants.h` - Runtime constants

---

# PART 2: FEATURE IMPLEMENTATIONS

## Semantic Checks for OpenMP Clauses

### Feature #173056: INIT Clause on DEPOBJ + Depinfo Modifier
**PR**: [#173056](https://github.com/llvm/llvm-project/pull/173056)  
**OpenMP Version**: 5.2+  
**Description**: Allow INIT clause on DEPOBJ directive, add support for depinfo modifier in DEPEND clause  
**Files Modified**:
- `flang/lib/Semantics/check-omp-structure.cpp` - Added INIT clause validation for DEPOBJ
- `flang/lib/Semantics/resolve-directives.cpp` - Handle depinfo modifier  
**Test**: `flang/test/Semantics/OpenMP/depobj-init.f90`  
**Learned**: DEPOBJ construct requires special handling since it manages dependency objects

### Feature #172036: COMBINER Clause Implementation
**PR**: [#172036](https://github.com/llvm/llvm-project/pull/172036)  
**OpenMP Version**: 6.0  
**Description**: Full implementation of COMBINER clause for reduction operations  
**Files Modified**:
- `flang/lib/Semantics/check-omp-structure.cpp` - Semantic validation
- `flang/include/flang/Parser/parse-tree.h` - Parse tree support
- `flang/lib/Parser/openmp-parsers.cpp` - Grammar  
**Validation Rules**:
- COMBINER must specify valid reduction operator
- Compatible with REDUCTION clause requirements  
**Test**: `flang/test/Semantics/OpenMP/combiner-clause.f90`  
**Learned**: Reduction clauses need operator validation and type checking

### Feature #172510: Assumed-Size Array Diagnostics
**PR**: [#172510](https://github.com/llvm/llvm-project/pull/172510)  
**Description**: Diagnose whole assumed-size arrays on MAP and DEPEND clauses  
**Files Modified**:
- `flang/lib/Semantics/check-omp-structure.cpp` - Added array extent checking  
**Restriction**: Whole assumed-size arrays (`array(*)`) not allowed on certain clauses due to unknown bounds  
**Error Message**: "Assumed-size array 'array' may not appear as whole array in MAP/DEPEND clause"  
**Test**: `flang/test/Semantics/OpenMP/assumed-size-array.f90`  
**Standard Reference**: OpenMP 5.2 map/depend clause restrictions  
**Learned**: Array shape information critical for data mapping validation

### Feature #171454: DIMS Modifier Frontend Support
**PR**: [#171454](https://github.com/llvm/llvm-project/pull/171454)  
**OpenMP Version**: 6.0  
**Description**: Frontend (parser + semantics) support for DIMS modifier on REDUCTION clause  
**Files Modified**:
- `flang/lib/Parser/openmp-parsers.cpp` - Parse DIMS syntax
- `flang/lib/Semantics/check-omp-structure.cpp` - Validate DIMS usage
- `flang/include/flang/Parser/parse-tree.h` - Add DIMS node  
**Syntax**: `REDUCTION(DIMS=dimension-list : operator : list)`  
**Validation**: Dimension list must be valid array dimension references  
**Test**: `flang/test/Parser/OpenMP/dims-modifier.f90`  
**Learned**: Modifiers extend clause functionality and require integrated parser/semantic support

### Feature #170351: GetOmpObjectList Expansion
**PR**: [#170351](https://github.com/llvm/llvm-project/pull/170351)  
**Description**: Expand GetOmpObjectList helper to handle all OmpClause subclasses uniformly  
**Files Modified**:
- `flang/lib/Semantics/check-omp-structure.cpp` - Refactored object list extraction  
**Impact**: Enables uniform processing of object lists across clause types (PRIVATE, SHARED, REDUCTION, etc.)  
**Learned**: Helper functions for common clause patterns improve code maintainability

### Feature #167806: Fix defaultmap(none) Over-Aggressive Checks
**PR**: [#167806](https://github.com/llvm/llvm-project/pull/167806)  
**Issue**: `defaultmap(none)` was incorrectly flagging some symbols  
**Files Modified**:
- `flang/lib/Semantics/check-omp-structure.cpp` - Refined scope analysis for defaultmap  
**Fix**: Consider implicit data-sharing rules before reporting errors  
**Test**: `flang/test/Semantics/OpenMP/defaultmap-none.f90`  
**Learned**: Data-sharing attribute determination involves complex scoping rules

### Feature #166214: DYN_GROUPPRIVATE Semantic Checks
**PR**: [#166214](https://github.com/llvm/llvm-project/pull/166214)  
**OpenMP Version**: 6.0  
**Description**: Semantic validation for DYN_GROUPPRIVATE clause  
**Files Modified**:
- `flang/lib/Semantics/check-omp-structure.cpp` - Added validation rules  
**Restrictions**:
- Variables must have appropriate scope
- Cannot appear with conflicting data-sharing clauses  
**Test**: `flang/test/Semantics/OpenMP/dyn-groupprivate.f90`  
**Learned**: Dynamic group privatization requires careful scope tracking

### Feature #167296: Derived Type Array Elements Semantics
**PR**: [#167296](https://github.com/llvm/llvm-project/pull/167296)  
**Description**: Improved semantic handling of derived type array element references in OpenMP clauses  
**Files Modified**:
- `flang/lib/Semantics/resolve-directives.cpp` - Enhanced component reference resolution  
**Examples**: `type_array(:)%component`, `type_array(i)%nested%field`  
**Test**: `flang/test/Semantics/OpenMP/derived-type-arrays.f90`  
**Learned**: Component references in clauses need special symbol resolution handling

---

## Parser Support for OpenMP Syntax

### Parser Implementation Pattern Reference

**Key Files for Parser Changes:**
1. **`llvm/include/llvm/Frontend/OpenMP/OMPKinds.def`** - Register directive/clause kinds
2. **`flang/include/flang/Parser/parse-tree.h`** - Define parse tree structures
3. **`flang/lib/Parser/openmp-parsers.cpp`** - Implement grammar rules

**Parser Pattern Example - Adding NEWCLAUSE:**

```cpp
// 1. Register in OMPKinds.def
__OMP_CLAUSE(newclause, OMPC_newclause)

// 2. Define parse tree node in parse-tree.h
struct OmpNewClauseModifier {
  ENUM_CLASS(Kind, Modifier1, Modifier2)
  WRAPPER_CLASS_BOILERPLATE(OmpNewClauseModifier, Kind);
};
EMPTY_CLASS(OmpClause::NewClause);
// For clauses with modifiers:
struct OmpClause::NewClause {
  TUPLE_CLASS_BOILERPLATE(OmpClause::NewClause);
  std::tuple<std::optional<OmpNewClauseModifier>, OmpObjectList> t;
};

// 3. Implement grammar in openmp-parsers.cpp
TYPE_PARSER(construct<OmpClause::NewClause>(
    "NEWCLAUSE" >>
    maybe(parenthesized(
        construct<OmpClause::NewClause>(
            maybe(construct<OmpNewClauseModifier>(
                Parser<OmpNewClauseModifier::Kind>{})) / ":"_tok,
            Parser<OmpObjectList>{}))) ||
    construct<OmpClause::NewClause>()))
```

**Common Parser Patterns:**

1. **Simple Keyword Clause** (e.g., NOWAIT):
```cpp
EMPTY_CLASS(OmpClause::Nowait);
TYPE_PARSER(construct<OmpClause::Nowait>("NOWAIT"_tok))
```

2. **Clause with Object List** (e.g., PRIVATE):
```cpp
struct OmpClause::Private {
  WRAPPER_CLASS_BOILERPLATE(OmpClause::Private, OmpObjectList);
};
TYPE_PARSER(construct<OmpClause::Private>(
    "PRIVATE" >> parenthesized(Parser<OmpObjectList>{})))
```

3. **Clause with Modifiers** (e.g., REDUCTION):
```cpp
struct OmpClause::Reduction {
  TUPLE_CLASS_BOILERPLATE(OmpClause::Reduction);
  std::tuple<std::optional<OmpReductionModifier>, 
             OmpReductionOperator, 
             OmpObjectList> t;
};
TYPE_PARSER(construct<OmpClause::Reduction>(
    "REDUCTION" >> parenthesized(
        maybe(Parser<OmpReductionModifier>{} / ":"_tok),
        Parser<OmpReductionOperator>{} / ":"_tok,
        Parser<OmpObjectList>{})))
```

4. **Clause with Scalar Expression** (e.g., IF):
```cpp
struct OmpClause::If {
  TUPLE_CLASS_BOILERPLATE(OmpClause::If);
  std::tuple<std::optional<OmpIfClauseModifier>, ScalarLogicalExpr> t;
};
TYPE_PARSER(construct<OmpClause::If>(
    "IF" >> parenthesized(
        maybe(Parser<OmpIfClauseModifier>{} / ":"_tok),
        scalarLogicalExpr)))
```

**Parse Tree Macros:**
- `EMPTY_CLASS(T)` - No-argument clauses (e.g., NOWAIT)
- `WRAPPER_CLASS_BOILERPLATE(T, Content)` - Single element wrapper
- `TUPLE_CLASS_BOILERPLATE(T)` - Multi-element tuple
- `UNION_CLASS_BOILERPLATE(T)` - Variant/union type

**Parser Combinator Operators:**
- `>>` - Sequence (must match in order)
- `||` - Alternative (try alternatives in order)
- `/` - Discard left, keep right result
- `maybe()` - Optional element (std::optional)
- `parenthesized()` - Wrapped in `()`
- `many()` - Zero or more repetitions
- `some()` - One or more repetitions
- `"TOKEN"_tok` - Literal token matching

**Testing Parser Changes:**
```bash
# Test parser output
llvm-lit flang/test/Parser/OpenMP/newclause.f90

# Check parse tree dump
flang-new -fc1 -fdebug-dump-parse-tree test.f90
```

---

### Feature #172080: OpenMP 6.0 Clause Definitions
**PR**: [#172080](https://github.com/llvm/llvm-project/pull/172080)  
**Description**: Define remaining OpenMP 6.0 clauses in parser, add Flang skeleton  
**Files Modified**:
- `flang/lib/Parser/openmp-parsers.cpp` - Grammar for new clauses
- `flang/include/flang/Parser/parse-tree.h` - Parse tree nodes
- `llvm/include/llvm/Frontend/OpenMP/OMPKinds.def` - Clause kind definitions  
**New Clauses**: Multiple OpenMP 6.0 clauses added (full skeleton for future implementation)  
**Approach**: Add parser support first, enable incremental semantic/lowering work  
**Learned**: Skeleton approach allows OpenMP spec tracking before full implementation

### Feature #161213: Loop Sequences and Loop Fuse Support
**PR**: [#161213](https://github.com/llvm/llvm-project/pull/161213)  
**Description**: Parser and semantic support for OpenMP loop sequences and LOOP directive with FUSE clause  
**Files Modified**:
- `flang/lib/Parser/openmp-parsers.cpp` - Parse loop sequences
- `flang/lib/Semantics/check-omp-loop.cpp` - Validate loop structure
- `flang/include/flang/Parser/parse-tree.h` - Loop sequence nodes  
**Syntax**: `!$OMP LOOP FUSE(n)`  
**Validation**: Ensure loop sequences meet OpenMP requirements  
**Test**: `flang/test/Semantics/OpenMP/loop-fuse.f90`  
**Learned**: Loop directive variations require specialized parsing and validation logic

---

## MLIR Dialect Operations

### Feature #172871: OmpDependenceKind Common Enum
**PR**: [#172871](https://github.com/llvm/llvm-project/pull/172871)  
**Description**: Make OmpDependenceKind a common enum shared between Flang and MLIR  
**Files Modified**:
- `mlir/include/mlir/Dialect/OpenMP/OpenMPOpsEnums.td` - Enum definition
- `flang/include/flang/Common/OpenMP-features.h` - Use common enum  
**Impact**: Eliminates duplicate definitions, ensures consistency  
**Values**: `in`, `out`, `inout`, `mutexinoutset`, `depobj`, `source`, `sink`  
**Learned**: Shared enums between frontend and dialect improve maintainability

### Feature: OpenMP Attribute Definitions
**Files**: `mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td`  
**Description**: MLIR attributes for OpenMP clauses  
**Key Attributes**:
- `ProcBindKindAttr` - Thread affinity (master, close, spread)
- `ScheduleKindAttr` - Loop scheduling (static, dynamic, guided, auto)
- `ReductionDeclareAttr` - Reduction operation metadata  
**Usage**: Attached to operations to represent clause semantics in MLIR  
**Learned**: Attributes provide type-safe clause representation in MLIR

---

## Lowering to MLIR OpenMP Dialect

### Feature #165719: ALLOCATE Directive Reorganization
**PR**: [#165719](https://github.com/llvm/llvm-project/pull/165719)  
**Description**: Reorganized and refactored semantic checks for ALLOCATE-related clauses  
**Files Modified**:
- `flang/lib/Lower/OpenMP/OpenMP.cpp` - Simplified lowering logic
- `flang/lib/Semantics/check-omp-structure.cpp` - Cleaner validation  
**Impact**: Better separation between ALLOCATE directive vs ALLOCATE clause on other directives  
**Learned**: Lowering benefits from well-organized semantic phase

### Feature #164420: ALLOCATE Directive Semantic Updates
**PR**: [#164420](https://github.com/llvm/llvm-project/pull/164420)  
**Description**: Refactor and update semantic checks for ALLOCATE directive  
**Files Modified**:
- `flang/lib/Semantics/check-omp-structure.cpp` - Updated validation
- `flang/lib/Lower/OpenMP/OpenMP.cpp` - Improved lowering  
**Validation**:
- Allocator must be valid OpenMP memory allocator
- Variables must be allocatable or pointer  
**Learned**: Allocator management requires coordination between semantics and lowering

### Feature: Data-Sharing Attribute Processing
**File**: `flang/lib/Lower/OpenMP/DataSharingProcessor.cpp`  
**Description**: Handles PRIVATE, FIRSTPRIVATE, LASTPRIVATE, SHARED, etc.  
**Process**:
1. Identify variables with data-sharing attributes
2. Create MLIR SSA values for private copies
3. Insert copy-in operations (firstprivate)
4. Insert copy-out operations (lastprivate)  
**Key Functions**:
- `processStep1()` - Collect variables
- `processStep2()` - Create local allocations  
**Learned**: Data-sharing lowering is multi-phase process requiring careful SSA value management

### Pattern: genParallelOp - Parallel Region Lowering
**Function**: `genParallelOp()` in `flang/lib/Lower/OpenMP/OpenMP.cpp`  
**Parse Tree**: `parser::OmpBeginBlockDirective` (PARALLEL)  
**MLIR Operation**: `omp.parallel`  
**Process**:
1. Process IF clause → conditional execution
2. Process NUM_THREADS clause → thread count
3. Process PRIVATE/SHARED clauses → DataSharingProcessor
4. Process REDUCTION clause → reduction symbols and operators
5. Process PROC_BIND clause → thread affinity
6. Create `omp.parallel` operation with processed clauses
7. Lower body statements inside parallel region  
**Key Code**:
```cpp
auto parallelOp = builder.create<mlir::omp::ParallelOp>(
    loc, ifClauseOperand, numThreadsClauseOperand,
    privateVars, reductionVars, procBindAttr);
```
**Learned**: Parallel regions require collecting all clause operands before creating MLIR op

### Pattern: genLoopOp - Worksharing Loop Lowering
**Function**: `genLoopOp()` in `flang/lib/Lower/OpenMP/OpenMP.cpp`  
**Parse Tree**: `parser::OmpLoopDirective` (DO, SIMD, etc.)  
**MLIR Operation**: `omp.wsloop` or `omp.simd`  
**Process**:
1. Extract loop bounds and step from DO construct
2. Process SCHEDULE clause → schedule kind and chunk size
3. Process COLLAPSE clause → determine collapsed loop nest depth
4. Process ORDERED clause → ordered iteration handling
5. Process REDUCTION/PRIVATE/LASTPRIVATE → variable handling
6. Create loop operation with iteration variable
7. Generate loop body with proper variable mappings  
**Schedule Kinds**: static, dynamic, guided, auto, runtime  
**Learned**: Loop lowering must handle Fortran-specific loop semantics (1-based indexing, step)

### Pattern: genSectionsOp - Sections Construct Lowering
**Function**: `genSectionsOp()` in `flang/lib/Lower/OpenMP/OpenMP.cpp`  
**Parse Tree**: `parser::OmpSectionsDirective`  
**MLIR Operations**: `omp.sections` containing `omp.section` ops  
**Process**:
1. Process NOWAIT clause → determine if barrier needed
2. Process PRIVATE/REDUCTION clauses
3. Create parent `omp.sections` operation
4. For each SECTION block:
   - Create nested `omp.section` operation
   - Lower section body statements
5. Implicit barrier unless NOWAIT specified  
**Learned**: Sections require nested operation structure (parent sections, child section ops)

### Pattern: genTaskOp - Task Construct Lowering
**Function**: `genTaskOp()` in `flang/lib/Lower/OpenMP/OpenMP.cpp`  
**Parse Tree**: `parser::OmpTaskDirective`  
**MLIR Operation**: `omp.task`  
**Process**:
1. Process IF clause → conditional task creation
2. Process FINAL clause → final task flag
3. Process PRIORITY clause → task priority
4. Process DEPEND clause → task dependencies (in/out/inout)
5. Process PRIVATE/FIRSTPRIVATE → captured variables
6. Create task operation with all operands
7. Lower task body inside operation region  
**Task Dependencies**: Represented as dependence operands with type (in/out/inout/mutexinoutset)  
**Learned**: Task lowering requires careful handling of captured variables and dependencies

### Pattern: genSingleOp - Single Construct Lowering
**Function**: `genSingleOp()` in `flang/lib/Lower/OpenMP/OpenMP.cpp`  
**Parse Tree**: `parser::OmpSingleDirective`  
**MLIR Operation**: `omp.single`  
**Process**:
1. Process PRIVATE/FIRSTPRIVATE clauses
2. Process COPYPRIVATE clause → variable broadcasting after single region
3. Process NOWAIT clause
4. Create single operation
5. Lower single-region body  
**COPYPRIVATE Handling**: Special broadcast mechanism from executing thread to other threads  
**Learned**: Single construct has unique COPYPRIVATE clause requiring broadcast semantics

### Pattern: genCriticalOp - Critical Section Lowering
**Function**: `genCriticalOp()` in `flang/lib/Lower/OpenMP/OpenMP.cpp`  
**Parse Tree**: `parser::OmpCriticalDirective`  
**MLIR Operation**: `omp.critical`  
**Process**:
1. Extract critical section name (optional)
2. Process HINT clause → lock hint attribute
3. Create critical operation with name symbol
4. Lower critical section body
5. Generate implicit lock acquisition/release  
**Named vs Unnamed**: Named criticals use different locks, unnamed share global lock  
**Learned**: Critical sections need unique naming to avoid unintended mutual exclusion

### Pattern: genAtomicOp - Atomic Operation Lowering
**Functions**: `genAtomicRead()`, `genAtomicWrite()`, `genAtomicUpdate()`, `genAtomicCapture()`  
**Parse Tree**: `parser::OmpAtomicDirective`  
**MLIR Operations**: `omp.atomic.read`, `omp.atomic.write`, `omp.atomic.update`, `omp.atomic.capture`  
**Process**:
1. Determine atomic operation type (read/write/update/capture)
2. Process MEMORY_ORDER clause (seq_cst, acquire, release, relaxed)
3. Extract memory location and value expressions
4. For UPDATE/CAPTURE: identify atomic operation (add, sub, mul, etc.)
5. Create appropriate atomic operation with memory order attribute  
**Memory Orders**: Translate Fortran memory order to MLIR attributes  
**Learned**: Atomic operations require precise identification of operation type and memory semantics

### Pattern: genTargetOp - Target Directive Lowering
**Function**: `genTargetOp()` in `flang/lib/Lower/OpenMP/OpenMP.cpp`  
**Parse Tree**: `parser::OmpTargetDirective`  
**MLIR Operation**: `omp.target`  
**Process**:
1. Process DEVICE clause → target device ID
2. Process MAP clause → data mapping (to/from/tofrom/alloc)
3. Process IF clause → conditional offloading
4. Process DEPEND clause → inter-task dependencies
5. Process PRIVATE/FIRSTPRIVATE for device variables
6. Create target operation with map operands
7. Lower target region body  
**Map Types**: Must specify direction (to/from/tofrom) and allocation behavior  
**Learned**: Target directives require complex data mapping between host and device

---

## LLVM IR Translation

### Feature: Parallel Region Translation
**File**: `mlir/lib/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.cpp`  
**MLIR Op**: `omp.parallel`  
**LLVM IR**: Calls to `__kmpc_fork_call()` runtime function  
**Translation**:
- Convert parallel region to outlined function
- Pass shared variables as function arguments
- Generate thread team invocation  
**Learned**: Parallel regions require function outlining at LLVM IR level

### Feature: Worksharing Loop Translation
**MLIR Op**: `omp.wsloop`  
**LLVM IR**: 
- `__kmpc_for_static_init()` - Static scheduling
- `__kmpc_dispatch_init()` / `__kmpc_dispatch_next()` - Dynamic scheduling  
**Translation**:
- Compute loop bounds for each thread
- Insert synchronization calls
- Handle reduction operations  
**Learned**: Different schedule kinds translate to different runtime entry points

### Feature: Task Translation
**MLIR Op**: `omp.task`  
**LLVM IR**: `__kmpc_omp_task_alloc()` + `__kmpc_omp_task()`  
**Translation**:
- Allocate task descriptor
- Capture firstprivate variables
- Enqueue task for execution  
**Learned**: Task translation requires capturing clause data in task descriptor structure

### Pattern: convertOmpSections - Sections Construct Translation
**Function**: `convertOmpSections()` in `OpenMPToLLVMIRTranslation.cpp`  
**MLIR Op**: `omp.sections` containing `omp.section`  
**LLVM IR**:
- `__kmpc_for_static_init()` - Distribute sections to threads
- `__kmpc_for_static_fini()` - Cleanup after sections
- `__kmpc_barrier()` - Implicit barrier (unless NOWAIT)  
**Translation Process**:
1. Convert sections to static loop over section indices
2. Each thread executes subset of sections
3. Switch statement dispatches to correct section body
4. Insert barrier unless NOWAIT specified  
**Learned**: Sections are implemented as static iteration space over section indices

### Pattern: convertOmpBarrier - Barrier Translation
**MLIR Op**: `omp.barrier`  
**LLVM IR**: `__kmpc_barrier(&loc, global_tid)`  
**Translation**:
- Get current location info
- Get global thread ID
- Call runtime barrier function
- All threads in team wait at this point  
**Implicit Barriers**: Inserted automatically at end of parallel/worksharing without NOWAIT  
**Learned**: Barrier is simplest translation - single runtime call with location and thread ID

### Pattern: convertOmpCritical - Critical Section Translation
**MLIR Op**: `omp.critical`  
**LLVM IR**: `__kmpc_critical(&loc, global_tid, &lock)` + `__kmpc_end_critical(&loc, global_tid, &lock)`  
**Translation**:
1. Generate or retrieve named lock variable (global)
2. Call `__kmpc_critical()` before region (acquires lock)
3. Translate critical region body
4. Call `__kmpc_end_critical()` after region (releases lock)  
**Named Locks**: Different names → different locks, unnamed → shared default lock  
**Learned**: Critical sections translate to runtime lock acquisition/release with persistent lock objects

### Pattern: convertOmpAtomic - Atomic Operation Translation
**MLIR Ops**: `omp.atomic.read`, `omp.atomic.write`, `omp.atomic.update`, `omp.atomic.capture`  
**LLVM IR**: `atomicrmw`, `cmpxchg`, or `load`/`store` with atomic ordering  
**Translation**:
- **READ**: `load atomic` with specified memory order
- **WRITE**: `store atomic` with specified memory order
- **UPDATE**: `atomicrmw` (add/sub/mul/min/max/and/or/xor) with memory order
- **CAPTURE**: Combination of `atomicrmw` and load to capture old/new value  
**Memory Orders**: seq_cst → `seq_cst`, acquire → `acquire`, release → `release`, relaxed → `monotonic`  
**Learned**: Atomic operations map directly to LLVM atomic instructions with memory ordering

### Pattern: convertOmpReduction - Reduction Translation
**Clause**: `reduction(operator:variables)` on parallel/loop constructs  
**LLVM IR**:
- Thread-local reduction variables (private storage)
- `__kmpc_reduce()` or `__kmpc_reduce_nowait()` - Initiate reduction
- Switch on return value for reduction strategy
- Atomic or critical reduction into shared variable
- `__kmpc_end_reduce()` or `__kmpc_end_reduce_nowait()`  
**Reduction Strategies**:
- Return 1: Use atomic operations
- Return 2: Use critical section
- Return 0: Already reduced (by runtime)  
**Translation Process**:
1. Create private copy for each thread
2. Perform local reduction in parallel region
3. Call reduction runtime to coordinate
4. Combine private copies into final result
5. End reduction phase  
**Learned**: Reduction is complex multi-phase process with runtime coordination and strategy selection

### Pattern: convertOmpTaskloop - Taskloop Translation
**Function**: `convertOmpTaskloopOp()` in `OpenMPToLLVMIRTranslation.cpp`  
**MLIR Op**: `omp.taskloop`  
**LLVM IR**: `__kmpc_taskloop()` - Single call for task loop iteration space  
**Translation**:
1. Create task function (outlined loop body)
2. Compute loop bounds and step
3. Process GRAINSIZE or NUM_TASKS clause
4. Call `__kmpc_taskloop()` with:
   - Task function pointer
   - Loop bounds and step
   - Grainsize or num_tasks
   - Task flags (untied, final, etc.)  
**Learned**: Taskloop uses special runtime function that creates and schedules multiple tasks automatically

### Pattern: convertOmpTaskgroup - Taskgroup Translation
**Function**: `convertOmpTaskgroupOp()` in `OpenMPToLLVMIRTranslation.cpp`  
**MLIR Op**: `omp.taskgroup`  
**LLVM IR**: `__kmpc_taskgroup(&loc, global_tid)` + `__kmpc_end_taskgroup(&loc, global_tid)`  
**Translation**:
1. Call `__kmpc_taskgroup()` to start taskgroup
2. Translate taskgroup region body (which may spawn tasks)
3. Call `__kmpc_end_taskgroup()` to wait for all child tasks  
**Semantics**: End taskgroup waits for all tasks spawned in the region to complete  
**Learned**: Taskgroup provides structured task synchronization point

### Pattern: convertOmpTaskwait - Taskwait Translation
**Function**: `convertOmpTaskwaitOp()` in `OpenMPToLLVMIRTranslation.cpp`  
**MLIR Op**: `omp.taskwait`  
**LLVM IR**: `__kmpc_omp_taskwait(&loc, global_tid)`  
**Translation**:
- Single runtime call
- Waits for all child tasks of current task to complete
- Simpler than taskgroup (only immediate children)  
**Difference from Barrier**: Taskwait only waits for tasks, barrier synchronizes all threads  
**Learned**: Task synchronization primitives (taskwait/taskgroup) are distinct from thread synchronization (barrier)

### Pattern: convertOmpFlush - Flush Translation
**MLIR Op**: `omp.flush`  
**LLVM IR**: `__kmpc_flush(&loc)` or memory fence instructions  
**Translation**:
1. If flush list specified: emit memory fence for those variables
2. If no list: flush all shared variables (full fence)
3. Call runtime flush function or use LLVM `fence` instruction  
**Memory Consistency**: Ensures memory operations before flush visible to other threads  
**Learned**: Flush translates to memory fences for cross-thread visibility

### Pattern: convertOmpOrdered - Ordered Translation
**MLIR Op**: `omp.ordered` (block) or `omp.ordered.region`  
**LLVM IR**: `__kmpc_ordered(&loc, global_tid)` + `__kmpc_end_ordered(&loc, global_tid)`  
**Translation**:
1. Call `__kmpc_ordered()` before ordered region
2. Translate ordered block (executed in loop iteration order)
3. Call `__kmpc_end_ordered()` after region  
**Usage**: Ensures code inside ordered region executes in sequential loop iteration order within parallel loop  
**Learned**: Ordered provides sequential execution guarantee within parallel loop context

### Pattern: convertOmpMaster - Master Translation
**MLIR Op**: `omp.master`  
**LLVM IR**: Conditional check on `omp_get_thread_num() == 0`  
**Translation**:
1. Get current thread number
2. Generate if statement checking for master thread (thread 0)
3. Translate master region body inside conditional
4. No runtime call needed (simple thread ID check)  
**Learned**: Master region is lightweight - just conditional execution, no synchronization

### Pattern: Reduction Operator Translation
**Reduction Operators**: +, *, -, .and., .or., .eqv., .neqv., max, min, iand, ior, ieor  
**LLVM Mapping**:
- `+` → `atomicrmw add` or `fadd`
- `*` → `atomicrmw mul` (integer) or `fmul` (float)
- `max`/`min` → `atomicrmw max`/`min`
- `.and.`/`.or.` → `and`/`or` operations
- `iand`/`ior`/`ieor` → `atomicrmw and`/`or`/`xor`  
**Custom Reductions**: User-defined reduction operators require function calls in combine phase  
**Learned**: Most intrinsic reductions map to LLVM atomic operations, custom reductions need function outlining

---

## Device Constructs and Offload Runtime Interaction

### Overview: OpenMP Device Execution Model

**Device Constructs** enable offloading computation to accelerators (GPUs, FPGAs, etc.). The key directives are:
- **TARGET**: Offload region to device
- **TEAMS**: Create league of thread teams on device
- **DISTRIBUTE**: Distribute loop iterations across teams
- **Combined**: TARGET TEAMS, TARGET TEAMS DISTRIBUTE, etc.

**Runtime**: `libomptarget` - OpenMP offload runtime library
**Device Plugins**: Architecture-specific plugins (CUDA, HIP, Level Zero, etc.)

---

### Architecture: Host-Device Interaction

```
┌─────────────────────────────────────────────────────────────────┐
│                         HOST (CPU)                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Flang Compiler                                          │   │
│  │  - Parse TARGET/TEAMS directives                         │   │
│  │  - Identify device code regions                          │   │
│  │  - Lower to MLIR omp.target operation                    │   │
│  │  - Generate device kernel outline                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  MLIR → LLVM IR Translation                              │   │
│  │  - Separate device code into kernel module               │   │
│  │  - Generate host-side runtime calls                      │   │
│  │  - Create device binary embedding                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Host Code (LLVM IR)          Device Kernel (LLVM IR)    │   │
│  │  __tgt_target_kernel()  ←───→  Compiled to PTX/AMDGCN   │   │
│  │  __tgt_target_data_begin()    Embedded in executable     │   │
│  │  __tgt_target_data_end()                                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Runtime Execution (libomptarget)                        │   │
│  │  1. Initialize device                                    │   │
│  │  2. Map data to device memory                            │   │
│  │  3. Launch kernel                                        │   │
│  │  4. Map data back to host                                │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                       DEVICE (GPU)                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Device Plugin (CUDA/HIP/Level Zero)                     │   │
│  │  - Load kernel binary                                    │   │
│  │  - Configure grid/block dimensions                       │   │
│  │  - Launch kernel                                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Kernel Execution                                        │   │
│  │  - TEAMS → Thread blocks/workgroups                      │   │
│  │  - DISTRIBUTE → Iterations across teams                  │   │
│  │  - PARALLEL → Threads within team                        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

### TARGET Directive: Complete Pipeline

#### Fortran Source
```fortran
program target_example
  integer :: n = 1024
  real :: a(n), b(n), c(n)
  
  ! Initialize on host
  a = 1.0
  b = 2.0
  
  !$omp target map(to: a, b) map(from: c)
    !$omp parallel do
    do i = 1, n
      c(i) = a(i) + b(i)
    end do
  !$omp end target
  
  print *, "Result:", c(1:10)
end program
```

---

#### Stage 1: Parser → Parse Tree

**File**: `flang/lib/Parser/openmp-parsers.cpp`

The parser recognizes:
```
OmpBeginBlockDirective: TARGET
OmpClauseList: MAP(TO: a, b), MAP(FROM: c)
Block: nested parallel do construct
OmpEndBlockDirective: END TARGET
```

**Parse Tree Structure**:
```cpp
struct OpenMPBlockConstruct {
  OmpBeginBlockDirective beginDirective;  // TARGET
  Block block;                            // nested constructs
  OmpEndBlockDirective endDirective;      // END TARGET
};

// MAP clause parsed as:
struct OmpClause::Map {
  OmpMapType::Type mapType;        // TO, FROM, TOFROM, ALLOC, etc.
  OmpObjectList objects;           // a, b, c
};
```

---

#### Stage 2: Semantic Analysis

**File**: `flang/lib/Semantics/check-omp-structure.cpp`

```cpp
void OmpStructureChecker::Enter(const parser::OpenMPBlockConstruct &x) {
  const auto &beginDir = std::get<parser::OmpBeginBlockDirective>(x.t);
  
  if (beginDir.v == llvm::omp::Directive::OMPD_target) {
    PushContext(beginDir.source, llvm::omp::Directive::OMPD_target);
    
    // Validate TARGET-specific restrictions
    CheckTargetNesting();
    CheckDeviceClausesAllowed();
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Map &map) {
  CheckAllowed(llvm::omp::Clause::OMPC_map);
  
  auto [mapTypeModifiers, mapType, mappers, objects] = map.t;
  
  // Validate mapped variables
  for (const auto &obj : objects.v) {
    const auto *symbol = ResolveOmp(*obj, SymbolFlag::ObjectName, context_);
    
    if (!symbol) {
      context_.Say(obj->source,
          "Variable in MAP clause must be declared"_err_en_US);
      continue;
    }
    
    // Check variable is mappable (no ALLOCATABLE THREADPRIVATE, etc.)
    if (symbol->test(Symbol::Flag::OmpThreadprivate)) {
      context_.Say(obj->source,
          "Threadprivate variable '%s' cannot be mapped"_err_en_US,
          symbol->name().ToString());
    }
    
    // Check for procedure pointers (not mappable)
    if (IsProcedurePointer(*symbol)) {
      context_.Say(obj->source,
          "Procedure pointers cannot be mapped to device"_err_en_US);
    }
    
    // Warn about assumed-size arrays (runtime error)
    if (IsAssumedSizeArray(*symbol)) {
      context_.Say(obj->source,
          "Mapping assumed-size array may fail at runtime"_warn_en_US);
    }
  }
}

void OmpStructureChecker::CheckTargetNesting() {
  // TARGET cannot be nested inside another TARGET
  if (auto *enclosing = GetEnclosingContext()) {
    if (enclosing->directive == llvm::omp::Directive::OMPD_target) {
      context_.Say(GetContext().directiveSource,
          "TARGET construct cannot be nested inside another TARGET"_err_en_US);
    }
  }
}
```

**Key Validations**:
- Variables in MAP clause must be declared
- Threadprivate variables cannot be mapped
- Procedure pointers are not mappable
- No nested TARGET regions
- MAP clause only on TARGET/TARGET DATA directives

---

#### Stage 3: MLIR Dialect - Target Operation

**File**: `mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td`

```tablegen
def TargetOp : OpenMP_Op<"target", [AttrSizedOperandSegments]> {
  let summary = "target construct";
  let description = [{
    Offloads execution of the region to a device. The device may be a GPU,
    FPGA, or other accelerator. Variables are mapped to device memory
    according to MAP clauses.
    
    Example:
    ```mlir
    omp.target map_entries(%map_a, %map_b -> %arg0, %arg1 : !fir.ref<f32>, !fir.ref<f32>) {
      // Device code here
      omp.terminator
    }
    ```
  }];
  
  let arguments = (ins
    Optional<AnyType>:$if_expr,
    Optional<AnyType>:$device,
    Optional<AnyType>:$thread_limit,
    Variadic<AnyType>:$map_vars,           // Variables to map
    OptionalAttr<MapTypesAttr>:$map_types  // Map types (TO/FROM/TOFROM)
  );
  
  let regions = (region AnyRegion:$region);
  
  let assemblyFormat = [{
    (`if` `(` $if_expr^ `:` type($if_expr) `)`)?
    (`device` `(` $device^ `:` type($device) `)`)?
    (`thread_limit` `(` $thread_limit^ `:` type($thread_limit) `)`)?
    (`map_entries` `(` $map_vars^ `:` type($map_vars) `)`)?
    $region attr-dict
  }];
}

// Map types enumeration
def MapType_TO      : I64EnumAttrCase<"TO", 0>;
def MapType_FROM    : I64EnumAttrCase<"FROM", 1>;
def MapType_TOFROM  : I64EnumAttrCase<"TOFROM", 2>;
def MapType_ALLOC   : I64EnumAttrCase<"ALLOC", 3>;
def MapType_RELEASE : I64EnumAttrCase<"RELEASE", 4>;
def MapType_DELETE  : I64EnumAttrCase<"DELETE", 5>;

def MapTypeAttr : I64EnumAttr<"MapType", "OpenMP map type",
    [MapType_TO, MapType_FROM, MapType_TOFROM, MapType_ALLOC, 
     MapType_RELEASE, MapType_DELETE]>;
```

**MLIR Representation**:
```mlir
// Target region with mapped variables
func.func @target_region(%arg0: !fir.ref<!fir.array<1024xf32>>) {
  %c1024 = arith.constant 1024 : index
  
  // Map variables with types
  %map_a = omp.map_info var_ptr(%arg0 : !fir.ref<!fir.array<1024xf32>>)
      map_type(to) map_capture_type(byref) : !fir.ref<!fir.array<1024xf32>>
  
  omp.target map_entries(%map_a -> %arg_device : !fir.ref<!fir.array<1024xf32>>) {
    // Device computation
    omp.parallel {
      omp.wsloop {
        // Loop body executes on device
        omp.yield
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}
```

---

#### Stage 4: Lowering - Parse Tree to MLIR

**File**: `flang/lib/Lower/OpenMP/OpenMP.cpp`

```cpp
void genTargetOp(
    lower::AbstractConverter &converter,
    lower::SymMap &symMap,
    semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval,
    const parser::OpenMPBlockConstruct &blockConstruct) {
  
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  
  // Extract TARGET directive and clauses
  const auto &beginDir = std::get<parser::OmpBeginBlockDirective>(blockConstruct.t);
  const auto &clauses = std::get<parser::OmpClauseList>(beginDir.t);
  
  // Process MAP clauses
  llvm::SmallVector<mlir::Value> mapVars;
  llvm::SmallVector<mlir::IntegerAttr> mapTypes;
  
  for (const auto &clause : clauses.v) {
    if (const auto *mapClause = std::get_if<parser::OmpClause::Map>(&clause.u)) {
      processMapClause(converter, *mapClause, mapVars, mapTypes);
    }
  }
  
  // Extract IF clause
  mlir::Value ifExpr;
  if (const auto *ifClause = findClause<parser::OmpClause::If>(clauses)) {
    ifExpr = genIfClauseValue(converter, *ifClause);
  }
  
  // Extract DEVICE clause
  mlir::Value deviceExpr;
  if (const auto *devClause = findClause<parser::OmpClause::Device>(clauses)) {
    deviceExpr = genDeviceValue(converter, *devClause);
  }
  
  // Create omp.target operation
  auto targetOp = firOpBuilder.create<mlir::omp::TargetOp>(
      loc,
      ifExpr,              // if clause expression
      deviceExpr,          // device ID
      mapVars,             // mapped variables
      mapTypes);           // map types (TO/FROM/etc)
  
  // Create region for device code
  mlir::Block *targetBlock = firOpBuilder.createBlock(&targetOp.getRegion());
  firOpBuilder.setInsertionPointToStart(targetBlock);
  
  // Lower nested constructs inside target region
  const auto &block = std::get<parser::Block>(blockConstruct.t);
  for (const auto &construct : block.v) {
    genOpenMPConstruct(converter, symMap, semaCtx, eval, construct);
  }
  
  // Terminate target region
  firOpBuilder.create<mlir::omp::TerminatorOp>(loc);
}

void processMapClause(
    lower::AbstractConverter &converter,
    const parser::OmpClause::Map &mapClause,
    llvm::SmallVector<mlir::Value> &mapVars,
    llvm::SmallVector<mlir::IntegerAttr> &mapTypes) {
  
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  
  auto [mapTypeModifiers, mapType, mappers, objects] = mapClause.t;
  
  // Determine map type (TO/FROM/TOFROM/etc)
  mlir::omp::MapType mapTypeVal;
  switch (mapType->v) {
    case parser::OmpMapClause::Type::To:
      mapTypeVal = mlir::omp::MapType::TO;
      break;
    case parser::OmpMapClause::Type::From:
      mapTypeVal = mlir::omp::MapType::FROM;
      break;
    case parser::OmpMapClause::Type::Tofrom:
      mapTypeVal = mlir::omp::MapType::TOFROM;
      break;
    // ... other map types
  }
  
  // Process each mapped object
  for (const auto &obj : objects.v) {
    const auto *symbol = ResolveOmpObject(*obj, converter);
    
    // Get base address of variable
    mlir::Value baseAddr = converter.getSymbolAddress(*symbol);
    
    // Create map_info operation (describes mapping)
    auto mapInfo = builder.create<mlir::omp::MapInfoOp>(
        loc,
        baseAddr.getType(),
        baseAddr,
        /*varPtrPtr=*/mlir::Value{},
        /*members=*/mlir::SmallVector<mlir::Value>{},
        /*bounds=*/mlir::SmallVector<mlir::Value>{},
        builder.getIntegerAttr(builder.getI64Type(), 
            static_cast<int64_t>(mapTypeVal)),
        /*varType=*/mlir::TypeAttr::get(baseAddr.getType()),
        /*varName=*/builder.getStringAttr(symbol->name().ToString()));
    
    mapVars.push_back(mapInfo.getResult());
    mapTypes.push_back(builder.getI64IntegerAttr(static_cast<int64_t>(mapTypeVal)));
  }
}
```

**Key Steps**:
1. Extract MAP clauses and process each mapped variable
2. Create `omp.map_info` operations describing memory mappings
3. Generate `omp.target` operation with map operands
4. Lower nested region (parallel loops, etc.) inside target
5. Add terminator to target region

---

#### Stage 5: LLVM IR Translation - Device Kernel Generation

**File**: `mlir/lib/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.cpp`

```cpp
static llvm::Error convertOmpTarget(
    mlir::omp::TargetOp targetOp,
    llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) {
  
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  llvm::Module *llvmModule = builder.GetInsertBlock()->getModule();
  
  // Step 1: Outline target region into separate function (kernel)
  llvm::Function *outlinedFn = outlineTargetRegion(
      targetOp, builder, moduleTranslation);
  
  // Step 2: Mark outlined function as device kernel
  outlinedFn->addFnAttr("omp_target_kernel", "true");
  
  // Step 3: Process map clauses - create host-side mapping arrays
  llvm::SmallVector<llvm::Value*> basePointers;   // Base addresses
  llvm::SmallVector<llvm::Value*> pointers;       // Actual pointers
  llvm::SmallVector<llvm::Value*> sizes;          // Sizes in bytes
  llvm::SmallVector<int64_t> mapTypes;            // Map type flags
  
  for (auto [mapVar, mapType] : llvm::zip(
      targetOp.getMapVars(), targetOp.getMapTypes())) {
    
    mlir::Value mlirValue = mapVar;
    llvm::Value *llvmValue = moduleTranslation.lookupValue(mlirValue);
    
    // Compute size of mapped region
    llvm::Type *elementType = llvmValue->getType()->getPointerElementType();
    llvm::Value *size = builder.getInt64(
        llvmModule->getDataLayout().getTypeAllocSize(elementType));
    
    basePointers.push_back(llvmValue);
    pointers.push_back(llvmValue);
    sizes.push_back(size);
    mapTypes.push_back(mapType.cast<mlir::IntegerAttr>().getInt());
  }
  
  // Step 4: Generate device runtime calls
  
  // 4a. Begin data mapping (allocate device memory, copy TO data)
  llvm::Value *mapperArrayPtr = createMapperArray(
      builder, basePointers, pointers, sizes, mapTypes);
  
  llvm::FunctionCallee beginFn = llvmModule->getOrInsertFunction(
      "__tgt_target_data_begin",
      llvm::FunctionType::get(
          builder.getVoidTy(),
          {builder.getInt64Ty(),    // device ID
           builder.getInt32Ty(),    // arg count
           mapperArrayPtr->getType(), mapperArrayPtr->getType(),
           sizes[0]->getType(), mapperArrayPtr->getType()},
          false));
  
  llvm::Value *deviceId = builder.getInt64(/*default device*/ -1);
  if (targetOp.getDevice()) {
    deviceId = moduleTranslation.lookupValue(targetOp.getDevice());
  }
  
  builder.CreateCall(beginFn, {
      deviceId,
      builder.getInt32(basePointers.size()),
      createArrayPointer(builder, basePointers),
      createArrayPointer(builder, pointers),
      createArrayPointer(builder, sizes),
      createArrayPointer(builder, mapTypes)
  });
  
  // 4b. Launch kernel on device
  llvm::FunctionCallee kernelLaunchFn = llvmModule->getOrInsertFunction(
      "__tgt_target_kernel",
      llvm::FunctionType::get(
          builder.getInt32Ty(),  // Returns 0 on success
          {builder.getInt64Ty(),           // device ID
           outlinedFn->getType(),          // kernel function pointer
           builder.getInt32Ty(),           // arg count
           mapperArrayPtr->getType(),      // arg pointers
           mapperArrayPtr->getType(),      // arg sizes
           builder.getInt64Ty()},          // num teams
          false));
  
  llvm::Value *kernelResult = builder.CreateCall(kernelLaunchFn, {
      deviceId,
      outlinedFn,
      builder.getInt32(basePointers.size()),
      createArrayPointer(builder, pointers),
      createArrayPointer(builder, sizes),
      builder.getInt64(/*default teams*/ 0)
  });
  
  // 4c. Check if kernel launch failed (non-zero return)
  // If failed, execute fallback host version
  llvm::BasicBlock *continueBB = llvm::BasicBlock::Create(
      builder.getContext(), "target.continue", builder.GetInsertBlock()->getParent());
  llvm::BasicBlock *fallbackBB = llvm::BasicBlock::Create(
      builder.getContext(), "target.fallback", builder.GetInsertBlock()->getParent());
  
  llvm::Value *launchFailed = builder.CreateICmpNE(
      kernelResult, builder.getInt32(0));
  builder.CreateCondBr(launchFailed, fallbackBB, continueBB);
  
  // Fallback: execute on host if device unavailable
  builder.SetInsertPoint(fallbackBB);
  llvm::SmallVector<llvm::Value*> hostArgs;
  for (auto ptr : pointers) {
    hostArgs.push_back(ptr);
  }
  builder.CreateCall(outlinedFn, hostArgs);
  builder.CreateBr(continueBB);
  
  // 4d. End data mapping (copy FROM data, deallocate device memory)
  builder.SetInsertPoint(continueBB);
  
  llvm::FunctionCallee endFn = llvmModule->getOrInsertFunction(
      "__tgt_target_data_end",
      beginFn.getFunctionType());
  
  builder.CreateCall(endFn, {
      deviceId,
      builder.getInt32(basePointers.size()),
      createArrayPointer(builder, basePointers),
      createArrayPointer(builder, pointers),
      createArrayPointer(builder, sizes),
      createArrayPointer(builder, mapTypes)
  });
  
  return llvm::Error::success();
}

// Outline target region into device kernel function
llvm::Function *outlineTargetRegion(
    mlir::omp::TargetOp targetOp,
    llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) {
  
  llvm::Module *llvmModule = builder.GetInsertBlock()->getModule();
  
  // Create function signature for kernel
  llvm::SmallVector<llvm::Type*> argTypes;
  for (auto mapVar : targetOp.getMapVars()) {
    llvm::Value *llvmValue = moduleTranslation.lookupValue(mapVar);
    argTypes.push_back(llvmValue->getType());
  }
  
  llvm::FunctionType *fnType = llvm::FunctionType::get(
      builder.getVoidTy(), argTypes, /*isVarArg=*/false);
  
  llvm::Function *outlinedFn = llvm::Function::Create(
      fnType,
      llvm::GlobalValue::InternalLinkage,
      ".__omp_offloading_kernel",
      llvmModule);
  
  // Set function attributes for device execution
  outlinedFn->addFnAttr("omp_target_kernel");
  
  // Translate target region body
  llvm::BasicBlock *entryBB = llvm::BasicBlock::Create(
      builder.getContext(), "entry", outlinedFn);
  llvm::IRBuilder<> fnBuilder(entryBB);
  
  // Map MLIR region arguments to LLVM function arguments
  for (auto [mlirArg, llvmArg] : llvm::zip(
      targetOp.getRegion().front().getArguments(),
      outlinedFn->args())) {
    moduleTranslation.mapValue(mlirArg, &llvmArg);
  }
  
  // Translate region operations
  for (mlir::Operation &op : targetOp.getRegion().front()) {
    if (mlir::failed(moduleTranslation.convertOperation(op, fnBuilder))) {
      llvm::errs() << "Failed to translate operation in target region\n";
    }
  }
  
  // Add return
  fnBuilder.CreateRetVoid();
  
  return outlinedFn;
}
```

**Generated LLVM IR** (simplified):
```llvm
; Host-side code
define void @target_region(ptr %a, ptr %b, ptr %c) {
entry:
  ; Setup mapping arrays
  %base_ptrs = alloca [3 x ptr]
  %ptrs = alloca [3 x ptr]
  %sizes = alloca [3 x i64]
  %map_types = alloca [3 x i64]
  
  ; Fill arrays
  store ptr %a, ptr %base_ptrs, align 8
  store i64 4096, ptr %sizes, align 8      ; sizeof(float[1024])
  store i64 1, ptr %map_types, align 8     ; TO
  ; ... similar for b and c
  
  ; Begin data mapping (copy to device)
  call void @__tgt_target_data_begin(
      i64 -1,              ; default device
      i32 3,               ; 3 mapped variables
      ptr %base_ptrs,
      ptr %ptrs,
      ptr %sizes,
      ptr %map_types)
  
  ; Launch kernel
  %result = call i32 @__tgt_target_kernel(
      i64 -1,                               ; device ID
      ptr @.__omp_offloading_kernel,        ; kernel function
      i32 3,                                ; arg count
      ptr %ptrs,                            ; kernel args
      ptr %sizes,                           ; arg sizes
      i64 0)                                ; num teams (0 = default)
  
  ; Check for fallback
  %failed = icmp ne i32 %result, 0
  br i1 %failed, label %fallback, label %continue
  
fallback:
  ; Execute on host if device unavailable
  call void @.__omp_offloading_kernel(ptr %a, ptr %b, ptr %c)
  br label %continue
  
continue:
  ; End data mapping (copy from device, deallocate)
  call void @__tgt_target_data_end(
      i64 -1, i32 3, ptr %base_ptrs, ptr %ptrs, ptr %sizes, ptr %map_types)
  
  ret void
}

; Device kernel (compiled separately for target architecture)
define void @.__omp_offloading_kernel(ptr %a, ptr %b, ptr %c) #0 {
entry:
  ; Parallel loop implementation
  ; (uses device-specific parallel constructs)
  ret void
}

attributes #0 = { "omp_target_kernel" }
```

---

### TEAMS Directive: Thread Team Management on Device

**Purpose**: Create a league of thread teams on the device. Each team executes the TEAMS region independently.

**Fortran Example**:
```fortran
!$omp target teams num_teams(4) thread_limit(256)
  !$omp distribute
  do i = 1, n
    a(i) = b(i) + c(i)
  end do
!$omp end target teams
```

#### MLIR Representation

```mlir
omp.target map_entries(...) {
  omp.teams num_teams(%c4 : i32) thread_limit(%c256 : i32) {
    omp.distribute {
      omp.wsloop (%i) : i32 = (%lb) to (%ub) step (%step) {
        // Loop body
        omp.yield
      }
      omp.terminator
    }
    omp.terminator
  }
  omp.terminator
}
```

#### LLVM IR Translation

**File**: `mlir/lib/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.cpp`

```cpp
static llvm::Error convertOmpTeams(
    mlir::omp::TeamsOp teamsOp,
    llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) {
  
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  
  // Extract num_teams and thread_limit
  llvm::Value *numTeams = nullptr;
  if (teamsOp.getNumTeams()) {
    numTeams = moduleTranslation.lookupValue(teamsOp.getNumTeams());
  }
  
  llvm::Value *threadLimit = nullptr;
  if (teamsOp.getThreadLimit()) {
    threadLimit = moduleTranslation.lookupValue(teamsOp.getThreadLimit());
  }
  
  // Outline teams region
  auto outlineCB = [&](llvm::OpenMPIRBuilder::InsertPointTy allocaIP,
                       llvm::OpenMPIRBuilder::InsertPointTy codeGenIP) {
    // Translate teams region body
    return moduleTranslation.convertBlock(
        teamsOp.getRegion().front(), /*ignoreArguments=*/true, builder);
  };
  
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  
  // Generate teams construct
  builder.restoreIP(ompBuilder->createTeams(
      ompLoc,
      outlineCB,
      numTeams,
      threadLimit));
  
  return llvm::Error::success();
}
```

**Generated LLVM IR**:
```llvm
; Inside device kernel from TARGET
define void @.__omp_offloading_kernel(...) {
entry:
  ; Get team and thread IDs
  %team_id = call i32 @__kmpc_get_team_num()
  %thread_id = call i32 @__kmpc_get_thread_num()
  
  ; Initialize teams construct
  call void @__kmpc_push_num_teams(ptr @loc, i32 %thread_id, i32 4, i32 256)
  
  ; Fork teams (each team executes outlined function)
  call void @__kmpc_fork_teams(ptr @loc, i32 0, ptr @.omp_outlined.)
  
  ret void
}

define internal void @.omp_outlined.(ptr %team_id, ptr %thread_id) {
  ; DISTRIBUTE loop implementation
  ; Each team processes subset of iterations
  ret void
}
```

---

### Runtime Interaction: libomptarget Entry Points

#### Data Mapping Functions

```cpp
// Begin data mapping - allocate device memory and copy TO data
void __tgt_target_data_begin(
    int64_t device_id,
    int32_t arg_num,
    void **args_base,    // Base pointers
    void **args,         // Actual pointers (may differ for array sections)
    int64_t *arg_sizes,  // Sizes in bytes
    int64_t *arg_types); // Map type flags (TO/FROM/ALLOC/etc)

// End data mapping - copy FROM data and deallocate
void __tgt_target_data_end(
    int64_t device_id,
    int32_t arg_num,
    void **args_base,
    void **args,
    int64_t *arg_sizes,
    int64_t *arg_types);

// Update device data (explicit synchronization)
void __tgt_target_data_update(
    int64_t device_id,
    int32_t arg_num,
    void **args_base,
    void **args,
    int64_t *arg_sizes,
    int64_t *arg_types);
```

#### Kernel Launch Functions

```cpp
// Launch kernel on device
int32_t __tgt_target_kernel(
    int64_t device_id,
    void *host_ptr,           // Kernel function pointer
    int32_t arg_num,
    void **args_base,
    void **args,
    int64_t *arg_sizes,
    int64_t *arg_types,
    int64_t *num_teams,       // TEAMS clause
    int64_t *thread_limit,    // THREAD_LIMIT clause
    int32_t flags);           // Target flags

// Returns:
//   0  - Success (kernel executed on device)
//  !0  - Failure (device unavailable, fallback to host)
```

#### Device Management

```cpp
// Initialize device
int32_t __tgt_target_init();

// Check if device is available
int32_t __tgt_is_present(void *ptr, int64_t device_id);

// Allocate device memory
void *__tgt_target_alloc(int64_t size, int64_t device_id);

// Free device memory
void __tgt_target_free(void *ptr, int64_t device_id);
```

---

### Map Types and Flags

**Map Type Values** (from `llvm/include/llvm/Frontend/OpenMP/OMPKinds.def`):

```cpp
enum MapType : uint64_t {
  OMP_MAP_TO          = 0x01,  // Copy host → device
  OMP_MAP_FROM        = 0x02,  // Copy device → host
  OMP_MAP_TOFROM      = 0x03,  // Copy both directions
  OMP_MAP_ALLOC       = 0x00,  // Allocate only (no copy)
  OMP_MAP_RELEASE     = 0x04,  // Decrement reference count
  OMP_MAP_DELETE      = 0x08,  // Delete device copy
  
  // Modifiers
  OMP_MAP_ALWAYS      = 0x100, // Always copy (even if already mapped)
  OMP_MAP_CLOSE       = 0x200, // Allocate in close memory (HBM)
  OMP_MAP_PRESENT     = 0x400, // Variable must already be present on device
};
```

**Usage in Generated Code**:
```cpp
// map(to: a)
int64_t map_type_a = OMP_MAP_TO;

// map(tofrom: b)
int64_t map_type_b = OMP_MAP_TOFROM;

// map(always, to: c)
int64_t map_type_c = OMP_MAP_TO | OMP_MAP_ALWAYS;
```

---

### Device Data Environment

**Implicit Mapping Rules**:
1. Scalar variables: `firstprivate` (each thread gets copy)
2. Array variables: Not implicitly mapped (must use MAP clause)
3. Global/module variables: Not implicitly mapped

**Structured vs Unstructured Data Mapping**:

**Structured** (data lifetime matches construct):
```fortran
!$omp target map(to: a) map(from: b)
  ! a and b allocated on entry, deallocated on exit
  b = a + 1.0
!$omp end target
```

**Unstructured** (explicit data lifetime management):
```fortran
!$omp target enter data map(to: a)
! a remains on device

!$omp target map(alloc: a)  ! Reuse existing device copy
  a = a + 1.0
!$omp end target

!$omp target exit data map(from: a)  ! Copy back and deallocate
```

---

### Complete Device Execution Model

**Hierarchy**:
```
TARGET (offload to device)
 └─ TEAMS (league of teams)
     └─ DISTRIBUTE (iterations across teams)
         └─ PARALLEL (threads within team)
             └─ SIMD (vectorization)
```

**Mapping to GPU Execution**:
- **TEAMS** → Grid of thread blocks (CUDA) / Workgroups (OpenCL/HIP)
- **DISTRIBUTE** → Distribute iterations across thread blocks
- **PARALLEL** → Threads within thread block
- **SIMD** → Vector instructions within thread

**Example: Full Offload**:
```fortran
!$omp target teams distribute parallel do simd map(tofrom: a)
do i = 1, n
  a(i) = a(i) * 2.0
end do
!$omp end target teams distribute parallel do simd
```

**GPU Execution**:
- **Teams**: Multiple thread blocks across GPU
- **Distribute**: Loop iterations divided among thread blocks
- **Parallel**: Threads within each block process iterations
- **SIMD**: Vectorized execution within each thread

---

### Debugging Device Constructs

#### Enable Offload Debug Output

```bash
# Set environment variables for libomptarget debugging
export LIBOMPTARGET_DEBUG=1        # Basic debug info
export LIBOMPTARGET_INFO=0xFF      # All info messages

# Plugin-specific debugging
export LIBOMPTARGET_PLUGIN_CUDA_DEBUG=1     # CUDA plugin debug
export LIBOMPTARGET_PLUGIN_AMDGPU_DEBUG=1   # AMD GPU plugin debug

# Run program
./my_target_program
```

**Debug Output Example**:
```
Libomptarget --> Loading device plugin libomp target.rtl.cuda.so
Libomptarget --> Device 0 initialized: NVIDIA A100-SXM4-40GB
Libomptarget --> Entering __tgt_target_data_begin with 3 mappings
Libomptarget -->   Mapping 0: base=0x7ffd1234, ptr=0x7ffd1234, size=4096, type=TO
Libomptarget -->   Allocated device memory: 0x7f8000000000
Libomptarget -->   Copying 4096 bytes to device
Libomptarget --> Launching kernel on device 0
Libomptarget -->   Teams: 32, Threads per team: 256
Libomptarget --> Kernel execution successful
Libomptarget --> Copying 4096 bytes from device
```

#### Dump Device IR

```bash
# Generate device-specific IR (PTX for NVIDIA)
flang-new -fopenmp -fopenmp-targets=nvptx64 -S -emit-llvm target.f90 -o target.ll

# View device kernel code
grep -A 50 "omp_offloading" target.ll

# For AMDGPU
flang-new -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -S -emit-llvm target.f90
```

#### Check Map Clause Lowering

```bash
# Dump MLIR with target operations
flang-new -fc1 -emit-mlir -fopenmp target.f90 | grep -A 20 "omp.target"

# Look for map_info operations
flang-new -fc1 -emit-mlir -fopenmp target.f90 | grep "omp.map_info"
```

---

### Common Device Construct Issues

**Issue 1: Variable Not Mapped**
```fortran
! ERROR: Implicit mapping not allowed for arrays
!$omp target
  do i = 1, n
    a(i) = b(i) + c(i)  ! a, b, c not mapped!
  end do
!$omp end target
```

**Fix**: Explicit MAP clause
```fortran
!$omp target map(to: b, c) map(from: a)
  do i = 1, n
    a(i) = b(i) + c(i)
  end do
!$omp end target
```

**Issue 2: Nested TARGET**
```fortran
! ERROR: TARGET cannot be nested
!$omp target
  !$omp target  ! Not allowed!
    ! ...
  !$omp end target
!$omp end target
```

**Issue 3: Runtime Fallback**
```
Warning: Target region executed on host (device unavailable)
```

**Causes**:
- No device available
- Device initialization failed
- Insufficient device memory

**Solutions**:
- Check device availability: `export OMP_DEFAULT_DEVICE=0`
- Reduce mapped data size
- Check driver installation

---

### Best Practices for Device Constructs

**1. Minimize Data Transfer**:
```fortran
! BAD: Map entire array for small computation
!$omp target map(tofrom: a(1:n))
  a(1) = a(1) + 1.0
!$omp end target

! GOOD: Map only needed elements
!$omp target map(tofrom: a(1:1))
  a(1) = a(1) + 1.0
!$omp end target
```

**2. Use Unstructured Data Mapping for Multiple Kernels**:
```fortran
! GOOD: Keep data on device across multiple kernels
!$omp target enter data map(to: a, b)

!$omp target map(alloc: a, b, c)
  c = a + b
!$omp end target

!$omp target map(alloc: a, c)
  a = a * c
!$omp end target

!$omp target exit data map(from: a) map(delete: b)
```

**3. Optimize Team/Thread Configuration**:
```fortran
! Match hardware capabilities
! NVIDIA A100: 108 SMs, max 1024 threads/block
!$omp target teams num_teams(108) thread_limit(256)
  !$omp distribute parallel do
  do i = 1, n
    a(i) = b(i) + c(i)
  end do
!$omp end target teams
```

**4. Use PRESENT Modifier for Debugging**:
```fortran
! Ensure variable is already on device (catches mapping errors)
!$omp target map(present, tofrom: a)
  ! If 'a' not on device, runtime error instead of silent fallback
!$omp end target
```

---

## LLVM IR Translationfortran
! Example: Adding NEWCLAUSE(list)
```

**File**: `flang/lib/Parser/openmp-parsers.cpp`
```cpp
// Add to clause grammar
TYPE_PARSER(
  "NEWCLAUSE" >> parenthesized(Parser<OmpObjectList>{})
)
```

**File**: `flang/include/flang/Parser/parse-tree.h`
```cpp
struct OmpClause::Newclause {
  OmpObjectList v;
};
```

### Step 2: Semantic Validation
**File**: `flang/lib/Semantics/check-omp-structure.cpp`

```cpp
void OmpStructureChecker::Enter(const parser::OmpClause::Newclause &x) {
  // Validate clause usage
  CheckAllowed(llvm::omp::Clause::OMPC_newclause);
  
  // Check restrictions
  for (const auto &obj : x.v) {
    // Validate each object in the list
  }
}
```

### Step 3: MLIR Dialect Operation
**File**: `mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td`

```tablegen
// Add attribute or operand to relevant operation
def SomeOperation : OpenMP_Op<"some_op"> {
  let arguments = (ins
    // ...existing operands...
    Optional<TypeRange>:$newclause_vars
  );
}
```

### Step 4: Lowering
**File**: `flang/lib/Lower/OpenMP/Clauses.cpp`

```cpp
case llvm::omp::Clause::OMPC_newclause: {
  // Extract objects from parse tree
  // Create MLIR operands
  // Add to clause processing
  break;
}
```

### Step 5: LLVM Translation
**File**: `mlir/lib/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.cpp`

```cpp
// Handle clause during operation translation
if (!op.getNewclauseVars().empty()) {
  // Generate appropriate LLVM IR
  // Call OpenMP runtime if needed
}
```

### Step 6: Testing
Create tests at each level:
- Parser: `flang/test/Parser/OpenMP/newclause.f90`
- Semantics: `flang/test/Semantics/OpenMP/newclause-errors.f90`
- Lowering: `flang/test/Lower/OpenMP/newclause.f90`
- LLVM IR: Check generated IR includes runtime calls

---

## Common Validation Patterns

### 1. Clause Allowed On Directive
```cpp
void OmpStructureChecker::Enter(const parser::OmpClause::Something &) {
  CheckAllowed(llvm::omp::Clause::OMPC_something);
}
```

### 2. Variable Type Restrictions
```cpp
const auto *symbol = ResolveOmp(*name, SymbolFlag::ObjectName, context);
if (symbol && !symbol->IsAllocatable()) {
  context_.Say(name->source,
    "Variable must be allocatable"_err_en_US);
}
```

### 3. Mutually Exclusive Clauses
```cpp
if (HasClause(llvm::omp::Clause::OMPC_private) &&
    HasClause(llvm::omp::Clause::OMPC_shared)) {
  context_.Say(currentDir.source,
    "PRIVATE and SHARED cannot be specified together"_err_en_US);
}
```

### 4. List Item Uniqueness
```cpp
std::set<const Symbol *> seenSymbols;
for (const auto &name : objectList) {
  const auto *symbol = ResolveOmp(*name, ...);
  if (!seenSymbols.insert(symbol).second) {
    context_.Say(name->source,
      "Variable appears multiple times"_err_en_US);
  }
}
```

---

## Best Practices for OpenMP Semantic Validation

Writing effective semantic checks ensures OpenMP directives and clauses are used correctly according to the specification. This section provides best practices with examples of good vs bad validation patterns.

---

### Principle 1: Check Clause Restrictions First

**Good Practice**: Always check if clause is allowed on directive before validating clause content

```cpp
// GOOD: Check allowed first, then validate content
void OmpStructureChecker::Enter(const parser::OmpClause::Private &x) {
  // Step 1: Verify clause is allowed on current directive
  CheckAllowed(llvm::omp::Clause::OMPC_private);
  
  // Step 2: Only if allowed, validate the clause contents
  if (GetContext().directive == llvm::omp::Directive::OMPD_unknown) {
    return;  // Don't validate if directive context is invalid
  }
  
  // Step 3: Validate variable list
  for (const auto &name : x.v.v) {
    CheckObjectInNamelist(name, ...);
  }
}

// BAD: Validate content without checking if allowed
void OmpStructureChecker::Enter(const parser::OmpClause::Private &x) {
  // BUG: May validate clause on wrong directive
  for (const auto &name : x.v.v) {
    CheckObjectInNamelist(name, ...);
  }
  // Missing CheckAllowed() - will not catch PRIVATE on BARRIER
}
```

**Why**: Prevents cascade of confusing error messages when clause appears on wrong directive.

---

### Principle 2: Use Specific, Actionable Error Messages

**Good Practice**: Error messages should explain what's wrong and how to fix it

```cpp
// GOOD: Specific, actionable messages
void OmpStructureChecker::CheckReductionVar(const parser::Name &name) {
  const auto *symbol = ResolveOmp(name, SymbolFlag::ObjectName, context_);
  
  if (!symbol) {
    context_.Say(name.source,
        "Variable '%s' in REDUCTION clause must be a declared variable"_err_en_US,
        name.ToString());
    return;
  }
  
  if (symbol->test(Symbol::Flag::OmpThreadprivate)) {
    context_.Say(name.source,
        "Threadprivate variable '%s' cannot appear in REDUCTION clause; "
        "use a different variable or remove THREADPRIVATE directive"_err_en_US,
        name.ToString());
    return;
  }
  
  if (!IsValidReductionType(symbol->GetType())) {
    context_.Say(name.source,
        "Variable '%s' has type '%s' which is not valid for REDUCTION; "
        "only numeric and logical types are allowed"_err_en_US,
        name.ToString(),
        symbol->GetType()->AsFortran());
    return;
  }
}

// BAD: Vague, unhelpful messages
void OmpStructureChecker::CheckReductionVar(const parser::Name &name) {
  const auto *symbol = ResolveOmp(name, SymbolFlag::ObjectName, context_);
  
  if (!symbol) {
    context_.Say(name.source, "Invalid variable"_err_en_US);  // What's invalid?
    return;
  }
  
  if (symbol->test(Symbol::Flag::OmpThreadprivate)) {
    context_.Say(name.source, "Not allowed"_err_en_US);  // Why not?
    return;
  }
  
  if (!IsValidReductionType(symbol->GetType())) {
    context_.Say(name.source, "Wrong type"_err_en_US);  // What type is needed?
  }
}
```

**Why**: Developers using Flang need clear guidance on how to fix their code.

---

### Principle 3: Validate at the Right Abstraction Level

**Good Practice**: Check semantic properties, not syntactic patterns

```cpp
// GOOD: Check semantic property (is it a variable?)
void OmpStructureChecker::CheckDataSharingVar(
    const parser::Name &name,
    const parser::OmpClause &clause) {
  
  const auto *symbol = ResolveOmp(name, SymbolFlag::ObjectName, context_);
  
  if (!symbol) {
    context_.Say(name.source,
        "Name in data-sharing clause must refer to a variable"_err_en_US);
    return;
  }
  
  // Check semantic properties
  if (symbol->owner().IsDerivedType()) {
    context_.Say(name.source,
        "Components of derived types cannot appear in data-sharing clauses"_err_en_US);
    return;
  }
  
  if (IsAssumedSizeArray(*symbol)) {
    context_.Say(name.source,
        "Assumed-size arrays cannot appear in data-sharing clauses"_err_en_US);
    return;
  }
}

// BAD: Check syntactic pattern instead of semantics
void OmpStructureChecker::CheckDataSharingVar(
    const parser::Name &name,
    const parser::OmpClause &clause) {
  
  // BUG: Checking name pattern, not actual symbol properties
  if (name.ToString().find('%') != std::string::npos) {
    context_.Say(name.source, "Invalid variable name"_err_en_US);
    return;  // Too simplistic - misses semantic issues
  }
  
  // BUG: Should check symbol, not string
  if (name.ToString().find('(') != std::string::npos) {
    context_.Say(name.source, "Arrays not allowed"_err_en_US);
    return;  // Wrong - array sections ARE allowed in some clauses
  }
}
```

**Why**: Syntax can be misleading; semantic properties capture actual OpenMP requirements.

---

### Principle 4: Check Mutual Exclusivity Correctly

**Good Practice**: Check for conflicting clauses at directive level, not clause level

```cpp
// GOOD: Check mutual exclusivity in directive Exit()
void OmpStructureChecker::Leave(const parser::OmpClauseList &clauses) {
  // Collect which clauses are present
  bool hasNowait = HasClause(llvm::omp::Clause::OMPC_nowait);
  bool hasOrdered = HasClause(llvm::omp::Clause::OMPC_ordered);
  
  // Check conflicts after all clauses processed
  if (hasNowait && hasOrdered) {
    context_.Say(GetContext().clauseSource,
        "NOWAIT and ORDERED clauses are mutually exclusive on worksharing constructs"_err_en_US);
  }
  
  // Check SCHEDULE and GRAINSIZE/NUM_TASKS
  bool hasSchedule = HasClause(llvm::omp::Clause::OMPC_schedule);
  bool hasGrainsize = HasClause(llvm::omp::Clause::OMPC_grainsize);
  bool hasNumTasks = HasClause(llvm::omp::Clause::OMPC_num_tasks);
  
  if (hasSchedule && (hasGrainsize || hasNumTasks)) {
    context_.Say(GetContext().clauseSource,
        "SCHEDULE clause is mutually exclusive with GRAINSIZE and NUM_TASKS"_err_en_US);
  }
  
  if (hasGrainsize && hasNumTasks) {
    context_.Say(GetContext().clauseSource,
        "GRAINSIZE and NUM_TASKS clauses are mutually exclusive"_err_en_US);
  }
}

// BAD: Try to check mutual exclusivity in individual Enter() methods
void OmpStructureChecker::Enter(const parser::OmpClause::Nowait &) {
  CheckAllowed(llvm::omp::Clause::OMPC_nowait);
  
  // BUG: Ordered clause might not be processed yet!
  if (HasClause(llvm::omp::Clause::OMPC_ordered)) {
    context_.Say(GetContext().clauseSource,
        "Cannot have both NOWAIT and ORDERED"_err_en_US);
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Ordered &) {
  CheckAllowed(llvm::omp::Clause::OMPC_ordered);
  
  // BUG: Nowait might not be processed yet!
  if (HasClause(llvm::omp::Clause::OMPC_nowait)) {
    context_.Say(GetContext().clauseSource,
        "Cannot have both ORDERED and NOWAIT"_err_en_US);
  }
  // Result: Duplicate error messages, or worse, missed conflicts depending on clause order!
}
```

**Why**: Clauses are processed in order; checking in `Leave()` ensures all clauses are seen.

---

### Principle 5: Handle Symbol Resolution Failures Gracefully

**Good Practice**: Check for null symbols and provide helpful messages

```cpp
// GOOD: Graceful handling of unresolved symbols
void OmpStructureChecker::CheckPrivateVar(const parser::Name &name) {
  const auto *symbol = ResolveOmp(name, SymbolFlag::ObjectName, context_);
  
  if (!symbol) {
    // Could be undeclared variable or non-variable name
    if (context_.FindScope(name.source).FindSymbol(name.source)) {
      // Symbol exists but isn't a variable
      context_.Say(name.source,
          "'%s' in PRIVATE clause must be a variable, not a %s"_err_en_US,
          name.ToString(),
          GetSymbolKindName(name));
    } else {
      // Undeclared
      context_.Say(name.source,
          "Variable '%s' in PRIVATE clause must be declared"_err_en_US,
          name.ToString());
    }
    return;  // Stop validation - can't proceed without symbol
  }
  
  // Continue with symbol-based checks
  ValidateDataSharingAttribute(*symbol, name.source);
}

// BAD: Crash or continue with null pointer
void OmpStructureChecker::CheckPrivateVar(const parser::Name &name) {
  const auto *symbol = ResolveOmp(name, SymbolFlag::ObjectName, context_);
  
  // BUG: No null check - will crash if symbol not found!
  if (symbol->test(Symbol::Flag::OmpThreadprivate)) {
    context_.Say(name.source,
        "Threadprivate variable cannot be PRIVATE"_err_en_US);
  }
  
  // More checks using symbol... all will crash if symbol is null
}
```

**Why**: Semantic analysis may encounter unresolved symbols; graceful handling prevents crashes.

---

### Principle 6: Validate Composite Directives Correctly

**Good Practice**: Check restrictions that apply to composite constructs

```cpp
// GOOD: Special handling for composite directives
void OmpStructureChecker::Enter(const parser::OpenMPLoopConstruct &x) {
  const auto &beginDir = std::get<parser::OmpBeginLoopDirective>(x.t);
  const auto directive = beginDir.v;
  
  PushContext(directive.source, directive.v);
  
  // Check if this is a composite construct
  if (directive.v == llvm::omp::Directive::OMPD_parallel_do) {
    // Restrictions specific to PARALLEL DO
    // Some clauses are allowed on separate PARALLEL/DO but not on composite
    CheckCompositeRestrictions(directive.v);
  }
  
  // Standard loop checks
  CheckLoopStructure(x);
}

void OmpStructureChecker::CheckCompositeRestrictions(
    llvm::omp::Directive directive) {
  
  // Example: ORDERED clause restrictions on composite constructs
  if (HasClause(llvm::omp::Clause::OMPC_ordered)) {
    if (directive == llvm::omp::Directive::OMPD_parallel_do) {
      // OpenMP 5.2: ORDERED not allowed on parallel do
      context_.Say(GetContext().clauseSource,
          "ORDERED clause is not allowed on composite PARALLEL DO construct"_err_en_US);
    }
  }
}

// BAD: Treat composite and separate directives the same
void OmpStructureChecker::Enter(const parser::OpenMPLoopConstruct &x) {
  const auto &beginDir = std::get<parser::OmpBeginLoopDirective>(x.t);
  
  PushContext(beginDir.v.source, beginDir.v.v);
  
  // BUG: No special handling for composite constructs
  // Will miss restrictions that apply only to composites
  CheckLoopStructure(x);
}
```

**Why**: Composite constructs have different restrictions than separate directives.

---

### Principle 7: Validate Expression Properties, Not Just Existence

**Good Practice**: Check that expressions meet OpenMP requirements

```cpp
// GOOD: Validate expression properties
void OmpStructureChecker::CheckNumThreadsClause(
    const parser::ScalarIntExpr &expr) {
  
  // Check it's an integer expression
  if (const auto *typedExpr = GetExpr(context_, expr)) {
    auto type = typedExpr->GetType();
    
    if (!type || !type->IsNumeric(common::TypeCategory::Integer)) {
      context_.Say(expr.thing.value().source,
          "NUM_THREADS expression must be a scalar integer"_err_en_US);
      return;
    }
    
    // Check if it's positive (if constant)
    if (auto constVal = evaluate::ToInt64(*typedExpr)) {
      if (*constVal <= 0) {
        context_.Say(expr.thing.value().source,
            "NUM_THREADS expression must be positive, got %d"_err_en_US,
            *constVal);
        return;
      }
    }
    
    // Check for side effects (optional but good practice)
    if (HasSideEffects(*typedExpr)) {
      context_.Say(expr.thing.value().source,
          "NUM_THREADS expression should not have side effects"_warn_en_US);
    }
  }
}

// BAD: Only check that expression exists
void OmpStructureChecker::CheckNumThreadsClause(
    const parser::ScalarIntExpr &expr) {
  
  // BUG: No type checking, no value validation
  if (!expr.thing.has_value()) {
    context_.Say(expr.thing.value().source,
        "NUM_THREADS requires an expression"_err_en_US);
  }
  // That's it - no validation of what kind of expression!
}
```

**Why**: OpenMP requires specific expression properties (type, constness, positivity, etc.).

---

### Principle 8: Check Variable Attributes, Not Just Names

**Good Practice**: Validate symbol attributes required by OpenMP

```cpp
// GOOD: Check all relevant symbol attributes
void OmpStructureChecker::CheckReductionVariable(
    const parser::Name &name,
    const parser::OmpReductionOperator &op) {
  
  const auto *symbol = ResolveOmp(name, SymbolFlag::ObjectName, context_);
  if (!symbol) return;
  
  // Check 1: Variable, not constant
  if (symbol->attrs().test(semantics::Attr::PARAMETER)) {
    context_.Say(name.source,
        "Named constant '%s' cannot appear in REDUCTION clause"_err_en_US,
        name.ToString());
    return;
  }
  
  // Check 2: Not allocatable (special handling needed)
  if (semantics::IsAllocatable(*symbol)) {
    context_.Say(name.source,
        "Allocatable variable '%s' in REDUCTION clause must be allocated"_warn_en_US,
        name.ToString());
  }
  
  // Check 3: Not pointer (unless associated)
  if (symbol->attrs().test(semantics::Attr::POINTER)) {
    context_.Say(name.source,
        "Pointer variable '%s' in REDUCTION clause must be associated"_warn_en_US,
        name.ToString());
  }
  
  // Check 4: Type compatible with operator
  if (!IsTypeCompatibleWithOperator(symbol->GetType(), op)) {
    context_.Say(name.source,
        "Variable '%s' has type '%s' which is incompatible with reduction operator '%s'"_err_en_US,
        name.ToString(),
        symbol->GetType()->AsFortran(),
        OperatorToString(op));
    return;
  }
  
  // Check 5: Not threadprivate
  if (symbol->test(Symbol::Flag::OmpThreadprivate)) {
    context_.Say(name.source,
        "Threadprivate variable '%s' cannot appear in REDUCTION clause"_err_en_US,
        name.ToString());
  }
}

// BAD: Only check variable existence
void OmpStructureChecker::CheckReductionVariable(
    const parser::Name &name,
    const parser::OmpReductionOperator &op) {
  
  const auto *symbol = ResolveOmp(name, SymbolFlag::ObjectName, context_);
  
  if (!symbol) {
    context_.Say(name.source, "Variable not found"_err_en_US);
  }
  // BUG: Missing all attribute checks!
  // Will accept PARAMETERs, wrong types, threadprivate vars, etc.
}
```

**Why**: OpenMP has specific requirements for variable attributes in different contexts.

---

### Principle 9: Provide Context in Error Messages

**Good Practice**: Include directive and clause context in errors

```cpp
// GOOD: Contextual error messages
void OmpStructureChecker::CheckScheduleClause(
    const parser::OmpClause::Schedule &scheduleClause) {
  
  const auto directive = GetContext().directive;
  const auto directiveName = llvm::omp::getOpenMPDirectiveName(directive);
  
  auto [modifiers, kind, chunk] = scheduleClause.t;
  
  // Check chunk size is valid for schedule kind
  if (kind.v == parser::OmpScheduleClause::Kind::Auto && chunk) {
    context_.Say(GetContext().clauseSource,
        "SCHEDULE(AUTO) on %s directive cannot have a chunk size; "
        "remove the chunk size or use a different schedule kind"_err_en_US,
        directiveName.str());
    return;
  }
  
  if (kind.v == parser::OmpScheduleClause::Kind::Runtime && chunk) {
    context_.Say(GetContext().clauseSource,
        "SCHEDULE(RUNTIME) on %s directive cannot have a chunk size; "
        "chunk size is determined by OMP_SCHEDULE environment variable"_err_en_US,
        directiveName.str());
    return;
  }
}

// BAD: Generic error without context
void OmpStructureChecker::CheckScheduleClause(
    const parser::OmpClause::Schedule &scheduleClause) {
  
  auto [modifiers, kind, chunk] = scheduleClause.t;
  
  if (kind.v == parser::OmpScheduleClause::Kind::Auto && chunk) {
    context_.Say(GetContext().clauseSource,
        "AUTO cannot have chunk size"_err_en_US);  // Which directive? What clause?
    return;
  }
}
```

**Why**: Users need to know which directive/clause triggered the error.

---

### Principle 10: Test Both Valid and Invalid Cases

**Good Practice**: Comprehensive test coverage for semantic checks

**Test File**: `flang/test/Semantics/OpenMP/schedule-errors.f90`

```fortran
! RUN: %python %S/test_errors.py %s %flang_fc1 -fopenmp

program test_schedule
  integer :: i, n, chunk
  
  !$omp parallel do schedule(static, 10)
  do i = 1, 100
    ! Valid: static with chunk size
  end do
  
  !$omp parallel do schedule(dynamic)
  do i = 1, 100
    ! Valid: dynamic without chunk size
  end do
  
  !$omp parallel do schedule(dynamic, chunk)
  do i = 1, 100
    ! Valid: dynamic with variable chunk size
  end do
  
  !ERROR: SCHEDULE(AUTO) on DO directive cannot have a chunk size
  !$omp parallel do schedule(auto, 10)
  do i = 1, 100
  end do
  
  !ERROR: SCHEDULE(RUNTIME) on DO directive cannot have a chunk size
  !$omp parallel do schedule(runtime, chunk)
  do i = 1, 100
  end do
  
  !ERROR: NUM_THREADS expression must be positive, got -1
  !$omp parallel do num_threads(-1)
  do i = 1, 100
  end do
  
  !ERROR: Threadprivate variable 'tp_var' cannot appear in REDUCTION clause
  integer :: tp_var
  !$omp threadprivate(tp_var)
  !$omp parallel do reduction(+:tp_var)
  do i = 1, 100
    tp_var = tp_var + i
  end do

end program
```

**Why**: Tests document expected behavior and catch regressions.

---

### Additional Semantic Check Patterns

#### Pattern: Required vs Optional Clauses

```cpp
// Check required clause is present
void OmpStructureChecker::Leave(
    const parser::OpenMPThreadprivate &threadprivate) {
  
  // Threadprivate must have variable list (not optional)
  if (GetContext().objectList.empty()) {
    context_.Say(GetContext().directiveSource,
        "THREADPRIVATE directive requires at least one variable"_err_en_US);
  }
}
```

#### Pattern: Directive-Specific Clause Validation

```cpp
// Different validation rules for same clause on different directives
void OmpStructureChecker::Enter(const parser::OmpClause::Collapse &collapse) {
  CheckAllowed(llvm::omp::Clause::OMPC_collapse);
  
  auto n = GetIntValue(collapse.v);
  if (!n || *n <= 0) {
    context_.Say(GetContext().clauseSource,
        "COLLAPSE parameter must be a positive integer constant"_err_en_US);
    return;
  }
  
  const auto directive = GetContext().directive;
  
  // SIMD has different restrictions
  if (directive == llvm::omp::Directive::OMPD_simd) {
    // Check perfectly nested loops for SIMD
    CheckPerfectlyNestedLoops(*n, /*allowInterveningCode=*/false);
  } else {
    // Worksharing loops allow some intervening code
    CheckPerfectlyNestedLoops(*n, /*allowInterveningCode=*/true);
  }
}
```

#### Pattern: Cross-Clause Dependencies

```cpp
// Check that dependent clauses are consistent
void OmpStructureChecker::Leave(const parser::OmpClauseList &clauses) {
  // LINEAR requires STEP if used with certain schedule kinds
  if (HasClause(llvm::omp::Clause::OMPC_linear)) {
    auto scheduleKind = GetScheduleKind();
    auto linearStep = GetLinearStep();
    
    if (scheduleKind == ScheduleKind::Static && !linearStep) {
      context_.Say(GetContext().clauseSource,
          "LINEAR clause with SCHEDULE(STATIC) should specify a step value"_warn_en_US);
    }
  }
}
```

#### Pattern: Version-Specific Features

```cpp
// Check OpenMP version for new features
void OmpStructureChecker::Enter(const parser::OmpClause::Affinity &affinity) {
  CheckAllowed(llvm::omp::Clause::OMPC_affinity);
  
  // AFFINITY clause is OpenMP 5.0+
  if (context_.languageFeatures().GetOpenMPVersion() < 50) {
    context_.Say(GetContext().clauseSource,
        "AFFINITY clause requires OpenMP 5.0 or later"_err_en_US);
    return;
  }
  
  // Validate affinity iterator and variable list
  ValidateAffinityClause(affinity);
}
```

---

### Handling Circular Dependencies Between Clauses

Circular dependencies occur when clause validation depends on information from other clauses that haven't been processed yet, or when multiple clauses depend on each other's validation state. This section shows how to detect and resolve these dependencies.

---

#### Problem: Order-Dependent Validation

**Scenario**: Two clauses need information from each other

```cpp
// PROBLEMATIC: Circular dependency between DEPEND and AFFINITY clauses
void OmpStructureChecker::Enter(const parser::OmpClause::Depend &depend) {
  CheckAllowed(llvm::omp::Clause::OMPC_depend);
  
  // BUG: Depends on affinity clause being processed first
  if (HasClause(llvm::omp::Clause::OMPC_affinity)) {
    // Validate that DEPEND variables don't conflict with AFFINITY
    CheckDependAffinityConflict(depend);  // Might not work if affinity not yet processed!
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Affinity &affinity) {
  CheckAllowed(llvm::omp::Clause::OMPC_affinity);
  
  // BUG: Depends on depend clause being processed first
  if (HasClause(llvm::omp::Clause::OMPC_depend)) {
    // Validate that AFFINITY variables don't conflict with DEPEND
    CheckAffinityDependConflict(affinity);  // Might not work if depend not yet processed!
  }
}

// Result: Depending on clause order in source code, validation might be incomplete!
```

---

#### Solution 1: Defer Cross-Clause Validation to Leave()

**Best Practice**: Collect clause information during `Enter()`, validate relationships in `Leave()`

```cpp
// GOOD: Collect information in Enter(), validate in Leave()

// Store clause information for later validation
struct DependInfo {
  std::vector<const Symbol*> dependVars;
  parser::CharBlock source;
};

struct AffinityInfo {
  std::vector<const Symbol*> affinityVars;
  parser::CharBlock source;
};

class OmpStructureChecker {
private:
  std::optional<DependInfo> dependInfo_;
  std::optional<AffinityInfo> affinityInfo_;
  
public:
  void Enter(const parser::OmpClause::Depend &depend) {
    CheckAllowed(llvm::omp::Clause::OMPC_depend);
    
    // Just collect information, don't cross-validate yet
    DependInfo info;
    info.source = GetContext().clauseSource;
    
    for (const auto &depObj : depend.v) {
      // Collect variables from depend clause
      if (const auto *symbol = GetDependVariable(depObj)) {
        info.dependVars.push_back(symbol);
      }
    }
    
    dependInfo_ = std::move(info);
  }
  
  void Enter(const parser::OmpClause::Affinity &affinity) {
    CheckAllowed(llvm::omp::Clause::OMPC_affinity);
    
    // Just collect information, don't cross-validate yet
    AffinityInfo info;
    info.source = GetContext().clauseSource;
    
    for (const auto &var : affinity.v.v) {
      if (const auto *symbol = ResolveOmp(*var, SymbolFlag::ObjectName, context_)) {
        info.affinityVars.push_back(symbol);
      }
    }
    
    affinityInfo_ = std::move(info);
  }
  
  void Leave(const parser::OmpClauseList &clauses) {
    // Now all clauses have been processed - validate relationships
    if (dependInfo_ && affinityInfo_) {
      CheckDependAffinityInteraction(*dependInfo_, *affinityInfo_);
    }
    
    // Clear for next directive
    dependInfo_.reset();
    affinityInfo_.reset();
  }
  
  void CheckDependAffinityInteraction(
      const DependInfo &depend,
      const AffinityInfo &affinity) {
    
    // Check for overlapping variables
    for (const auto *depVar : depend.dependVars) {
      for (const auto *affVar : affinity.affinityVars) {
        if (depVar == affVar) {
          context_.Say(affinity.source,
              "Variable '%s' appears in both DEPEND and AFFINITY clauses"_err_en_US,
              depVar->name().ToString());
          context_.Say(depend.source,
              "Previous DEPEND clause here"_note_en_US);
        }
      }
    }
  }
};
```

**Why**: All clause information is collected before validation, eliminating order dependency.

---

#### Solution 2: Two-Pass Validation

**Use Case**: When validation logic is too complex for simple deferred checking

```cpp
// GOOD: Explicit two-pass approach for complex validations

class OmpStructureChecker {
private:
  // Pass 1: Collect all clause information
  struct ClauseRegistry {
    std::optional<ScheduleInfo> schedule;
    std::optional<OrderedInfo> ordered;
    std::optional<CollapseInfo> collapse;
    std::vector<LinearInfo> linear;
    std::vector<ReductionInfo> reduction;
    // ... other clauses
  };
  
  ClauseRegistry registry_;
  
public:
  // Pass 1: Enter methods just collect
  void Enter(const parser::OmpClause::Schedule &x) {
    CheckAllowed(llvm::omp::Clause::OMPC_schedule);
    registry_.schedule = ExtractScheduleInfo(x);
  }
  
  void Enter(const parser::OmpClause::Ordered &x) {
    CheckAllowed(llvm::omp::Clause::OMPC_ordered);
    registry_.ordered = ExtractOrderedInfo(x);
  }
  
  void Enter(const parser::OmpClause::Linear &x) {
    CheckAllowed(llvm::omp::Clause::OMPC_linear);
    registry_.linear.push_back(ExtractLinearInfo(x));
  }
  
  // Pass 2: Leave performs comprehensive validation
  void Leave(const parser::OmpClauseList &clauses) {
    // Phase 1: Individual clause validation
    if (registry_.schedule) {
      ValidateSchedule(*registry_.schedule);
    }
    if (registry_.ordered) {
      ValidateOrdered(*registry_.ordered);
    }
    
    // Phase 2: Pairwise clause interactions
    if (registry_.schedule && registry_.ordered) {
      ValidateScheduleOrdered(*registry_.schedule, *registry_.ordered);
    }
    
    if (registry_.ordered && registry_.collapse) {
      ValidateOrderedCollapse(*registry_.ordered, *registry_.collapse);
    }
    
    // Phase 3: Multi-clause interactions
    if (registry_.schedule && !registry_.linear.empty()) {
      ValidateScheduleLinear(*registry_.schedule, registry_.linear);
    }
    
    // Phase 4: Global consistency checks
    ValidateGlobalConsistency(registry_);
    
    // Clear registry for next directive
    registry_ = ClauseRegistry{};
  }
  
private:
  void ValidateScheduleOrdered(
      const ScheduleInfo &schedule,
      const OrderedInfo &ordered) {
    
    // Can now safely check both clauses
    if (schedule.kind == ScheduleKind::Auto && ordered.hasParameter) {
      context_.Say(ordered.source,
          "ORDERED with parameter cannot be used with SCHEDULE(AUTO)"_err_en_US);
      context_.Say(schedule.source,
          "SCHEDULE(AUTO) specified here"_note_en_US);
    }
  }
  
  void ValidateScheduleLinear(
      const ScheduleInfo &schedule,
      const std::vector<LinearInfo> &linearClauses) {
    
    // Complex validation involving multiple LINEAR clauses and SCHEDULE
    if (schedule.kind == ScheduleKind::Dynamic) {
      for (const auto &linear : linearClauses) {
        if (!linear.hasExplicitStep) {
          context_.Say(linear.source,
              "LINEAR clause with SCHEDULE(DYNAMIC) should specify explicit step"_warn_en_US);
        }
      }
    }
  }
};
```

**Why**: Explicit two-pass approach makes validation logic clear and maintainable.

---

#### Solution 3: Dependency Graph for Complex Scenarios

**Use Case**: Many interdependent clauses requiring specific validation order

```cpp
// ADVANCED: Build dependency graph for validation order

class OmpClauseValidator {
private:
  struct ClauseNode {
    llvm::omp::Clause clauseKind;
    parser::CharBlock source;
    std::vector<llvm::omp::Clause> dependencies;  // What this clause needs
    std::function<void()> validationFunc;
  };
  
  std::vector<ClauseNode> validationGraph_;
  std::map<llvm::omp::Clause, bool> validated_;
  
public:
  void RegisterClauseValidation(
      llvm::omp::Clause clause,
      parser::CharBlock source,
      std::vector<llvm::omp::Clause> deps,
      std::function<void()> validationFunc) {
    
    validationGraph_.push_back(ClauseNode{
      clause, source, std::move(deps), std::move(validationFunc)
    });
  }
  
  void ValidateAllClauses() {
    // Topological sort to determine validation order
    std::vector<ClauseNode*> sortedNodes;
    TopologicalSort(validationGraph_, sortedNodes);
    
    // Validate in dependency order
    for (auto *node : sortedNodes) {
      // Check all dependencies are satisfied
      bool canValidate = true;
      for (auto dep : node->dependencies) {
        if (!validated_[dep]) {
          context_.Say(node->source,
              "Internal error: clause validation dependency not satisfied"_err_en_US);
          canValidate = false;
          break;
        }
      }
      
      if (canValidate) {
        node->validationFunc();
        validated_[node->clauseKind] = true;
      }
    }
    
    // Detect circular dependencies
    if (sortedNodes.size() < validationGraph_.size()) {
      ReportCircularDependency();
    }
  }
  
private:
  void TopologicalSort(
      const std::vector<ClauseNode> &graph,
      std::vector<ClauseNode*> &sorted) {
    
    std::set<const ClauseNode*> visited;
    std::set<const ClauseNode*> inProgress;
    
    for (const auto &node : graph) {
      if (visited.find(&node) == visited.end()) {
        if (!DFSVisit(&node, visited, inProgress, sorted)) {
          // Circular dependency detected
          return;
        }
      }
    }
  }
  
  bool DFSVisit(
      const ClauseNode *node,
      std::set<const ClauseNode*> &visited,
      std::set<const ClauseNode*> &inProgress,
      std::vector<ClauseNode*> &sorted) {
    
    if (inProgress.find(node) != inProgress.end()) {
      // Circular dependency!
      return false;
    }
    
    if (visited.find(node) != visited.end()) {
      return true;  // Already processed
    }
    
    inProgress.insert(node);
    
    // Visit dependencies first
    for (auto depKind : node->dependencies) {
      const ClauseNode *depNode = FindNode(depKind);
      if (depNode && !DFSVisit(depNode, visited, inProgress, sorted)) {
        return false;
      }
    }
    
    inProgress.erase(node);
    visited.insert(node);
    sorted.push_back(const_cast<ClauseNode*>(node));
    
    return true;
  }
  
  void ReportCircularDependency() {
    context_.Say(GetContext().directiveSource,
        "Internal compiler error: circular dependency in clause validation"_err_en_US);
    // In production code, this indicates a bug in the compiler
  }
};

// Usage example:
void OmpStructureChecker::Leave(const parser::OmpClauseList &clauses) {
  OmpClauseValidator validator;
  
  if (HasClause(llvm::omp::Clause::OMPC_schedule)) {
    validator.RegisterClauseValidation(
        llvm::omp::Clause::OMPC_schedule,
        scheduleSource_,
        {},  // No dependencies
        [this]() { ValidateScheduleClause(); });
  }
  
  if (HasClause(llvm::omp::Clause::OMPC_ordered)) {
    validator.RegisterClauseValidation(
        llvm::omp::Clause::OMPC_ordered,
        orderedSource_,
        {llvm::omp::Clause::OMPC_schedule, llvm::omp::Clause::OMPC_collapse},
        [this]() { ValidateOrderedClause(); });
  }
  
  if (HasClause(llvm::omp::Clause::OMPC_linear)) {
    validator.RegisterClauseValidation(
        llvm::omp::Clause::OMPC_linear,
        linearSource_,
        {llvm::omp::Clause::OMPC_schedule},
        [this]() { ValidateLinearClause(); });
  }
  
  // Validate all in correct order
  validator.ValidateAllClauses();
}
```

**Why**: Handles arbitrarily complex dependencies while detecting circular references.

---

#### Pattern: Breaking Dependency Cycles

**Problem**: Sometimes dependencies truly are circular

```cpp
// Example: REDUCTION and LASTPRIVATE may have complex interactions

// DON'T try to make one depend on the other - they're truly independent
// but have a combined constraint

// GOOD: Validate each independently, then check combined constraint
void OmpStructureChecker::Leave(const parser::OmpClauseList &clauses) {
  // Collect information from both clauses
  std::set<const Symbol*> reductionVars;
  std::set<const Symbol*> lastprivateVars;
  
  if (HasClause(llvm::omp::Clause::OMPC_reduction)) {
    reductionVars = GetReductionVariables();
  }
  
  if (HasClause(llvm::omp::Clause::OMPC_lastprivate)) {
    lastprivateVars = GetLastprivateVariables();
  }
  
  // Now check the combined constraint
  for (const auto *var : reductionVars) {
    if (lastprivateVars.count(var)) {
      // OpenMP allows this but the semantics are complex
      context_.Say(GetContext().clauseSource,
          "Variable '%s' appears in both REDUCTION and LASTPRIVATE; "
          "last iteration value will be used after reduction"_warn_en_US,
          var->name().ToString());
    }
  }
}
```

---

#### Common Dependency Patterns

**Pattern 1: SCHEDULE → LINEAR**
```cpp
// LINEAR step validation may depend on SCHEDULE kind
// Solution: Validate LINEAR in Leave() after SCHEDULE collected
```

**Pattern 2: COLLAPSE → ORDERED**
```cpp
// ORDERED(n) parameter must match COLLAPSE(n)
// Solution: Check in Leave() when both are available
if (collapseCount && orderedCount && *collapseCount != *orderedCount) {
  context_.Say(..., "ORDERED and COLLAPSE counts must match"_err_en_US);
}
```

**Pattern 3: Data-Sharing ↔ REDUCTION**
```cpp
// Variables can't be in both PRIVATE and REDUCTION
// Solution: Track all data-sharing attributes, validate uniqueness in Leave()
std::map<const Symbol*, std::vector<llvm::omp::Clause>> varToClause;
// ... collect all data-sharing clauses ...
for (const auto &[var, clauses] : varToClause) {
  if (clauses.size() > 1) {
    ReportDataSharingConflict(var, clauses);
  }
}
```

**Pattern 4: IF clause with multiple directive-name-modifiers**
```cpp
// IF(parallel:cond1) IF(simd:cond2) - validate all modifiers match directive
// Solution: Collect all IF clauses, validate directive applicability in Leave()
```

---

#### Debugging Circular Dependencies

**Technique 1: Add Validation Logging**

```cpp
#define LOG_VALIDATION(msg) \
  llvm::dbgs() << "VALIDATION[" << __LINE__ << "]: " << msg << "\n"

void OmpStructureChecker::Enter(const parser::OmpClause::Schedule &x) {
  LOG_VALIDATION("Entering SCHEDULE clause validation");
  CheckAllowed(llvm::omp::Clause::OMPC_schedule);
  
  if (HasClause(llvm::omp::Clause::OMPC_ordered)) {
    LOG_VALIDATION("SCHEDULE checking ORDERED - ordered present");
  } else {
    LOG_VALIDATION("SCHEDULE checking ORDERED - ordered absent");
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Ordered &x) {
  LOG_VALIDATION("Entering ORDERED clause validation");
  CheckAllowed(llvm::omp::Clause::OMPC_ordered);
  
  if (HasClause(llvm::omp::Clause::OMPC_schedule)) {
    LOG_VALIDATION("ORDERED checking SCHEDULE - schedule present");
  } else {
    LOG_VALIDATION("ORDERED checking SCHEDULE - schedule absent");
  }
}
```

**Output** (reveals order-dependency):
```
VALIDATION[123]: Entering SCHEDULE clause validation
VALIDATION[125]: SCHEDULE checking ORDERED - ordered absent  ← BUG: Ordered not processed yet!
VALIDATION[234]: Entering ORDERED clause validation
VALIDATION[236]: ORDERED checking SCHEDULE - schedule present
```

**Technique 2: Write Ordering-Agnostic Tests**

```fortran
! Test file: test-clause-order.f90
! Both orderings should produce identical validation

! Order 1: SCHEDULE then ORDERED
!$omp do schedule(static) ordered
do i = 1, 100
end do

! Order 2: ORDERED then SCHEDULE
!$omp do ordered schedule(static)
do i = 1, 100
end do

! Both should work identically - if not, you have order-dependency bug!
```

---

### Best Practices Summary

**Avoid Circular Dependencies**:
1. ✅ **Defer cross-clause validation to `Leave()`** - simplest and most common solution
2. ✅ **Use two-pass validation** - collect first, validate second
3. ✅ **Build dependency graph** - for complex multi-clause interactions
4. ✅ **Test with different clause orderings** - catch order dependencies early

**When Dependencies Are Necessary**:
1. ✅ Make dependencies explicit (document which clause needs what)
2. ✅ Validate in topological order
3. ✅ Detect and report true circular dependencies
4. ✅ Break cycles by validating combined constraints instead of individual implications

**Common Mistakes**:
1. ❌ Checking `HasClause()` in `Enter()` methods
2. ❌ Assuming clause processing order matches source order
3. ❌ Not testing with reversed clause orders
4. ❌ Validating relationships before all information collected

---

### Debugging Semantic Checks

#### Enable Semantic Debug Output

```bash
# Run semantic analysis with verbose output
flang-new -fc1 -fopenmp -fsyntax-only -fdebug-semantics test.f90

# Check specific OpenMP semantic errors
flang-new -fc1 -fopenmp -fsyntax-only test.f90 2>&1 | grep -i "openmp\|omp"
```

#### Add Debug Output in Checker

```cpp
void OmpStructureChecker::Enter(const parser::OmpClause::Private &x) {
  #ifdef DEBUG_OMP_SEMANTICS
  llvm::errs() << "Checking PRIVATE clause on directive: "
               << llvm::omp::getOpenMPDirectiveName(GetContext().directive)
               << "\n";
  llvm::errs() << "Variable count: " << x.v.v.size() << "\n";
  #endif
  
  CheckAllowed(llvm::omp::Clause::OMPC_private);
  
  for (const auto &name : x.v.v) {
    #ifdef DEBUG_OMP_SEMANTICS
    llvm::errs() << "  Checking variable: " << name.ToString() << "\n";
    #endif
    CheckPrivateVar(name);
  }
}
```

#### Test Harness for Semantic Checks

```python
# In test_errors.py test harness
# Verifies that expected errors are produced

# Usage in .f90 test files:
# !ERROR: <expected error message>
# !WARNING: <expected warning message>
# !NOTE: <expected note message>
```

---

### Summary: Semantic Validation Checklist

When writing OpenMP semantic checks:

**Structure**:
- [ ] Check `CheckAllowed()` first before validating clause content
- [ ] Use `Enter()` for clause-specific checks
- [ ] Use `Leave()` for cross-clause validation and mutual exclusivity
- [ ] Handle composite directives separately from individual directives

**Error Messages**:
- [ ] Specific and actionable (explain what's wrong and how to fix)
- [ ] Include variable/clause/directive names in messages
- [ ] Provide context (which directive, which clause)
- [ ] Use appropriate severity (error vs warning vs note)

**Symbol Validation**:
- [ ] Check for null symbols (handle unresolved gracefully)
- [ ] Validate symbol attributes (PARAMETER, ALLOCATABLE, POINTER, etc.)
- [ ] Check threadprivate status where relevant
- [ ] Verify symbol type matches clause requirements

**Expression Validation**:
- [ ] Check expression type (integer, logical, etc.)
- [ ] Validate constness where required
- [ ] Check value ranges for constants (positive, non-zero, etc.)
- [ ] Warn about side effects in expressions

**Testing**:
- [ ] Test valid cases (should compile without errors)
- [ ] Test invalid cases (should produce specific errors)
- [ ] Test edge cases (empty lists, extreme values, etc.)
- [ ] Document expected errors in test files using `!ERROR:` comments

---

## Implementing Composite Constructs (PARALLEL DO, TARGET TEAMS, etc.)

### Overview: What are Composite Constructs?

**Composite constructs** combine multiple OpenMP directives into a single directive for convenience and efficiency. Instead of writing nested directives, users can specify a composite form.

**Common Composite Constructs**:
- `PARALLEL DO` = `PARALLEL` + `DO`
- `PARALLEL SECTIONS` = `PARALLEL` + `SECTIONS`
- `PARALLEL WORKSHARE` = `PARALLEL` + `WORKSHARE`
- `TARGET PARALLEL` = `TARGET` + `PARALLEL`
- `TARGET TEAMS` = `TARGET` + `TEAMS`
- `TARGET TEAMS DISTRIBUTE` = `TARGET` + `TEAMS` + `DISTRIBUTE`
- `TARGET TEAMS DISTRIBUTE PARALLEL DO` = `TARGET` + `TEAMS` + `DISTRIBUTE` + `PARALLEL` + `DO`
- `PARALLEL DO SIMD` = `PARALLEL` + `DO` + `SIMD`

**Fortran Example**:
```fortran
! Composite form (concise)
!$omp parallel do private(x)
do i = 1, n
  x = a(i) * b(i)
  c(i) = x + d(i)
end do
!$omp end parallel do

! Equivalent nested form (verbose)
!$omp parallel private(x)
  !$omp do
  do i = 1, n
    x = a(i) * b(i)
    c(i) = x + d(i)
  end do
  !$omp end do
!$omp end parallel
```

**Benefits**:
1. **Conciseness**: Less code to write
2. **Efficiency**: Runtime can optimize combined construct
3. **Clarity**: Intent is clearer (parallelize this loop)
4. **Restrictions**: Some clauses only allowed on composite form

---

### Key Differences: Composite vs Nested

**Clause Placement Rules Differ**:
```fortran
! PARALLEL DO: clauses can apply to either part
!$omp parallel do private(x) schedule(static)
! - PRIVATE applies to PARALLEL
! - SCHEDULE applies to DO

! Nested form: must distribute clauses
!$omp parallel private(x)
  !$omp do schedule(static)
  ! ...
  !$omp end do
!$omp end parallel
```

**Implicit Barriers**:
```fortran
! PARALLEL DO: single implicit barrier at end
!$omp parallel do
! ... loop ...
!$omp end parallel do
! <-- Implicit barrier here (end of composite construct)

! Nested form: barrier at end of worksharing, then end of parallel
!$omp parallel
  !$omp do
  ! ... loop ...
  !$omp end do
  ! <-- Implicit barrier here (end of worksharing)
!$omp end parallel
! <-- Implicit barrier here (end of parallel)
```

**Specific Restrictions**:
- `ORDERED` clause behavior differs on composite vs separate
- `NOWAIT` on composite cancels both barriers
- Some clauses forbidden on composite that are allowed on separate

---

### Implementation Strategy: Two Approaches

#### Approach 1: Single MLIR Operation (Preferred for Simple Composites)

**Concept**: Create a dedicated MLIR operation for the composite construct

**Example**: `PARALLEL DO` → `omp.parallel.loop`

**Advantages**:
- Single operation easier to analyze/optimize
- Matches source code structure
- Runtime can optimize combined construct

**Disadvantages**:
- Need separate operation for each composite
- More code to maintain
- Complex composites become unwieldy

**When to Use**:
- Common composites (PARALLEL DO, TARGET TEAMS)
- Simple combinations (2-3 directives)
- Performance-critical patterns

---

#### Approach 2: Nested MLIR Operations (Preferred for Complex Composites)

**Concept**: Lower composite to nested MLIR operations

**Example**: `PARALLEL DO` → `omp.parallel { omp.wsloop { ... } }`

**Advantages**:
- Reuses existing operations
- Works for arbitrary composites
- Less code to maintain
- Easier to implement new composites

**Disadvantages**:
- May miss optimization opportunities
- More complex IR structure
- Harder to pattern-match for optimizations

**When to Use**:
- Complex composites (3+ directives)
- Rare combinations
- Device composites (TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD)

---

### Implementation: PARALLEL DO (Nested Approach)

This walkthrough implements PARALLEL DO using nested operations, which is the approach Flang uses for most composites.

---

#### Stage 1: Parser - Recognize Composite Directive

**File**: `llvm/include/llvm/Frontend/OpenMP/OMPKinds.def`

```cpp
// Composite directive definition
__OMP_DIRECTIVE_EXT(parallel_do, OMPD_parallel_do)
```

**File**: `flang/lib/Parser/openmp-parsers.cpp`

The parser recognizes `PARALLEL DO` as a single directive:

```cpp
// Parser recognizes composite loop constructs
TYPE_PARSER(
  construct<OpenMPLoopConstruct>(
    Parser<OmpBeginLoopDirective>{} / 
    Parser<OmpLoopDirective>{}))

// OmpBeginLoopDirective handles both DO and composite forms
TYPE_PARSER(
  sourced(construct<OmpBeginLoopDirective>(
    "PARALLEL DO" >> pure(llvm::omp::Directive::OMPD_parallel_do) ||
    "DO SIMD" >> pure(llvm::omp::Directive::OMPD_do_simd) ||
    "PARALLEL DO SIMD" >> pure(llvm::omp::Directive::OMPD_parallel_do_simd) ||
    // ... other loop directives
    "DO" >> pure(llvm::omp::Directive::OMPD_do))))
```

**Parse Tree Structure**:
```cpp
// Same structure as regular loop construct
struct OpenMPLoopConstruct {
  OmpBeginLoopDirective beginDirective;  // Contains OMPD_parallel_do
  parser::DoConstruct doConstruct;       // The Fortran loop
  OmpEndLoopDirective endDirective;      // END PARALLEL DO
};
```

**Key Point**: Parser treats composite as atomic unit, not nested constructs

---

#### Stage 2: Semantic Validation - Composite-Specific Checks

**File**: `flang/lib/Semantics/check-omp-structure.cpp`

```cpp
void OmpStructureChecker::Enter(const parser::OpenMPLoopConstruct &x) {
  const auto &beginDir = std::get<parser::OmpBeginLoopDirective>(x.t);
  const auto directive = beginDir.v;
  
  PushContext(directive.source, directive.v);
  
  // Check if this is a composite construct
  if (directive.v == llvm::omp::Directive::OMPD_parallel_do) {
    CheckParallelDoRestrictions();
  } else if (directive.v == llvm::omp::Directive::OMPD_parallel_do_simd) {
    CheckParallelDoSimdRestrictions();
  } else if (directive.v == llvm::omp::Directive::OMPD_do_simd) {
    CheckDoSimdRestrictions();
  }
  
  // Standard loop checks
  CheckLoopStructure(x);
}

void OmpStructureChecker::CheckParallelDoRestrictions() {
  // PARALLEL DO specific restrictions
  
  // 1. ORDERED clause restrictions
  if (HasClause(llvm::omp::Clause::OMPC_ordered)) {
    auto orderedClause = FindClause(llvm::omp::Clause::OMPC_ordered);
    
    // ORDERED(n) not allowed on PARALLEL DO in some OpenMP versions
    if (orderedClause && HasOrderedParameter(orderedClause)) {
      context_.Say(GetContext().clauseSource,
          "ORDERED clause with parameter not allowed on PARALLEL DO; "
          "use separate PARALLEL and DO directives"_err_en_US);
    }
  }
  
  // 2. Check clause applicability
  // Some clauses apply to PARALLEL part, some to DO part
  const auto &clauses = GetContext().clauses;
  for (const auto &clause : clauses) {
    ValidateClauseOnComposite(clause, 
        llvm::omp::Directive::OMPD_parallel_do);
  }
  
  // 3. COPYIN only makes sense on PARALLEL part
  if (HasClause(llvm::omp::Clause::OMPC_copyin)) {
    // Valid - COPYIN applies to implicit PARALLEL
  }
  
  // 4. SCHEDULE only makes sense on DO part
  if (HasClause(llvm::omp::Clause::OMPC_schedule)) {
    // Valid - SCHEDULE applies to implicit DO
  }
  
  // 5. NUM_THREADS applies to PARALLEL part
  if (HasClause(llvm::omp::Clause::OMPC_num_threads)) {
    // Valid - applies to team creation
  }
}

void OmpStructureChecker::ValidateClauseOnComposite(
    const parser::OmpClause &clause,
    llvm::omp::Directive composite) {
  
  auto clauseKind = GetClauseKind(clause);
  
  // Determine which part of composite this clause applies to
  // Based on OpenMP spec tables
  
  bool allowedOnParallel = IsAllowedOnDirective(
      clauseKind, llvm::omp::Directive::OMPD_parallel);
  bool allowedOnDo = IsAllowedOnDirective(
      clauseKind, llvm::omp::Directive::OMPD_do);
  
  if (!allowedOnParallel && !allowedOnDo) {
    context_.Say(GetContext().clauseSource,
        "Clause '%s' not allowed on PARALLEL DO construct"_err_en_US,
        GetClauseName(clauseKind));
  }
}
```

**Key Validations**:
1. Check composite-specific restrictions (e.g., ORDERED parameter)
2. Validate each clause is allowed on at least one component
3. Handle clauses that apply to specific parts (SCHEDULE → DO, NUM_THREADS → PARALLEL)
4. Check for conflicts between parallel and worksharing clauses

---

#### Stage 3: MLIR Lowering - Decompose to Nested Operations

**File**: `flang/lib/Lower/OpenMP/OpenMP.cpp`

```cpp
void genOpenMPConstruct(
    lower::AbstractConverter &converter,
    lower::SymMap &symMap,
    semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval,
    const parser::OpenMPLoopConstruct &loopConstruct) {
  
  const auto &beginDir = std::get<parser::OmpBeginLoopDirective>(loopConstruct.t);
  const auto directive = beginDir.v.v;
  
  if (directive == llvm::omp::Directive::OMPD_parallel_do) {
    genParallelDoConstruct(converter, symMap, semaCtx, eval, loopConstruct);
  } else if (directive == llvm::omp::Directive::OMPD_parallel_do_simd) {
    genParallelDoSimdConstruct(converter, symMap, semaCtx, eval, loopConstruct);
  } else {
    genLoopConstruct(converter, symMap, semaCtx, eval, loopConstruct);
  }
}

void genParallelDoConstruct(
    lower::AbstractConverter &converter,
    lower::SymMap &symMap,
    semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval,
    const parser::OpenMPLoopConstruct &loopConstruct) {
  
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  
  const auto &beginDir = std::get<parser::OmpBeginLoopDirective>(loopConstruct.t);
  const auto &clauses = std::get<parser::OmpClauseList>(beginDir.t);
  
  // Step 1: Partition clauses into PARALLEL and DO parts
  ClauseProcessor cp(converter, clauses);
  
  llvm::SmallVector<const parser::OmpClause*> parallelClauses;
  llvm::SmallVector<const parser::OmpClause*> doClauses;
  
  for (const auto &clause : clauses.v) {
    auto kind = GetClauseKind(clause);
    
    // Determine which directive this clause applies to
    if (IsParallelClause(kind)) {
      parallelClauses.push_back(&clause);
    }
    if (IsWorkshareClause(kind)) {
      doClauses.push_back(&clause);
    }
    // Some clauses (like PRIVATE) apply to both
  }
  
  // Step 2: Process PARALLEL-specific clauses
  mlir::Value ifExpr, numThreads;
  llvm::SmallVector<mlir::Value> reductionVars;
  llvm::SmallVector<mlir::Attribute> reductionSymbols;
  
  cp.processIf(llvm::omp::Directive::OMPD_parallel, ifExpr);
  cp.processNumThreads(numThreads);
  cp.processCopyin();
  
  // Process data-sharing clauses (apply to parallel)
  llvm::SmallVector<mlir::Value> privateVars, firstprivateVars;
  cp.processPrivate(privateVars);
  cp.processFirstprivate(firstprivateVars);
  
  // REDUCTION applies to both parallel and loop
  cp.processReduction(loc, reductionVars, reductionSymbols);
  
  // Step 3: Create omp.parallel operation
  auto parallelOp = firOpBuilder.create<mlir::omp::ParallelOp>(
      loc,
      ifExpr,
      numThreads,
      privateVars,
      firstprivateVars,
      /*shared=*/mlir::ValueRange{},
      reductionVars,
      reductionSymbols);
  
  // Step 4: Create region for parallel
  mlir::Block *parallelBlock = firOpBuilder.createBlock(&parallelOp.getRegion());
  firOpBuilder.setInsertionPointToStart(parallelBlock);
  
  // Step 5: Inside parallel region, create omp.wsloop for DO part
  auto loopOp = genLoopOp(converter, symMap, loc, doClauses, loopConstruct);
  
  // Step 6: Lower loop body
  const auto &doConstruct = std::get<parser::DoConstruct>(loopConstruct.t);
  genLoopBody(converter, symMap, loopOp, doConstruct);
  
  // Step 7: Terminate regions
  firOpBuilder.create<mlir::omp::TerminatorOp>(loc);  // End wsloop
  firOpBuilder.setInsertionPointAfter(loopOp);
  firOpBuilder.create<mlir::omp::TerminatorOp>(loc);  // End parallel
}

bool IsParallelClause(llvm::omp::Clause clauseKind) {
  switch (clauseKind) {
    case llvm::omp::Clause::OMPC_num_threads:
    case llvm::omp::Clause::OMPC_if:  // Can apply to both
    case llvm::omp::Clause::OMPC_default:
    case llvm::omp::Clause::OMPC_shared:
    case llvm::omp::Clause::OMPC_private:
    case llvm::omp::Clause::OMPC_firstprivate:
    case llvm::omp::Clause::OMPC_copyin:
    case llvm::omp::Clause::OMPC_reduction:
      return true;
    default:
      return false;
  }
}

bool IsWorkshareClause(llvm::omp::Clause clauseKind) {
  switch (clauseKind) {
    case llvm::omp::Clause::OMPC_schedule:
    case llvm::omp::Clause::OMPC_ordered:
    case llvm::omp::Clause::OMPC_collapse:
    case llvm::omp::Clause::OMPC_nowait:
    case llvm::omp::Clause::OMPC_private:
    case llvm::omp::Clause::OMPC_lastprivate:
    case llvm::omp::Clause::OMPC_reduction:
      return true;
    default:
      return false;
  }
}
```

**Generated MLIR** (for PARALLEL DO):
```mlir
func.func @parallel_do_example(%arg0: !fir.ref<!fir.array<100xi32>>) {
  %c1 = arith.constant 1 : i32
  %c100 = arith.constant 100 : i32
  %c1_step = arith.constant 1 : i32
  
  // Outer parallel operation
  omp.parallel {
    // Inner worksharing loop
    omp.wsloop schedule(static) {
      omp.loop_nest (%i) : i32 = (%c1) to (%c100) step (%c1_step) {
        // Loop body
        %addr = fir.coordinate_of %arg0, %i : (!fir.ref<!fir.array<100xi32>>, i32) -> !fir.ref<i32>
        %val = fir.load %addr : !fir.ref<i32>
        %result = arith.muli %val, %c2 : i32
        fir.store %result to %addr : !fir.ref<i32>
        
        omp.yield
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}
```

**Key Points**:
1. Composite is decomposed into nested operations
2. Clauses are partitioned to appropriate directive
3. Data-sharing attributes propagate correctly
4. REDUCTION handled on both levels (parallel + loop)

---

#### Stage 4: LLVM IR Translation - Nested Runtime Calls

**File**: `mlir/lib/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.cpp`

The nested MLIR operations translate to nested runtime calls:

```cpp
// Translation of nested parallel + wsloop
static llvm::Error convertOmpParallel(
    mlir::omp::ParallelOp parallelOp,
    llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) {
  
  // Create outlined function for parallel region
  llvm::Function *outlinedFn = outlineParallelRegion(
      parallelOp, builder, moduleTranslation);
  
  // Generate fork_call
  llvm::Value *numThreads = ...;
  builder.CreateCall(
      getForkCallFunction(),
      {getLocationInfo(), getThreadId(), outlinedFn, numThreads, ...});
  
  return llvm::Error::success();
}

// Inside outlined parallel function, wsloop translates to loop runtime calls
static llvm::Error convertOmpWsloop(
    mlir::omp::WsloopOp wsloopOp,
    llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) {
  
  // Extract schedule
  auto schedule = wsloopOp.getSchedule();
  
  if (schedule == ScheduleKind::Static) {
    // Static scheduling
    builder.CreateCall(
        getStaticInitFunction(),
        {getLocationInfo(), getThreadId(), 
         getScheduleType(), getLowerBound(), getUpperBound(), ...});
  } else {
    // Dynamic/guided scheduling
    builder.CreateCall(
        getDispatchInitFunction(),
        {getLocationInfo(), getThreadId(), getScheduleType(), ...});
  }
  
  // Generate loop body with runtime bounds
  // ...
  
  // Loop finalization
  builder.CreateCall(getStaticFiniFunction(), {...});
  
  return llvm::Error::success();
}
```

**Generated LLVM IR** (simplified):
```llvm
define void @parallel_do_example(ptr %arr) {
entry:
  ; Create parallel region via fork_call
  call void @__kmpc_fork_call(
      ptr @.loc,
      i32 1,                    ; 1 captured variable
      ptr @.omp_outlined.,      ; Outlined function
      ptr %arr)                 ; Captured array
  ret void
}

; Outlined function for parallel region
define internal void @.omp_outlined.(
    ptr %tid.addr,
    ptr %bound_tid.addr,
    ptr %arr) {
entry:
  %tid = load i32, ptr %tid.addr
  
  ; Worksharing loop initialization (static schedule)
  %lb = alloca i32
  %ub = alloca i32
  %stride = alloca i32
  %is_last = alloca i32
  
  store i32 1, ptr %lb
  store i32 100, ptr %ub
  store i32 1, ptr %stride
  
  call void @__kmpc_for_static_init_4(
      ptr @.loc,
      i32 %tid,
      i32 34,          ; Schedule kind: static
      ptr %is_last,
      ptr %lb,
      ptr %ub,
      ptr %stride,
      i32 1,           ; Increment
      i32 1)           ; Chunk size
  
  ; Load bounds computed by runtime
  %lb_val = load i32, ptr %lb
  %ub_val = load i32, ptr %ub
  
  ; Loop over assigned iterations
  br label %loop.header
  
loop.header:
  %i = phi i32 [ %lb_val, %entry ], [ %i.next, %loop.body ]
  %cmp = icmp sle i32 %i, %ub_val
  br i1 %cmp, label %loop.body, label %loop.exit
  
loop.body:
  ; Loop body: arr[i] = arr[i] * 2
  %idx = sext i32 %i to i64
  %addr = getelementptr i32, ptr %arr, i64 %idx
  %val = load i32, ptr %addr
  %result = mul i32 %val, 2
  store i32 %result, ptr %addr
  
  %i.next = add i32 %i, 1
  br label %loop.header
  
loop.exit:
  ; Worksharing loop finalization
  call void @__kmpc_for_static_fini(ptr @.loc, i32 %tid)
  
  ; Implicit barrier at end of worksharing
  call void @__kmpc_barrier(ptr @.loc, i32 %tid)
  
  ret void
}
```

**Key Points**:
1. `omp.parallel` → `__kmpc_fork_call` with outlined function
2. `omp.wsloop` (inside outlined) → `__kmpc_for_static_init` + loop + `__kmpc_for_static_fini`
3. Implicit barrier after worksharing (unless NOWAIT)
4. Thread gets assigned subset of iterations via static_init

---

### Implementation: TARGET TEAMS DISTRIBUTE (Complex Composite)

This example shows a more complex 3-level composite for device execution.

#### Fortran Source
```fortran
!$omp target teams distribute map(tofrom: a)
do i = 1, n
  a(i) = a(i) * 2
end do
!$omp end target teams distribute
```

#### MLIR Lowering (Nested 3-Level)

```cpp
void genTargetTeamsDistribute(
    lower::AbstractConverter &converter,
    const parser::OpenMPLoopConstruct &loopConstruct) {
  
  // Level 1: TARGET
  auto targetOp = createTargetOp(converter, mapClauses);
  
  // Enter target region
  auto &targetRegion = targetOp.getRegion();
  builder.setInsertionPointToStart(&targetRegion.front());
  
  // Level 2: TEAMS (inside target)
  auto teamsOp = createTeamsOp(converter, numTeamsClauses);
  
  // Enter teams region
  auto &teamsRegion = teamsOp.getRegion();
  builder.setInsertionPointToStart(&teamsRegion.front());
  
  // Level 3: DISTRIBUTE (inside teams)
  auto distributeOp = createDistributeOp(converter, loopConstruct);
  
  // Lower loop body into distribute
  genLoopBody(converter, distributeOp, loopConstruct);
  
  // Terminate all regions
  builder.create<omp::TerminatorOp>();  // End distribute
  builder.setInsertionPointAfter(distributeOp);
  builder.create<omp::TerminatorOp>();  // End teams
  builder.setInsertionPointAfter(teamsOp);
  builder.create<omp::TerminatorOp>();  // End target
}
```

#### Generated MLIR
```mlir
omp.target map_entries(%map_a -> %arg0 : !fir.ref<!fir.array<1024xi32>>) {
  omp.teams num_teams(%c32 : i32) {
    omp.distribute {
      omp.loop_nest (%i) : i32 = (%c1) to (%c1024) step (%c1_step) {
        // Loop body
        omp.yield
      }
      omp.terminator
    }
    omp.terminator
  }
  omp.terminator
}
```

#### LLVM IR Translation
```llvm
; Host-side: Setup and launch kernel
define void @target_teams_distribute(ptr %a) {
  ; Map data to device
  call void @__tgt_target_data_begin(...)
  
  ; Launch kernel with teams configuration
  %result = call i32 @__tgt_target_kernel(
      i64 -1,                           ; device
      ptr @.kernel_outlined,            ; kernel function
      i32 32,                           ; num_teams
      i32 0,                            ; thread_limit (default)
      ...)
  
  ; Map data back
  call void @__tgt_target_data_end(...)
  ret void
}

; Device kernel
define void @.kernel_outlined(ptr %a) #0 {
  ; Get team ID
  %team_id = call i32 @__kmpc_get_team_num()
  %num_teams = call i32 @__kmpc_get_num_teams()
  
  ; Distribute iterations across teams
  ; Each team processes chunk: [start, end)
  %chunk_size = udiv i32 1024, %num_teams
  %start = mul i32 %team_id, %chunk_size
  %end = add i32 %start, %chunk_size
  
  ; Loop over team's chunk
  br label %loop
  
loop:
  %i = phi i32 [ %start, %entry ], [ %i.next, %loop ]
  ; Process a[i]
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, %end
  br i1 %done, label %exit, label %loop
  
exit:
  ret void
}

attributes #0 = { "omp_target_kernel" }
```

---

### Clause Partitioning Guide

Different clauses apply to different parts of composite constructs. This table shows the mapping:

| Clause | PARALLEL | DO/SIMD | TEAMS | TARGET | DISTRIBUTE |
|--------|----------|---------|-------|--------|------------|
| PRIVATE | ✓ | ✓ | ✓ | ✓ | ✓ |
| FIRSTPRIVATE | ✓ | ✓ | ✓ | ✓ | ✓ |
| LASTPRIVATE | | ✓ | | | ✓ |
| SHARED | ✓ | | ✓ | | |
| REDUCTION | ✓ | ✓ | ✓ | | ✓ |
| NUM_THREADS | ✓ | | | | |
| THREAD_LIMIT | | | ✓ | | |
| NUM_TEAMS | | | ✓ | | |
| SCHEDULE | | ✓ | | | |
| COLLAPSE | | ✓ | | | ✓ |
| ORDERED | | ✓ | | | |
| NOWAIT | | ✓ | | | |
| IF | ✓ | | ✓ | ✓ | |
| MAP | | | | ✓ | |
| DEVICE | | | | ✓ | |
| DIST_SCHEDULE | | | | | ✓ |

**Implementation Helper**:
```cpp
bool clauseAppliesToDirective(
    llvm::omp::Clause clause,
    llvm::omp::Directive directive) {
  
  // Use LLVM's built-in mapping table
  return llvm::omp::isAllowedClauseForDirective(
      directive, clause, /*version=*/52);
}

std::vector<llvm::omp::Directive> getComponentDirectives(
    llvm::omp::Directive composite) {
  
  switch (composite) {
    case llvm::omp::Directive::OMPD_parallel_do:
      return {OMPD_parallel, OMPD_do};
    
    case llvm::omp::Directive::OMPD_target_teams:
      return {OMPD_target, OMPD_teams};
    
    case llvm::omp::Directive::OMPD_target_teams_distribute:
      return {OMPD_target, OMPD_teams, OMPD_distribute};
    
    case llvm::omp::Directive::OMPD_parallel_do_simd:
      return {OMPD_parallel, OMPD_do, OMPD_simd};
    
    // ... etc
  }
}
```

---

### Testing Composite Constructs

#### Test Structure
**File**: `flang/test/Lower/OpenMP/parallel-do.f90`

```fortran
! RUN: %flang_fc1 -emit-mlir -fopenmp %s -o - | FileCheck %s

! Test basic PARALLEL DO
subroutine test_parallel_do(n, a, b)
  integer :: n, i
  real :: a(n), b(n)
  
  ! CHECK-LABEL: func @_QPtest_parallel_do
  ! CHECK: omp.parallel {
  ! CHECK:   omp.wsloop {
  ! CHECK:     omp.loop_nest {{.*}} {
  ! CHECK:       omp.yield
  ! CHECK:     }
  ! CHECK:     omp.terminator
  ! CHECK:   }
  ! CHECK:   omp.terminator
  ! CHECK: }
  
  !$omp parallel do
  do i = 1, n
    a(i) = b(i) + 1.0
  end do
  !$omp end parallel do
end subroutine

! Test PARALLEL DO with clauses
subroutine test_parallel_do_clauses(n, a, b, chunk)
  integer :: n, i, chunk
  real :: a(n), b(n), temp
  
  ! CHECK-LABEL: func @_QPtest_parallel_do_clauses
  ! CHECK: omp.parallel
  ! CHECK-SAME: num_threads(%{{.*}} : i32)
  ! CHECK-SAME: private({{.*}} : {{.*}})
  ! CHECK: {
  ! CHECK:   omp.wsloop
  ! CHECK-SAME: schedule(dynamic, %{{.*}} : i32)
  ! CHECK:   {
  
  !$omp parallel do num_threads(4) private(temp) schedule(dynamic, chunk)
  do i = 1, n
    temp = b(i) * 2.0
    a(i) = temp + 1.0
  end do
  !$omp end parallel do
end subroutine

! Test PARALLEL DO with REDUCTION
subroutine test_parallel_do_reduction(n, a, sum)
  integer :: n, i
  real :: a(n), sum
  
  ! CHECK-LABEL: func @_QPtest_parallel_do_reduction
  ! CHECK: omp.parallel
  ! CHECK-SAME: reduction(@add_reduction_f32
  ! CHECK: {
  ! CHECK:   omp.wsloop
  ! CHECK-SAME: reduction(@add_reduction_f32
  
  sum = 0.0
  !$omp parallel do reduction(+:sum)
  do i = 1, n
    sum = sum + a(i)
  end do
  !$omp end parallel do
end subroutine
```

#### Semantic Tests
**File**: `flang/test/Semantics/OpenMP/parallel-do-errors.f90`

```fortran
! RUN: %python %S/test_errors.py %s %flang_fc1 -fopenmp

subroutine test_ordered_restriction()
  integer :: i
  
  ! ERROR: ORDERED clause with parameter not allowed on PARALLEL DO
  !$omp parallel do ordered(1)
  do i = 1, 100
    !$omp ordered
    print *, i
    !$omp end ordered
  end do
  !$omp end parallel do
end subroutine

subroutine test_invalid_clause()
  integer :: i
  
  ! ERROR: COPYPRIVATE clause not allowed on PARALLEL DO
  !$omp parallel do copyprivate(i)
  do i = 1, 100
  end do
  !$omp end parallel do
end subroutine
```

---

### Common Issues with Composite Constructs

#### Issue 1: Clause Applied to Wrong Component

**Problem**:
```cpp
// BUG: SCHEDULE applied to PARALLEL instead of DO
void genParallelDo(...) {
  // Wrong: scheduleClause goes to parallel
  auto parallelOp = builder.create<omp::ParallelOp>(
      loc, ifExpr, numThreads, scheduleAttr);  // BUG!
}
```

**Fix**:
```cpp
// Correct: partition clauses
ClauseProcessor cp(converter, clauses);

// PARALLEL clauses
cp.processNumThreads(numThreads);

// Create parallel
auto parallelOp = builder.create<omp::ParallelOp>(...);

// DO clauses processed inside parallel region
cp.processSchedule(scheduleAttr);
auto wsloopOp = builder.create<omp::WsloopOp>(loc, scheduleAttr);
```

---

#### Issue 2: Missing Implicit Barrier

**Problem**:
```llvm
; BUG: No barrier at end of composite
define void @.omp_outlined.(...) {
  call void @__kmpc_for_static_init(...)
  ; ... loop ...
  call void @__kmpc_for_static_fini(...)
  ret void  ; BUG: Missing barrier!
}
```

**Fix**:
```llvm
; Correct: implicit barrier unless NOWAIT
define void @.omp_outlined.(...) {
  call void @__kmpc_for_static_init(...)
  ; ... loop ...
  call void @__kmpc_for_static_fini(...)
  
  ; Check for NOWAIT clause
  ; If not present, insert barrier
  call void @__kmpc_barrier(ptr @.loc, i32 %tid)
  
  ret void
}
```

---

#### Issue 3: Incorrect Nesting Order

**Problem**:
```mlir
// BUG: Wrong nesting order (DO outside PARALLEL)
omp.wsloop {
  omp.parallel {
    // This is wrong!
  }
}
```

**Fix**:
```mlir
// Correct: PARALLEL contains DO
omp.parallel {
  omp.wsloop {
    // Correct nesting
  }
}
```

**Validation**:
```cpp
void validateCompositeNesting(mlir::Operation *op) {
  if (auto wsloop = dyn_cast<omp::WsloopOp>(op)) {
    auto parent = wsloop->getParentOp();
    if (!isa<omp::ParallelOp>(parent)) {
      emitError(wsloop.getLoc(),
          "wsloop must be nested inside parallel for PARALLEL DO");
    }
  }
}
```

---

### Best Practices for Composite Constructs

**1. Reuse Existing Operations**
```cpp
// GOOD: Decompose to existing ops
genParallelOp(...);
  genWsloopOp(...);

// AVOID: Creating new composite operation unless necessary
// (Only for performance-critical common composites)
```

**2. Systematic Clause Partitioning**
```cpp
// GOOD: Use table-driven approach
struct ClauseMapping {
  llvm::omp::Clause clause;
  llvm::SmallVector<llvm::omp::Directive> appliesTo;
};

const ClauseMapping clauseMappings[] = {
  {OMPC_schedule, {OMPD_do, OMPD_simd}},
  {OMPC_num_threads, {OMPD_parallel}},
  {OMPC_private, {OMPD_parallel, OMPD_do, OMPD_teams}},
  // ...
};
```

**3. Validate Component Restrictions**
```cpp
// GOOD: Check restrictions for each component
void validateComposite(Directive composite, const ClauseList &clauses) {
  auto components = getComponentDirectives(composite);
  
  for (auto component : components) {
    validateDirectiveRestrictions(component, clauses);
  }
  
  validateCompositeSpecificRestrictions(composite, clauses);
}
```

**4. Test All Clause Combinations**
```fortran
! Test matrix: composite × clause combinations
!$omp parallel do schedule(static) private(x)
!$omp parallel do schedule(dynamic) reduction(+:sum)
!$omp parallel do num_threads(4) firstprivate(y)
! etc.
```

**5. Handle NOWAIT Correctly**
```cpp
// NOWAIT on composite cancels barriers at all levels
if (hasNowait) {
  // No barrier after DO
  // No barrier after PARALLEL
  wsloopOp.setNowaitAttr(builder.getUnitAttr());
}
```

---

## Memory Management and Privatization Strategies for Complex Data Types

### Overview: Data-Sharing Attributes

OpenMP provides several data-sharing attributes that control how variables are accessed in parallel regions:

- **SHARED**: Variable is shared across all threads (default for most variables)
- **PRIVATE**: Each thread gets its own uninitialized copy
- **FIRSTPRIVATE**: Each thread gets its own copy, initialized from master thread's value
- **LASTPRIVATE**: Thread executing last iteration writes back to original variable
- **THREADPRIVATE**: Persistent private storage across parallel regions

**Challenge**: For complex data types (arrays, derived types, allocatables, pointers), privatization requires careful memory management to handle descriptors, deep copies, and deallocation.

---

### Memory Model: Fortran Descriptors

Fortran represents complex data types using **descriptors** that contain metadata:

```cpp
// FIR representation of a descriptor
struct BoxType {
  void* base_addr;       // Pointer to data
  size_t elem_size;      // Size of each element
  int rank;              // Array rank (dimensions)
  Dimension dims[rank];  // Per-dimension metadata
  int type_code;         // Type information
  int attribute;         // Allocatable, pointer, etc.
};

struct Dimension {
  int64_t lower_bound;
  int64_t extent;        // Number of elements
  int64_t stride;        // Byte offset between elements
};
```

**Example**: Array `REAL :: A(10:20)` has descriptor:
- `base_addr` → memory containing 11 reals
- `rank` = 1
- `dims[0].lower_bound` = 10
- `dims[0].extent` = 11
- `dims[0].stride` = 4 (sizeof(float))

---

### Strategy 1: Scalar Privatization (Simple Case)

#### Fortran Code
```fortran
!$omp parallel private(x)
  integer :: x
  x = omp_get_thread_num()
  print *, x
!$omp end parallel
```

#### MLIR Lowering

**File**: `flang/lib/Lower/OpenMP/OpenMP.cpp`

```cpp
void processPrivateClause(
    lower::AbstractConverter &converter,
    const parser::OmpClause::Private &privateClause) {
  
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  
  for (const auto &obj : privateClause.v.v) {
    const auto *symbol = ResolveOmpObject(obj, converter);
    
    // For scalars, privatization is simple: allocate stack storage
    if (symbol->GetType()->IsNumeric() || symbol->GetType()->IsLogical()) {
      // Get type
      mlir::Type varType = converter.genType(*symbol);
      
      // Allocate private copy (in outlined function, not here)
      // Mark symbol as needing privatization
      markAsPrivate(symbol, /*needsAllocation=*/true);
    }
  }
}

// In outlined function generation
void genOutlinedFunction(...) {
  // For each private variable
  for (auto *privateVar : privateVars) {
    // Allocate on thread's stack
    mlir::Value privateAlloca = builder.create<fir::AllocaOp>(
        loc,
        privateVar->getType(),
        /*name=*/privateVar->getName());
    
    // Map symbol to private storage
    symMap.addSymbol(*privateVar->getSymbol(), privateAlloca);
  }
  
  // Now references to 'x' use privateAlloca, not original variable
}
```

**Generated MLIR**:
```mlir
// Outlined parallel function
func.func private @_QPomp_outlined(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>) {
  // Private variable allocation
  %x_private = fir.alloca i32
  
  // Use private copy
  %tid = fir.call @omp_get_thread_num() : () -> i32
  fir.store %tid to %x_private : !fir.ref<i32>
  
  return
}
```

**LLVM IR**:
```llvm
define internal void @_QPomp_outlined(ptr %tid, ptr %bound_tid) {
entry:
  ; Private variable on stack
  %x_private = alloca i32, align 4
  
  ; Get thread ID
  %tid_val = call i32 @omp_get_thread_num()
  
  ; Store to private copy
  store i32 %tid_val, ptr %x_private, align 4
  
  ret void
}
```

**Key Points**:
- Private scalars: simple `alloca` in outlined function
- Each thread has its own stack frame → automatic isolation
- No initialization (PRIVATE semantics)
- Automatically deallocated when function returns

---

### Strategy 2: FIRSTPRIVATE - Initialization Required

#### Fortran Code
```fortran
integer :: x = 100

!$omp parallel firstprivate(x)
  x = x + omp_get_thread_num()
  print *, x
!$omp end parallel
```

#### Implementation

**File**: `flang/lib/Lower/OpenMP/OpenMP.cpp`

```cpp
void processFirstprivateClause(
    lower::AbstractConverter &converter,
    const parser::OmpClause::Firstprivate &firstprivateClause,
    llvm::SmallVector<mlir::Value> &firstprivateVars) {
  
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  
  for (const auto &obj : firstprivateClause.v.v) {
    const auto *symbol = ResolveOmpObject(obj, converter);
    
    // Get original value
    mlir::Value originalAddr = converter.getSymbolAddress(*symbol);
    
    // Load original value (to pass to outlined function)
    mlir::Value originalValue = builder.create<fir::LoadOp>(
        loc, originalAddr);
    
    // Add to firstprivate list (passed to outlined function)
    firstprivateVars.push_back(originalValue);
  }
}

// In outlined function
void genOutlinedFunctionWithFirstprivate(..., 
    const llvm::SmallVector<mlir::Value> &firstprivateVals) {
  
  // Add block arguments for firstprivate variables
  mlir::Block *block = builder.createBlock(&parallelOp.getRegion());
  for (auto fpVal : firstprivateVals) {
    block->addArgument(fpVal.getType(), loc);
  }
  
  // Inside block: allocate private storage and initialize
  for (auto [idx, privateVar] : llvm::enumerate(privateVars)) {
    mlir::Value privateAlloca = builder.create<fir::AllocaOp>(
        loc, privateVar->getType());
    
    // Initialize from passed value
    mlir::Value initValue = block->getArgument(idx);
    builder.create<fir::StoreOp>(loc, initValue, privateAlloca);
    
    symMap.addSymbol(*privateVar->getSymbol(), privateAlloca);
  }
}
```

**Generated MLIR**:
```mlir
func.func @test_firstprivate() {
  %c100 = arith.constant 100 : i32
  %x_orig = fir.alloca i32
  fir.store %c100 to %x_orig : !fir.ref<i32>
  
  // Load original value
  %x_val = fir.load %x_orig : !fir.ref<i32>
  
  // Parallel with firstprivate
  omp.parallel {
    // Outlined function receives value as argument
    call @_QPomp_outlined(%x_val) : (i32) -> ()
    omp.terminator
  }
  return
}

func.func private @_QPomp_outlined(%x_init: i32) {
  // Allocate private copy
  %x_private = fir.alloca i32
  
  // Initialize from passed value
  fir.store %x_init to %x_private : !fir.ref<i32>
  
  // Use private copy
  %tid = fir.call @omp_get_thread_num() : () -> i32
  %x_val = fir.load %x_private : !fir.ref<i32>
  %x_new = arith.addi %x_val, %tid : i32
  fir.store %x_new to %x_private : !fir.ref<i32>
  
  return
}
```

**LLVM IR**:
```llvm
define void @test_firstprivate() {
entry:
  %x_orig = alloca i32
  store i32 100, ptr %x_orig
  
  ; Load value to pass
  %x_val = load i32, ptr %x_orig
  
  ; Fork with firstprivate value
  call void @__kmpc_fork_call(
      ptr @.loc,
      i32 1,                    ; 1 captured variable
      ptr @_QPomp_outlined,
      i32 %x_val)               ; Pass by value
  
  ret void
}

define internal void @_QPomp_outlined(ptr %tid, ptr %bound_tid, i32 %x_init) {
  ; Allocate private copy
  %x_private = alloca i32
  
  ; Initialize from passed value
  store i32 %x_init, ptr %x_private
  
  ; Use private copy
  %tid_val = call i32 @omp_get_thread_num()
  %x_val = load i32, ptr %x_private
  %x_new = add i32 %x_val, %tid_val
  store i32 %x_new, ptr %x_private
  
  ret void
}
```

**Key Points**:
- Original value loaded in parent function
- Passed by value to outlined function
- Private copy allocated and initialized in outlined function
- Works for scalars; arrays need different approach

---

### Strategy 3: Array Privatization

#### Fortran Code
```fortran
real :: temp(100)

!$omp parallel private(temp)
  integer :: i
  do i = 1, 100
    temp(i) = compute_value(i)
  end do
  call process_array(temp)
!$omp end parallel
```

#### Challenge

Arrays require:
1. **Descriptor allocation**: Create new descriptor for private copy
2. **Data allocation**: Allocate memory for array elements
3. **Descriptor initialization**: Set bounds, strides, base address
4. **Deallocation**: Free memory when region ends

#### Implementation

**File**: `flang/lib/Lower/OpenMP/OpenMP.cpp`

```cpp
void genPrivateArray(
    lower::AbstractConverter &converter,
    const semantics::Symbol &arraySymbol,
    mlir::Value &privateDescriptor) {
  
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  
  // Get array type information
  const auto &arrayDetails = arraySymbol.get<semantics::ObjectEntityDetails>();
  const auto &shape = arrayDetails.shape();
  
  // Extract array bounds
  llvm::SmallVector<mlir::Value> lowerBounds, upperBounds;
  for (const auto &dim : shape) {
    auto lb = getConstantBound(dim.lbound());
    auto ub = getConstantBound(dim.ubound());
    lowerBounds.push_back(lb);
    upperBounds.push_back(ub);
  }
  
  // Compute array size
  mlir::Value arraySize = computeArraySize(builder, loc, shape);
  
  // Get element type
  mlir::Type elementType = converter.genType(arraySymbol);
  
  // Allocate private array storage
  mlir::Value privateData = builder.create<fir::AllocMemOp>(
      loc,
      fir::HeapType::get(elementType),
      /*typeparams=*/mlir::ValueRange{},
      /*shape=*/arraySize);
  
  // Create descriptor for private array
  mlir::Type boxType = fir::BoxType::get(
      fir::SequenceType::get(shape, elementType));
  
  privateDescriptor = builder.create<fir::EmboxOp>(
      loc,
      boxType,
      privateData,
      /*shape=*/mlir::ValueRange{},  // Will set from lbound/extent
      /*slice=*/mlir::Value{},
      /*typeparams=*/mlir::ValueRange{});
  
  // Set descriptor metadata (bounds, strides)
  setDescriptorBounds(builder, loc, privateDescriptor, 
                      lowerBounds, upperBounds);
}

// Need to deallocate at end of region
void genPrivateArrayDeallocation(
    fir::FirOpBuilder &builder,
    mlir::Location loc,
    mlir::Value privateDescriptor) {
  
  // Extract base address from descriptor
  mlir::Value baseAddr = builder.create<fir::BoxAddrOp>(
      loc, privateDescriptor);
  
  // Free allocated memory
  builder.create<fir::FreeMemOp>(loc, baseAddr);
}
```

**Generated MLIR**:
```mlir
func.func private @_QPomp_outlined_with_array() {
  %c100 = arith.constant 100 : index
  
  // Allocate heap memory for private array
  %temp_data = fir.allocmem !fir.array<100xf32>
  
  // Create descriptor
  %temp_private = fir.embox %temp_data : (!fir.heap<!fir.array<100xf32>>) 
      -> !fir.box<!fir.array<100xf32>>
  
  // Use private array
  %c1 = arith.constant 1 : index
  omp.wsloop (%i) : index = (%c1) to (%c100) step (%c1) {
    %val = fir.call @compute_value(%i) : (index) -> f32
    %addr = fir.coordinate_of %temp_private, %i : (!fir.box<!fir.array<100xf32>>, index) -> !fir.ref<f32>
    fir.store %val to %addr : !fir.ref<f32>
    omp.yield
  }
  
  fir.call @process_array(%temp_private) : (!fir.box<!fir.array<100xf32>>) -> ()
  
  // Deallocate private array
  %temp_addr = fir.box_addr %temp_private : (!fir.box<!fir.array<100xf32>>) -> !fir.heap<!fir.array<100xf32>>
  fir.freemem %temp_addr : !fir.heap<!fir.array<100xf32>>
  
  return
}
```

**LLVM IR**:
```llvm
define internal void @_QPomp_outlined_with_array(ptr %tid, ptr %bound_tid) {
entry:
  ; Allocate heap memory for private array (100 floats)
  %temp_data = call ptr @malloc(i64 400)  ; 100 * 4 bytes
  
  ; Initialize descriptor
  %temp_desc = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }
  %desc_base = getelementptr { ptr, ... }, ptr %temp_desc, i32 0, i32 0
  store ptr %temp_data, ptr %desc_base
  ; ... set rank, bounds, strides ...
  
  ; Use array
  br label %loop
  
loop:
  ; ... loop using temp_data ...
  br i1 %done, label %exit, label %loop
  
exit:
  ; Call process_array with descriptor
  call void @process_array(ptr %temp_desc)
  
  ; Free private array
  call void @free(ptr %temp_data)
  
  ret void
}
```

**Key Points**:
- Arrays require **heap allocation** (`fir.allocmem` → `malloc`)
- Descriptor created with `fir.embox`
- Must **explicitly deallocate** with `fir.freemem` → `free`
- Bounds/strides copied from original descriptor

---

### Strategy 4: FIRSTPRIVATE Arrays (Deep Copy)

#### Fortran Code
```fortran
real :: template(100)
template = initial_values()

!$omp parallel firstprivate(template)
  ! Each thread has initialized copy of template
  call modify_array(template)
!$omp end parallel
```

#### Implementation: Deep Copy Required

```cpp
void genFirstprivateArray(
    lower::AbstractConverter &converter,
    const semantics::Symbol &arraySymbol,
    mlir::Value originalDescriptor,
    mlir::Value &privateDescriptor) {
  
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  
  // Step 1: Allocate private array (same as PRIVATE)
  genPrivateArray(converter, arraySymbol, privateDescriptor);
  
  // Step 2: Copy data from original to private (DEEP COPY)
  
  // Get base addresses
  mlir::Value origData = builder.create<fir::BoxAddrOp>(
      loc, originalDescriptor);
  mlir::Value privateData = builder.create<fir::BoxAddrOp>(
      loc, privateDescriptor);
  
  // Get array size
  mlir::Value arraySize = getArraySize(builder, loc, originalDescriptor);
  
  // Generate memcpy
  builder.create<fir::CallOp>(
      loc,
      builder.getSymbolRefAttr("memcpy"),
      mlir::ValueRange{privateData, origData, arraySize});
  
  // Alternative: Element-wise copy loop for non-contiguous arrays
  if (!isContiguous(arraySymbol)) {
    genElementWiseCopy(builder, loc, originalDescriptor, privateDescriptor);
  }
}

void genElementWiseCopy(
    fir::FirOpBuilder &builder,
    mlir::Location loc,
    mlir::Value srcDescriptor,
    mlir::Value dstDescriptor) {
  
  // Get bounds
  auto bounds = getArrayBounds(builder, loc, srcDescriptor);
  
  // Generate nested loops for multi-dimensional arrays
  // For each element: dst[i] = src[i]
  for (size_t dim = 0; dim < bounds.size(); ++dim) {
    auto [lb, ub] = bounds[dim];
    
    // Create loop
    auto loopOp = builder.create<fir::DoLoopOp>(
        loc, lb, ub, /*step=*/builder.createIntegerConstant(loc, 1));
    
    // Inside loop: copy element
    builder.setInsertionPointToStart(loopOp.getBody());
    mlir::Value idx = loopOp.getInductionVar();
    
    // src_addr = coordinate_of(src, idx)
    mlir::Value srcAddr = builder.create<fir::CoordinateOp>(
        loc, srcDescriptor, idx);
    
    // dst_addr = coordinate_of(dst, idx)
    mlir::Value dstAddr = builder.create<fir::CoordinateOp>(
        loc, dstDescriptor, idx);
    
    // Load and store
    mlir::Value val = builder.create<fir::LoadOp>(loc, srcAddr);
    builder.create<fir::StoreOp>(loc, val, dstAddr);
  }
}
```

**Generated MLIR**:
```mlir
func.func @test_firstprivate_array() {
  %template = fir.alloca !fir.array<100xf32>
  fir.call @initial_values(%template) : (!fir.ref<!fir.array<100xf32>>) -> ()
  
  omp.parallel {
    call @_QPomp_outlined(%template) : (!fir.ref<!fir.array<100xf32>>) -> ()
    omp.terminator
  }
  return
}

func.func private @_QPomp_outlined(%template_orig: !fir.ref<!fir.array<100xf32>>) {
  // Allocate private copy
  %template_data = fir.allocmem !fir.array<100xf32>
  %template_private = fir.embox %template_data 
      : (!fir.heap<!fir.array<100xf32>>) -> !fir.box<!fir.array<100xf32>>
  
  // Deep copy: memcpy or element-wise loop
  %size = arith.constant 400 : i64  ; 100 * sizeof(float)
  %orig_addr = fir.convert %template_orig : (!fir.ref<!fir.array<100xf32>>) -> !fir.ref<i8>
  %priv_addr = fir.box_addr %template_private : (...) -> !fir.heap<!fir.array<100xf32>>
  %priv_addr_i8 = fir.convert %priv_addr : (...) -> !fir.ref<i8>
  
  fir.call @memcpy(%priv_addr_i8, %orig_addr, %size) 
      : (!fir.ref<i8>, !fir.ref<i8>, i64) -> ()
  
  // Use private copy
  fir.call @modify_array(%template_private) : (!fir.box<!fir.array<100xf32>>) -> ()
  
  // Deallocate
  fir.freemem %template_data : !fir.heap<!fir.array<100xf32>>
  
  return
}
```

**Key Points**:
- **Deep copy** required: allocate + copy data
- For contiguous arrays: use `memcpy`
- For non-contiguous: element-wise copy loop
- Deallocation still required

---

### Strategy 5: Derived Types (Structures)

#### Fortran Code
```fortran
type :: particle
  real :: x, y, z
  real :: vx, vy, vz
  integer :: id
end type

type(particle) :: p

!$omp parallel firstprivate(p)
  p%x = p%x + p%vx * dt
  p%y = p%y + p%vy * dt
  p%z = p%z + p%vz * dt
!$omp end parallel
```

#### Implementation: Component-wise Copy

```cpp
void genFirstprivateDerivedType(
    lower::AbstractConverter &converter,
    const semantics::Symbol &derivedSymbol,
    mlir::Value originalAddr,
    mlir::Value &privateAddr) {
  
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  
  // Get derived type
  mlir::Type derivedType = converter.genType(derivedSymbol);
  
  // Allocate private copy
  privateAddr = builder.create<fir::AllocaOp>(loc, derivedType);
  
  // For each component: copy from original to private
  const auto &derivedDetails = derivedSymbol.GetType()
      ->derivedTypeSpec().typeSymbol();
  
  for (const auto &component : derivedDetails.get<semantics::DerivedTypeDetails>()
       .componentNames()) {
    
    const auto *compSymbol = derivedDetails.FindComponent(component);
    mlir::Type compType = converter.genType(*compSymbol);
    
    // Get field index
    auto fieldIndex = getFieldIndex(derivedType, component);
    
    // Access original component
    mlir::Value origCompAddr = builder.create<fir::FieldIndexOp>(
        loc, fir::ReferenceType::get(compType),
        originalAddr, component.ToString());
    
    // Access private component
    mlir::Value privCompAddr = builder.create<fir::FieldIndexOp>(
        loc, fir::ReferenceType::get(compType),
        privateAddr, component.ToString());
    
    // Copy value
    if (compSymbol->GetType()->IsNumeric() || 
        compSymbol->GetType()->IsLogical()) {
      // Simple scalar: load and store
      mlir::Value val = builder.create<fir::LoadOp>(loc, origCompAddr);
      builder.create<fir::StoreOp>(loc, val, privCompAddr);
    } else if (IsArray(*compSymbol)) {
      // Array component: recursive deep copy
      genFirstprivateArray(converter, *compSymbol, 
                           origCompAddr, privCompAddr);
    } else if (IsDerivedType(*compSymbol)) {
      // Nested derived type: recursive copy
      genFirstprivateDerivedType(converter, *compSymbol,
                                 origCompAddr, privCompAddr);
    }
  }
}
```

**Generated MLIR**:
```mlir
!particle_type = !fir.type<particle{x:f32,y:f32,z:f32,vx:f32,vy:f32,vz:f32,id:i32}>

func.func private @_QPomp_outlined(%p_orig: !fir.ref<!particle_type>) {
  // Allocate private copy
  %p_private = fir.alloca !particle_type
  
  // Copy each component
  %orig_x_addr = fir.field_index %p_orig, ["x"] : (!fir.ref<!particle_type>) -> !fir.ref<f32>
  %priv_x_addr = fir.field_index %p_private, ["x"] : (!fir.ref<!particle_type>) -> !fir.ref<f32>
  %x_val = fir.load %orig_x_addr : !fir.ref<f32>
  fir.store %x_val to %priv_x_addr : !fir.ref<f32>
  
  %orig_y_addr = fir.field_index %p_orig, ["y"] : (!fir.ref<!particle_type>) -> !fir.ref<f32>
  %priv_y_addr = fir.field_index %p_private, ["y"] : (!fir.ref<!particle_type>) -> !fir.ref<f32>
  %y_val = fir.load %orig_y_addr : !fir.ref<f32>
  fir.store %y_val to %priv_y_addr : !fir.ref<f32>
  
  // ... similar for z, vx, vy, vz, id ...
  
  // Use private copy
  // ... update positions ...
  
  return
}
```

**Key Points**:
- Component-wise copy for derived types
- Recursive handling for nested arrays/derived types
- Stack allocation (struct is value type)

---

### Strategy 6: Allocatable Variables

#### Fortran Code
```fortran
real, allocatable :: buffer(:)
allocate(buffer(1000))

!$omp parallel firstprivate(buffer)
  ! Each thread has own allocated copy
  call process(buffer)
!$omp end parallel
```

#### Challenge: Reference Counting and Descriptors

Allocatable variables have complex semantics:
1. **Descriptor** indicates allocation status
2. **Reference counting** for automatic deallocation
3. **Deep copy** required for FIRSTPRIVATE

```cpp
void genFirstprivateAllocatable(
    lower::AbstractConverter &converter,
    const semantics::Symbol &allocSymbol,
    mlir::Value originalBox,
    mlir::Value &privateBox) {
  
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  
  // Step 1: Check if original is allocated
  mlir::Value isAllocated = builder.create<fir::BoxIsAllocOp>(
      loc, builder.getI1Type(), originalBox);
  
  // Step 2: Conditional allocation
  auto ifOp = builder.create<fir::IfOp>(
      loc, isAllocated, /*withElseRegion=*/true);
  
  // Then block: allocated - make private copy
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  {
    // Get original data pointer and size
    mlir::Value origData = builder.create<fir::BoxAddrOp>(
        loc, originalBox);
    mlir::Value allocSize = builder.create<fir::BoxDimsOp>(
        loc, originalBox);
    
    // Allocate private storage
    mlir::Value privateData = builder.create<fir::AllocMemOp>(
        loc, fir::HeapType::get(elementType), allocSize);
    
    // Create private box/descriptor
    privateBox = builder.create<fir::EmboxOp>(
        loc, boxType, privateData, /*shape=*/allocSize);
    
    // Copy data (deep copy)
    builder.create<fir::CallOp>(
        loc,
        builder.getSymbolRefAttr("memcpy"),
        mlir::ValueRange{privateData, origData, allocSize});
    
    // Set reference count to 1 (private copy owned by thread)
    setReferenceCount(builder, loc, privateBox, /*refCount=*/1);
  }
  
  // Else block: not allocated - create unallocated descriptor
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  {
    // Create unallocated box
    privateBox = builder.create<fir::ZeroOp>(loc, boxType);
  }
  
  // After if: privateBox has correct allocation status
}

// At end of region: decrement reference count, deallocate if zero
void genAllocatableDeallocation(
    fir::FirOpBuilder &builder,
    mlir::Location loc,
    mlir::Value allocBox) {
  
  // Check if allocated
  mlir::Value isAllocated = builder.create<fir::BoxIsAllocOp>(
      loc, builder.getI1Type(), allocBox);
  
  auto ifOp = builder.create<fir::IfOp>(loc, isAllocated, /*withElse=*/false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  
  // Deallocate
  mlir::Value dataAddr = builder.create<fir::BoxAddrOp>(loc, allocBox);
  builder.create<fir::FreeMemOp>(loc, dataAddr);
}
```

**Generated MLIR**:
```mlir
func.func private @_QPomp_outlined(%buffer_orig: !fir.box<!fir.heap<!fir.array<?xf32>>>) {
  // Check if original is allocated
  %is_alloc = fir.box_isalloc %buffer_orig : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> i1
  
  %buffer_private = fir.if %is_alloc -> (!fir.box<!fir.heap<!fir.array<?xf32>>>) {
    // Allocated: make private copy
    %orig_data = fir.box_addr %buffer_orig : (...) -> !fir.heap<!fir.array<?xf32>>
    %size = fir.box_dims %buffer_orig : (...) -> index
    
    // Allocate private data
    %priv_data = fir.allocmem !fir.array<?xf32>, %size
    
    // Copy data
    %size_bytes = arith.muli %size, %c4 : index  ; sizeof(float)
    fir.call @memcpy(%priv_data, %orig_data, %size_bytes)
    
    // Create private box
    %priv_box = fir.embox %priv_data(%size) : (!fir.heap<!fir.array<?xf32>>, index) 
        -> !fir.box<!fir.heap<!fir.array<?xf32>>>
    fir.result %priv_box : !fir.box<!fir.heap<!fir.array<?xf32>>>
  } else {
    // Not allocated: create unallocated box
    %unalloc_box = fir.zero_bits !fir.box<!fir.heap<!fir.array<?xf32>>>
    fir.result %unalloc_box : !fir.box<!fir.heap<!fir.array<?xf32>>>
  }
  
  // Use private buffer
  fir.call @process(%buffer_private) : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> ()
  
  // Deallocate private copy
  %is_alloc_end = fir.box_isalloc %buffer_private : (...) -> i1
  fir.if %is_alloc_end {
    %data_addr = fir.box_addr %buffer_private : (...) -> !fir.heap<!fir.array<?xf32>>
    fir.freemem %data_addr : !fir.heap<!fir.array<?xf32>>
  }
  
  return
}
```

**Key Points**:
- Check allocation status before copying
- Deep copy for allocated variables
- Unallocated descriptor for unallocated variables
- Reference counting for automatic deallocation
- Explicit deallocation at end of region

---

### Strategy 7: LASTPRIVATE - Write-Back

#### Fortran Code
```fortran
real :: result

!$omp parallel do lastprivate(result)
do i = 1, n
  result = compute(i)
end do
!$omp end parallel do

print *, "Final result:", result  ! Value from last iteration
```

#### Implementation: Conditional Write-Back

```cpp
void genLastprivateWriteback(
    lower::AbstractConverter &converter,
    const semantics::Symbol &lastprivateSymbol,
    mlir::Value privateAddr,
    mlir::Value originalAddr,
    mlir::Value isLastIteration) {
  
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  
  // Only thread executing last iteration writes back
  auto ifOp = builder.create<fir::IfOp>(
      loc, isLastIteration, /*withElse=*/false);
  
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  
  // Copy private value back to original
  if (IsScalar(lastprivateSymbol)) {
    // Scalar: simple load and store
    mlir::Value val = builder.create<fir::LoadOp>(loc, privateAddr);
    builder.create<fir::StoreOp>(loc, val, originalAddr);
    
  } else if (IsArray(lastprivateSymbol)) {
    // Array: element-wise copy back
    genElementWiseCopy(builder, loc, privateAddr, originalAddr);
    
  } else if (IsDerivedType(lastprivateSymbol)) {
    // Derived type: component-wise copy back
    genComponentCopy(builder, loc, privateAddr, originalAddr);
  }
}

// Integrated into loop lowering
void genWsloopWithLastprivate(...) {
  // Generate loop
  auto loopOp = builder.create<omp::WsloopOp>(...);
  
  // Inside loop body: use private variable
  // ...
  
  // After loop: check if this thread did last iteration
  mlir::Value isLast = builder.create<fir::LoadOp>(loc, isLastIterPtr);
  
  // Write back if last iteration
  for (auto lastprivateVar : lastprivateVars) {
    genLastprivateWriteback(converter, *lastprivateVar, 
                            privateAddr, originalAddr, isLast);
  }
}
```

**Generated LLVM IR**:
```llvm
define internal void @_QPomp_outlined_lastprivate(
    ptr %tid, ptr %bound_tid, ptr %result_orig) {
  
  ; Allocate private copy
  %result_private = alloca float
  
  ; Worksharing loop init
  %is_last = alloca i32
  call void @__kmpc_for_static_init_4(
      ptr @.loc, i32 %tid_val, i32 34,
      ptr %is_last, ...)  ; is_last set by runtime
  
  ; Loop body using private copy
  ; ...
  
  ; After loop: check if last iteration
  %is_last_val = load i32, ptr %is_last
  %did_last = icmp ne i32 %is_last_val, 0
  br i1 %did_last, label %writeback, label %skip_writeback
  
writeback:
  ; Write private value back to original
  %final_val = load float, ptr %result_private
  store float %final_val, ptr %result_orig
  br label %skip_writeback
  
skip_writeback:
  call void @__kmpc_for_static_fini(ptr @.loc, i32 %tid_val)
  ret void
}
```

**Key Points**:
- Runtime sets `is_last` flag for thread executing final iteration
- Only that thread writes back to original variable
- Write-back happens **after** loop completes
- Supports scalars, arrays, derived types

---

### Performance Considerations

#### 1. Minimize Allocations

```fortran
! BAD: Large array privatized - each thread allocates 10MB
real :: workspace(1000000)
!$omp parallel private(workspace)
  ! Each thread allocates 10MB on heap
!$omp end parallel

! BETTER: Use smaller thread-local workspace or shared with synchronization
```

#### 2. Use SHARED When Possible

```fortran
! If threads only read, use SHARED
real :: lookup_table(1000)
lookup_table = initialize_table()

!$omp parallel shared(lookup_table)  ! Read-only - no copying
  value = lookup_table(compute_index())
!$omp end parallel
```

#### 3. Avoid FIRSTPRIVATE for Large Arrays

```fortran
! BAD: Deep copy 100MB array to each thread
real :: data(10000000)
!$omp parallel firstprivate(data)  ! Each thread copies 100MB!

! BETTER: Pass only what's needed
!$omp parallel
  integer :: tid
  tid = omp_get_thread_num()
  call process_chunk(data, tid)  ! Access shared, process different parts
!$omp end parallel
```

#### 4. Careful with Allocatables

```fortran
! Each thread allocates/deallocates
!$omp parallel
  real, allocatable :: buffer(:)
  allocate(buffer(10000))  ! Allocation overhead in each thread
  ! ...
  deallocate(buffer)
!$omp end parallel

! Better: Pre-allocate if possible
real, allocatable :: buffers(:,:)
allocate(buffers(10000, num_threads))
!$omp parallel
  integer :: tid
  tid = omp_get_thread_num() + 1
  call process(buffers(:, tid))  ! Each thread uses pre-allocated slice
!$omp end parallel
```

---

### Summary: Privatization Strategy Selection

| Data Type | PRIVATE | FIRSTPRIVATE | LASTPRIVATE | Notes |
|-----------|---------|--------------|-------------|-------|
| **Scalar** | Stack alloca | Pass by value + alloca | Alloca + conditional writeback | Simple, efficient |
| **Fixed Array** | Heap alloc + descriptor | Heap alloc + deep copy | Heap alloc + copy back | Requires memcpy |
| **Allocatable** | Descriptor copy | Check + conditional deep copy | Descriptor + writeback | Complex, expensive |
| **Pointer** | Shallow copy (ptr only) | Shallow copy | Shallow copy + writeback | No deep copy |
| **Derived Type** | Stack alloca | Component-wise copy | Component-wise + writeback | Recursive for nested |
| **Large Array** | ⚠️ Expensive | ⚠️ Very expensive | ⚠️ Very expensive | Consider SHARED |

**Decision Tree**:
```
Is variable shared across threads?
  YES → Use SHARED (no privatization)
  NO → Does it need initialization from master?
    YES → FIRSTPRIVATE (expensive for large data)
    NO → Does final value need to propagate?
      YES → LASTPRIVATE
      NO → PRIVATE (cheapest)
```

---

## Deprecated Patterns and Modern Replacements

This section documents deprecated or outdated implementation patterns in Flang's OpenMP compiler infrastructure and provides modern alternatives. Following these guidelines ensures code maintainability and compatibility with current LLVM/MLIR conventions.

---

### Pattern 1: Direct TableGen Definition Modification

#### ❌ DEPRECATED: Directly Editing Generated Files

```cpp
// WRONG: Editing auto-generated files
// File: llvm/include/llvm/Frontend/OpenMP/OMPKinds.gen (AUTO-GENERATED)

// Adding clause definition here - will be overwritten!
OMPC_my_new_clause,
```

**Problems**:
- Changes overwritten on next TableGen build
- Not traceable in source control
- Breaks build reproducibility

#### ✅ MODERN: Modify TableGen Source

**File**: `llvm/include/llvm/Frontend/OpenMP/OMPKinds.def`

```cpp
// Correct: Edit the source definition file
__OMP_CLAUSE(my_new_clause, OMPC_my_new_clause)
```

**Benefits**:
- Persists across rebuilds
- Single source of truth
- Proper version control

**Migration**:
```bash
# Find the .def file, not the .gen file
vim llvm/include/llvm/Frontend/OpenMP/OMPKinds.def

# Rebuild to regenerate .gen files
ninja llvm-tblgen
```

---

### Pattern 2: String-Based Directive/Clause Matching

#### ❌ DEPRECATED: String Comparison for Directives

```cpp
// WRONG: String-based matching (fragile, slow)
void processDirective(const std::string &directiveName) {
  if (directiveName == "parallel") {
    // Handle parallel
  } else if (directiveName == "do") {
    // Handle do
  } else if (directiveName == "parallel do") {
    // Handle parallel do
  }
  // Brittle, case-sensitive, error-prone
}
```

**Problems**:
- Case sensitivity issues
- Typos not caught at compile time
- No compile-time validation
- String allocations at runtime

#### ✅ MODERN: Enum-Based Directive Handling

```cpp
// Correct: Use llvm::omp::Directive enum
void processDirective(llvm::omp::Directive directive) {
  switch (directive) {
    case llvm::omp::Directive::OMPD_parallel:
      // Handle parallel
      break;
    case llvm::omp::Directive::OMPD_do:
      // Handle do
      break;
    case llvm::omp::Directive::OMPD_parallel_do:
      // Handle parallel do
      break;
    default:
      llvm_unreachable("Unknown directive");
  }
}

// Convert from string if needed
llvm::omp::Directive getDirectiveKind(llvm::StringRef name) {
  return llvm::omp::getOpenMPDirectiveKind(name);
}
```

**Benefits**:
- Compile-time type safety
- Compiler catches typos
- Fast switch dispatch
- Works with LLVM utilities

**Migration**:
```cpp
// Old code
if (directiveName == "parallel") { ... }

// New code
if (directive == llvm::omp::Directive::OMPD_parallel) { ... }
```

---

### Pattern 3: Manual Outlining with createFunction

#### ❌ DEPRECATED: Manual Function Creation for Parallel Regions

```cpp
// WRONG: Manually creating outlined functions
llvm::Function *outlineParallel(...) {
  llvm::FunctionType *fnType = llvm::FunctionType::get(
      builder.getVoidTy(), {ptrType, ptrType}, false);
  
  llvm::Function *outlinedFn = llvm::Function::Create(
      fnType, 
      llvm::GlobalValue::InternalLinkage,
      ".omp_outlined",
      module);
  
  // Manually populate function...
  llvm::BasicBlock *entry = llvm::BasicBlock::Create(context, "entry", outlinedFn);
  // ... lots of manual IR construction
  
  return outlinedFn;
}
```

**Problems**:
- Error-prone manual IR construction
- Misses LLVM metadata attributes
- Inconsistent with other outlining
- Hard to maintain

#### ✅ MODERN: Use OpenMPIRBuilder

```cpp
// Correct: Use OpenMPIRBuilder for outlining
llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

auto outlineCallback = [&](
    llvm::OpenMPIRBuilder::InsertPointTy allocaIP,
    llvm::OpenMPIRBuilder::InsertPointTy codeGenIP) -> llvm::Error {
  
  // Translate region body
  builder.restoreIP(codeGenIP);
  return moduleTranslation.convertBlock(region.front(), true, builder);
};

llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
builder.restoreIP(ompBuilder->createParallel(
    ompLoc,
    outlineCallback,
    /*IfCondition=*/nullptr,
    /*NumThreads=*/nullptr,
    /*ProcBind=*/llvm::omp::ProcBindKind::OMP_PROC_BIND_default,
    /*IsCancellable=*/false));
```

**Benefits**:
- Correct metadata and attributes
- Consistent with LLVM conventions
- Handles edge cases automatically
- Future-proof for runtime changes

**Migration Guide**:
```cpp
// Replace manual outlining
llvm::Function *fn = outlineParallelManually(region);

// With OMPIRBuilder
ompBuilder->createParallel(loc, callback, ...);
```

---

### Pattern 4: Hardcoded Runtime Function Names

#### ❌ DEPRECATED: String Literals for Runtime Calls

```cpp
// WRONG: Hardcoded runtime function names
llvm::Function *barrierFn = module->getFunction("__kmpc_barrier");
if (!barrierFn) {
  barrierFn = llvm::Function::Create(
      llvm::FunctionType::get(voidTy, {ptrTy, i32Ty}, false),
      llvm::GlobalValue::ExternalLinkage,
      "__kmpc_barrier",
      module);
}
builder.CreateCall(barrierFn, {loc, tid});
```

**Problems**:
- Typos cause runtime failures
- Function signatures may change
- No type checking
- Misses ABI changes

#### ✅ MODERN: Use OMPIRBuilder Runtime Functions

```cpp
// Correct: Use OMPIRBuilder's runtime function wrappers
llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
ompBuilder->createBarrier(
    ompLoc,
    llvm::omp::Directive::OMPD_barrier,
    /*ForceSimpleCall=*/false,
    /*CheckCancelFlag=*/false);
```

**Benefits**:
- Correct function signatures guaranteed
- Automatic location info creation
- Compatible with runtime changes
- Type-safe

**Available OMPIRBuilder Methods**:
```cpp
// Instead of manual runtime calls, use:
ompBuilder->createBarrier(...)         // __kmpc_barrier
ompBuilder->createCancel(...)          // __kmpc_cancel
ompBuilder->createFlush(...)           // __kmpc_flush
ompBuilder->createMaster(...)          // __kmpc_master
ompBuilder->createCritical(...)        // __kmpc_critical
ompBuilder->createAtomicRead(...)      // atomic operations
ompBuilder->createAtomicWrite(...)
ompBuilder->createAtomicUpdate(...)
```

---

### Pattern 5: Custom Symbol Mapping Instead of SymMap

#### ❌ DEPRECATED: Manual Symbol Tracking

```cpp
// WRONG: Custom symbol-to-value mapping
class MySymbolMapper {
  std::map<const semantics::Symbol*, mlir::Value> symbolMap;
  
public:
  void addMapping(const semantics::Symbol *sym, mlir::Value val) {
    symbolMap[sym] = val;
  }
  
  mlir::Value lookup(const semantics::Symbol *sym) {
    return symbolMap[sym];  // No scope handling!
  }
};
```

**Problems**:
- No scope management
- Doesn't handle shadowing
- Missing symbol not detected
- No integration with Flang lowering

#### ✅ MODERN: Use lower::SymMap

```cpp
// Correct: Use Flang's SymMap for symbol tracking
lower::SymMap &symMap = converter.getSymMap();

// Add symbol mapping with scope
symMap.addSymbol(*symbol, mlirValue);

// Lookup symbol (respects scopes)
if (auto *symBox = symMap.lookupSymbol(*symbol)) {
  mlir::Value value = symBox->getAddr();
  // Use value
}

// Create new scope (for nested constructs)
lower::SymMap::ScopeTy scope(symMap);
// Symbols added here are scoped
symMap.addSymbol(*privateVar, privateValue);
// Scope exits here - symbols removed
```

**Benefits**:
- Automatic scope management
- Handles variable shadowing
- Integrated with Flang semantics
- Type-safe symbol boxes

**Migration**:
```cpp
// Old: Custom map
std::map<Symbol*, mlir::Value> myMap;
myMap[symbol] = value;
mlir::Value v = myMap[symbol];

// New: SymMap
symMap.addSymbol(*symbol, value);
mlir::Value v = symMap.lookupSymbol(*symbol)->getAddr();
```

---

### Pattern 6: Parsing OpenMP Pragmas in Fortran Parser

#### ❌ DEPRECATED: Custom OpenMP Parsing Logic

```cpp
// WRONG: Custom OpenMP parsing outside grammar
class CustomOpenMPParser {
  bool parseParallelDirective() {
    if (current_token == "parallel") {
      // Custom parsing logic
      parseClauseList();
      // ...
    }
  }
  
  void parseClauseList() {
    // Fragile, incomplete
  }
};
```

**Problems**:
- Duplicates parser grammar
- Incomplete clause support
- Out of sync with spec
- No error recovery

#### ✅ MODERN: Use Flang's OpenMP Parser Infrastructure

**File**: `flang/lib/Parser/openmp-parsers.cpp`

```cpp
// Correct: Extend existing parser infrastructure
TYPE_PARSER(
  construct<OpenMPConstruct>(
    Parser<OpenMPBlockConstruct>{} ||
    Parser<OpenMPLoopConstruct>{} ||
    Parser<OpenMPStandaloneConstruct>{}))

// Add new constructs using parser combinators
TYPE_PARSER(
  sourced(construct<MyNewConstruct>(
    "MY DIRECTIVE"_tok >> 
    Parser<OmpClauseList>{} /
    Parser<Block>{} /
    "END MY DIRECTIVE"_tok)))
```

**Benefits**:
- Unified grammar specification
- Automatic error recovery
- Consistent with Fortran parsing
- Generates proper parse tree

**Parser Combinators**:
```cpp
// Use Flang's parser combinators
"PARALLEL"_tok              // Token literal
pure(value)                 // Always succeeds with value
maybe(parser)               // Optional
some(parser)                // One or more
many(parser)                // Zero or more
parser1 / parser2           // Sequence (discard second)
parser1 >> parser2          // Sequence (keep second)
parser1 || parser2          // Alternative
construct<T>(parsers...)    // Build parse tree node
```

---

### Pattern 7: Bypassing ClauseProcessor

#### ❌ DEPRECATED: Direct Clause Handling in genFoo Functions

```cpp
// WRONG: Manually processing clauses in lowering
void genParallelOp(..., const parser::OmpClauseList &clauses) {
  mlir::Value numThreads;
  mlir::Value ifExpr;
  
  // Manual clause iteration (error-prone)
  for (const auto &clause : clauses.v) {
    if (const auto *ntClause = std::get_if<parser::OmpClause::NumThreads>(&clause.u)) {
      // Manually process NumThreads
      numThreads = genExpr(ntClause->v);
    } else if (const auto *ifClause = std::get_if<parser::OmpClause::If>(&clause.u)) {
      // Manually process If
      ifExpr = genExpr(ifClause->v);
    }
    // Missing many clause types!
  }
  
  auto parallelOp = builder.create<omp::ParallelOp>(loc, ifExpr, numThreads, ...);
}
```

**Problems**:
- Code duplication across directives
- Easy to miss clause types
- No unified error handling
- Hard to maintain

#### ✅ MODERN: Use ClauseProcessor

**File**: `flang/lib/Lower/OpenMP/ClauseProcessor.h`

```cpp
// Correct: Use ClauseProcessor for consistent clause handling
void genParallelOp(..., const parser::OmpClauseList &clauses) {
  ClauseProcessor cp(converter, clauses);
  
  mlir::Value numThreads, ifExpr;
  llvm::SmallVector<mlir::Value> privateVars, reductionVars;
  llvm::SmallVector<mlir::Attribute> reductionSymbols;
  
  // Process all relevant clauses through ClauseProcessor
  cp.processIf(llvm::omp::Directive::OMPD_parallel, ifExpr);
  cp.processNumThreads(numThreads);
  cp.processPrivate(privateVars);
  cp.processReduction(loc, reductionVars, reductionSymbols);
  cp.processDefault();
  cp.processCopyin();
  
  // ClauseProcessor handles all the details
  auto parallelOp = builder.create<omp::ParallelOp>(
      loc, ifExpr, numThreads, privateVars, /*firstprivate=*/{},
      /*shared=*/{}, reductionVars, reductionSymbols);
}
```

**Benefits**:
- Centralized clause processing logic
- Consistent error messages
- All clauses handled
- Easy to extend

**ClauseProcessor Methods**:
```cpp
void processIf(Directive directive, mlir::Value &result);
void processNumThreads(mlir::Value &result);
void processPrivate(llvm::SmallVectorImpl<mlir::Value> &result);
void processFirstprivate(llvm::SmallVectorImpl<mlir::Value> &result);
void processShared(llvm::SmallVectorImpl<mlir::Value> &result);
void processReduction(Location loc, 
                      llvm::SmallVectorImpl<mlir::Value> &vars,
                      llvm::SmallVectorImpl<mlir::Attribute> &symbols);
void processSchedule(mlir::omp::ScheduleModifier &modifier,
                     mlir::omp::ScheduleKind &kind,
                     mlir::Value &chunkSize);
void processCollapse(int64_t &result);
void processOrdered(int64_t &result);
void processNowait(mlir::UnitAttr &result);
void processDefault();
void processCopyin();
void processCopyprivate();
```

---

### Pattern 8: Creating MLIR Operations Without Builders

#### ❌ DEPRECATED: Direct OperationState Construction

```cpp
// WRONG: Bypassing operation builders
mlir::OperationState state(loc, "omp.parallel");
state.addOperands({ifExpr, numThreads});
state.addRegion();
mlir::Operation *op = builder.create(state);

// Manual region setup
mlir::Region &region = op->getRegion(0);
mlir::Block *block = new mlir::Block();
region.push_back(block);
// Error-prone!
```

**Problems**:
- Bypasses operation verifiers
- Easy to create invalid IR
- No type checking
- Missing attributes

#### ✅ MODERN: Use Generated Operation Builders

```cpp
// Correct: Use TableGen-generated builders
auto parallelOp = builder.create<mlir::omp::ParallelOp>(
    loc,
    /*if_expr=*/ifExpr,
    /*num_threads=*/numThreads,
    /*private_vars=*/privateVars,
    /*firstprivate_vars=*/llvm::ValueRange{},
    /*shared_vars=*/llvm::ValueRange{},
    /*reduction_vars=*/reductionVars,
    /*reduction_symbols=*/reductionSymbols);

// Builder automatically creates region
mlir::Block *block = builder.createBlock(&parallelOp.getRegion());
builder.setInsertionPointToStart(block);

// Translate body
// ...

// Verify operation
assert(mlir::succeeded(parallelOp.verify()));
```

**Benefits**:
- Type-safe operands
- Automatic verification
- Correct attributes
- IDE autocomplete

---

### Pattern 9: Ignoring Location Information

#### ❌ DEPRECATED: Using UnknownLoc Everywhere

```cpp
// WRONG: Losing source location information
mlir::Location loc = builder.getUnknownLoc();

auto parallelOp = builder.create<omp::ParallelOp>(loc, ...);
// Error messages won't show source location!
```

**Problems**:
- Poor error messages
- Hard to debug
- Lost source context
- User confusion

#### ✅ MODERN: Propagate Source Locations

```cpp
// Correct: Convert parser location to MLIR location
mlir::Location loc = converter.genLocation(directive.source);

auto parallelOp = builder.create<omp::ParallelOp>(loc, ...);

// For nested operations, use operation's location
mlir::Location nestedLoc = parallelOp.getLoc();
auto wsloopOp = builder.create<omp::WsloopOp>(nestedLoc, ...);

// For clauses, use clause source location
mlir::Location clauseLoc = converter.genLocation(clause.source);
if (error) {
  emitError(clauseLoc) << "Invalid clause usage";
}
```

**Benefits**:
- Accurate error locations
- Better diagnostics
- Easier debugging
- User-friendly messages

**Location Types**:
```cpp
// Use appropriate location type
FileLineColLoc    // For source locations
FusedLoc          // For combining locations
NameLoc           // For named entities
CallSiteLoc       // For inlined code
UnknownLoc        // Only as last resort
```

---

### Pattern 10: Monolithic Translation Functions

#### ❌ DEPRECATED: Giant switch Statement for All Operations

```cpp
// WRONG: One huge function handling all OpenMP operations
llvm::Error translateOpenMP(mlir::Operation *op, ...) {
  if (auto parallelOp = dyn_cast<omp::ParallelOp>(op)) {
    // 200 lines of parallel translation
  } else if (auto wsloopOp = dyn_cast<omp::WsloopOp>(op)) {
    // 300 lines of wsloop translation
  } else if (auto taskOp = dyn_cast<omp::TaskOp>(op)) {
    // 250 lines of task translation
  }
  // ... 50 more operations in one function (3000+ lines)
  
  return llvm::Error::success();
}
```

**Problems**:
- Unmaintainable code
- Hard to test individual operations
- Merge conflicts
- Long compile times

#### ✅ MODERN: Modular Operation Translation

```cpp
// Correct: Separate translation function per operation
static llvm::Error convertOmpParallel(
    mlir::omp::ParallelOp parallelOp,
    llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) {
  // 50-100 lines focused on parallel
  return llvm::Error::success();
}

static llvm::Error convertOmpWsloop(
    mlir::omp::WsloopOp wsloopOp,
    llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) {
  // 50-100 lines focused on wsloop
  return llvm::Error::success();
}

// Register handlers
void registerOpenMPTranslation(DialectRegistry &registry) {
  registry.insert<omp::OpenMPDialect>();
  registry.addExtension(+[](MLIRContext *ctx, omp::OpenMPDialect *dialect) {
    dialect->addOperationTranslation<omp::ParallelOp>(convertOmpParallel);
    dialect->addOperationTranslation<omp::WsloopOp>(convertOmpWsloop);
    dialect->addOperationTranslation<omp::TaskOp>(convertOmpTask);
    // ...
  });
}
```

**Benefits**:
- Modular, maintainable code
- Easy to test
- Parallel development
- Clear ownership

---

### Pattern 11: Ignoring OpenMP Versions

#### ❌ DEPRECATED: No Version Checking

```cpp
// WRONG: Assuming latest OpenMP version
void processAffinityClause(...) {
  // AFFINITY clause added in OpenMP 5.0
  // But no check if user's code is OpenMP 4.5!
  genAffinityClause(...);
}
```

**Problems**:
- Accepts invalid code
- Confusing errors
- Spec non-compliance
- User frustration

#### ✅ MODERN: Check OpenMP Version

```cpp
// Correct: Validate features against OpenMP version
void OmpStructureChecker::Enter(const parser::OmpClause::Affinity &affinity) {
  CheckAllowed(llvm::omp::Clause::OMPC_affinity);
  
  // Check OpenMP version
  unsigned ompVersion = context_.languageFeatures().GetOpenMPVersion();
  if (ompVersion < 50) {
    context_.Say(GetContext().clauseSource,
        "AFFINITY clause requires OpenMP 5.0 or later; "
        "currently using OpenMP %d.%d"_err_en_US,
        ompVersion / 10, ompVersion % 10);
    return;
  }
  
  // Proceed with validation
  validateAffinityClause(affinity);
}

// Use LLVM's version utilities
bool isAllowedInVersion(llvm::omp::Directive directive, unsigned version) {
  return llvm::omp::getOpenMPVersionIntroduced(directive) <= version;
}
```

**Benefits**:
- Spec compliance
- Clear error messages
- Future-proof
- Version-aware compilation

**Version Constants**:
```cpp
// OpenMP version codes
#define OMP_VERSION_20  20
#define OMP_VERSION_25  25
#define OMP_VERSION_30  30
#define OMP_VERSION_31  31
#define OMP_VERSION_40  40
#define OMP_VERSION_45  45
#define OMP_VERSION_50  50
#define OMP_VERSION_51  51
#define OMP_VERSION_52  52

// Check feature availability
if (ompVersion >= OMP_VERSION_50) {
  // Use OpenMP 5.0+ features
}
```

---

### Pattern 12: Not Using Test Utilities

#### ❌ DEPRECATED: Manual Test Case Writing

```cpp
// WRONG: Writing tests without test utilities
// test.f90 - no structure
program test
  !$omp parallel
  !$omp end parallel
end program

// No error checking, no FileCheck directives, no automation
```

**Problems**:
- Hard to verify correctness
- Manual output inspection
- No regression detection
- Time-consuming

#### ✅ MODERN: Use LIT and FileCheck

**File**: `flang/test/Lower/OpenMP/parallel-test.f90`

```fortran
! RUN: %flang_fc1 -emit-mlir -fopenmp %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-llvm -fopenmp %s -o - | FileCheck %s --check-prefix=LLVMIR

! CHECK-LABEL: func @_QPtest_parallel
subroutine test_parallel()
  ! CHECK: omp.parallel {
  ! CHECK:   omp.terminator
  ! CHECK: }
  !$omp parallel
  !$omp end parallel
end subroutine

! Test with clauses
! CHECK-LABEL: func @_QPtest_parallel_clauses
subroutine test_parallel_clauses()
  integer :: x
  
  ! CHECK: omp.parallel
  ! CHECK-SAME: num_threads(%{{.*}} : i32)
  ! CHECK-SAME: private(%{{.*}} : !fir.ref<i32>)
  !$omp parallel num_threads(4) private(x)
    x = 42
  !$omp end parallel
  
  ! LLVMIR-LABEL: define {{.*}} @_QPtest_parallel_clauses
  ! LLVMIR: call void @__kmpc_fork_call
end subroutine
```

**Benefits**:
- Automated testing
- Regression detection
- Clear pass/fail
- CI/CD integration

**Test Structure**:
```bash
# RUN line structure
! RUN: <compiler> <flags> %s -o - | FileCheck [options] %s

# Common FileCheck directives
! CHECK:           Match line (in order)
! CHECK-NEXT:      Match next line
! CHECK-DAG:       Match line (any order in block)
! CHECK-NOT:       Must not appear
! CHECK-LABEL:     Reset match position
! CHECK-SAME:      Continue previous CHECK line
```

---

### Migration Checklist

When modernizing OpenMP implementation code:

**Phase 1: Assessment**
- [ ] Identify deprecated patterns using grep/search
- [ ] Review against this guide
- [ ] Estimate migration effort
- [ ] Plan incremental migration

**Phase 2: Code Updates**
- [ ] Replace string matching with enum types
- [ ] Switch to OMPIRBuilder for runtime calls
- [ ] Adopt ClauseProcessor for clause handling
- [ ] Use SymMap for symbol tracking
- [ ] Add proper location information
- [ ] Modularize large translation functions

**Phase 3: Testing**
- [ ] Convert manual tests to LIT + FileCheck
- [ ] Add version checking tests
- [ ] Test with multiple OpenMP versions
- [ ] Verify error messages show locations

**Phase 4: Documentation**
- [ ] Update comments with modern patterns
- [ ] Document any remaining technical debt
- [ ] Add examples of correct usage
- [ ] Update contribution guidelines

---

### Quick Reference: Pattern Replacements

| Deprecated | Modern | File/Function |
|------------|--------|---------------|
| Edit `.gen` files | Edit `.def` files | `OMPKinds.def` |
| String directive matching | Enum-based switch | `llvm::omp::Directive` |
| Manual function outlining | `OMPIRBuilder::createParallel` | `OpenMPIRBuilder` |
| Hardcoded `"__kmpc_*"` | `ompBuilder->createBarrier` | `OpenMPIRBuilder` |
| Custom symbol map | `lower::SymMap` | Flang lowering |
| Custom OpenMP parser | Extend `openmp-parsers.cpp` | Parser combinators |
| Manual clause iteration | `ClauseProcessor` | `ClauseProcessor.h` |
| `OperationState` | Generated builders | TableGen ops |
| `UnknownLoc` | `converter.genLocation` | Proper source locations |
| Monolithic translation | Modular per-op functions | Operation translation |
| No version check | Check `GetOpenMPVersion()` | Semantics |
| Manual test writing | LIT + FileCheck | Test infrastructure |

---

# PART 4: TROUBLESHOOTING AND DEBUGGING

## Common MLIR Lowering Errors and Solutions

This section provides diagnosis and fixes for the most common errors encountered when lowering OpenMP constructs from Flang's parse tree to MLIR operations.

---

### Error Category 1: Operation Verification Failures

#### Error: "operand #N does not dominate this use"

**Symptom**:
```
error: 'omp.parallel' op operand #0 does not dominate this use
note: see current operation: %0 = "omp.parallel"(%1) ...
note: operand defined here: %1 = fir.load %2
```

**Cause**: SSA value is used before it's defined, or defined inside a nested region but used in parent region.

**Common Scenarios**:
```cpp
// WRONG: Creating value inside region, using in operation arguments
auto parallelOp = builder.create<omp::ParallelOp>(loc, ...);
builder.createBlock(&parallelOp.getRegion());
auto value = builder.create<fir::LoadOp>(loc, addr);  // Defined inside
// BUG: Can't use 'value' as operand to parallelOp

// CORRECT: Create operands before the operation
auto value = builder.create<fir::LoadOp>(loc, addr);  // Define first
auto parallelOp = builder.create<omp::ParallelOp>(
    loc, value, ...);  // Then use as operand
```

**Fix Pattern**:
1. Identify the value that doesn't dominate
2. Move its definition before the operation that uses it
3. Ensure SSA dominance: definitions before uses

**Code Fix**:
```cpp
// In Clauses.cpp or OpenMP.cpp
void ClauseProcessor::processNumThreads(mlir::Value &result) {
  const parser::CharBlock *source = nullptr;
  if (auto *clause = findClause<parser::OmpClause::NumThreads>(source)) {
    mlir::Location loc = converter.getCurrentLocation();
    auto &firOpBuilder = converter.getFirOpBuilder();
    
    // CRITICAL: Evaluate expression BEFORE creating the operation
    mlir::Value numThreadsExpr = 
        fir::getBase(converter.genExprValue(loc, clause->v, stmtCtx));
    
    // Convert to i32 if needed
    result = firOpBuilder.createConvert(loc, firOpBuilder.getI32Type(), 
                                        numThreadsExpr);
  }
}
```

---

#### Error: "expected operation to have N operands"

**Symptom**:
```
error: 'omp.wsloop' op expected operation to have 3 operands
note: see current operation: "omp.wsloop"(%0, %1) ...
```

**Cause**: Mismatch between TableGen operation definition and actual operands provided during lowering.

**Common Scenario**: Missing `AttrSizedOperandSegments` handling

```tablegen
// In OpenMPOps.td
def WsLoopOp : OpenMP_Op<"wsloop", [AttrSizedOperandSegments]> {
  let arguments = (ins
    AnyType:$lower_bound,
    AnyType:$upper_bound,
    AnyType:$step,
    Optional<I32>:$schedule_chunk,  // Optional!
    Variadic<AnyType>:$reduction_vars
  );
}
```

**Fix**: Always pass operand counts for operations with `AttrSizedOperandSegments`

```cpp
// WRONG: Not setting operand segment sizes
auto wsloopOp = builder.create<omp::WsLoopOp>(
    loc, lowerBound, upperBound, step, 
    /*schedule_chunk=*/nullptr,  // Optional not provided correctly
    reductionVars);

// CORRECT: Let operation builder handle optional operands
auto wsloopOp = builder.create<omp::WsLoopOp>(
    loc, lowerBound, upperBound, step, 
    scheduleChunk,  // Pass nullptr or actual value
    reductionVars);
// Operation builder automatically generates AttrSizedOperandSegments
```

**Debugging**:
```bash
# Check operation definition
grep -A 20 "def WsLoopOp" mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td

# Verify generated code
ninja MLIROpenMPOpsIncGen
cat build/tools/mlir/include/mlir/Dialect/OpenMP/OpenMPOps.h.inc | grep -A 30 "class WsLoopOp"
```

---

#### Error: "expects different type than prior uses"

**Symptom**:
```
error: use of value '%0' expects different type than prior uses: '!fir.ref<i64>' vs '!fir.ref<i32>'
note: prior use here: %1 = fir.load %0 : !fir.ref<i32>
note: prior use here: omp.wsloop (%0 : !fir.ref<i64>) ...
```

**Cause**: Type mismatch between SSA value definition and usage.

**Common in OpenMP**: Loop bounds type inconsistency

```cpp
// WRONG: Type mismatch between Fortran kind and MLIR type
mlir::Value lowerBound = genLowerBound();  // Returns i32
mlir::Value upperBound = genUpperBound();  // Returns i64 (BUG!)
mlir::Value step = genStep();              // Returns i32

auto wsloopOp = builder.create<omp::WsLoopOp>(
    loc, lowerBound, upperBound, step, ...);  // Type error!

// CORRECT: Ensure all loop bounds have same type
mlir::Type loopVarType = builder.getI64Type();

mlir::Value lowerBound = builder.createConvert(
    loc, loopVarType, genLowerBound());
mlir::Value upperBound = builder.createConvert(
    loc, loopVarType, genUpperBound());
mlir::Value step = builder.createConvert(
    loc, loopVarType, genStep());

auto wsloopOp = builder.create<omp::WsLoopOp>(
    loc, lowerBound, upperBound, step, ...);
```

**Fix Checklist**:
- [ ] Check TableGen definition for expected types
- [ ] Insert `fir.convert` operations for type mismatches
- [ ] Ensure integer kind matches between Fortran variables and MLIR values
- [ ] Verify pointer types (`!fir.ref<T>`) vs value types

---

#### Error: "region #N expects N block arguments"

**Symptom**:
```
error: 'omp.wsloop' op region #0 expects 1 block argument
note: see current operation: "omp.wsloop"(...) ({
  ^bb0:  // No arguments (BUG!)
  ...
})
```

**Cause**: Missing block arguments in operation region (e.g., loop induction variable).

**Fix**:
```cpp
// WRONG: Empty block
auto wsloopOp = builder.create<omp::WsLoopOp>(loc, ...);
auto *loopRegion = &wsloopOp.getRegion();
builder.createBlock(loopRegion);  // No arguments!
// ... generate loop body ...

// CORRECT: Add induction variable as block argument
auto wsloopOp = builder.create<omp::WsLoopOp>(loc, ...);
auto *loopRegion = &wsloopOp.getRegion();

mlir::Type loopVarType = builder.getI64Type();
auto *loopBlock = builder.createBlock(
    loopRegion, {}, {loopVarType}, {loc});  // Add induction variable

mlir::Value inductionVar = loopBlock->getArgument(0);
// Use inductionVar in loop body
```

**Pattern for Loop Operations**:
```cpp
// Standard pattern for loop-based constructs
static void genLoopRegion(
    mlir::Operation *loopOp,
    mlir::Region &region,
    mlir::Type loopVarType,
    mlir::Location loc) {
  
  auto &builder = ...;
  auto *block = builder.createBlock(&region, {}, {loopVarType}, {loc});
  
  mlir::Value iv = block->getArgument(0);
  // Generate loop body using 'iv'
  
  // Terminator
  builder.create<omp::YieldOp>(loc);
}
```

---

### Error Category 2: Clause Processing Errors

#### Error: "unhandled clause in genXXX"

**Symptom**:
```
flang-new: error: unhandled clause PRIVATE in genParallelOp
```

**Cause**: Clause appears in parse tree but not processed in lowering code.

**Fix**: Add clause processing in `ClauseProcessor`

```cpp
// In flang/lib/Lower/OpenMP/OpenMP.cpp

static mlir::omp::ParallelOp genParallelOp(...) {
  ClauseProcessor cp(converter, clauseList);
  
  // Process all relevant clauses
  mlir::Value ifOperand, numThreadsOperand;
  cp.processIf(ifOperand, llvm::omp::Directive::OMPD_parallel);
  cp.processNumThreads(numThreadsOperand);
  
  // ADD MISSING CLAUSE PROCESSING:
  llvm::SmallVector<mlir::Value> privateVars;
  cp.processPrivate(privateVars);  // ← Add this!
  
  llvm::SmallVector<mlir::Value> reductionVars;
  llvm::SmallVector<mlir::Attribute> reductionSyms;
  cp.processReduction(reductionVars, reductionSyms);
  
  // Create operation with all processed clauses
  auto parallelOp = builder.create<mlir::omp::ParallelOp>(
      loc, ifOperand, numThreadsOperand, 
      privateVars,      // ← Pass processed values
      reductionVars, reductionSyms, ...);
  
  return parallelOp;
}
```

**Debugging**:
```bash
# Find which clauses are allowed on directive
grep -A 50 "OMPD_parallel" llvm/include/llvm/Frontend/OpenMP/OMPKinds.def

# Check if processing function exists
grep "processPrivate" flang/lib/Lower/OpenMP/Clauses.cpp
```

---

#### Error: "unable to find symbol for XXX"

**Symptom**:
```
error: unable to find symbol for 'x' in OpenMP clause
note: in PRIVATE(x) clause
```

**Cause**: Variable in clause list not resolved during semantic analysis or not found in symbol table during lowering.

**Common Scenario**: Implicit variable or wrong scope

**Fix**: Ensure proper symbol resolution

```cpp
// In Clauses.cpp
void ClauseProcessor::processPrivate(
    llvm::SmallVector<mlir::Value> &privateVars) {
  
  const parser::CharBlock *source = nullptr;
  if (auto *clause = findClause<parser::OmpClause::Private>(source)) {
    
    for (const auto &object : clause->v.v) {
      if (const auto *name = std::get_if<parser::Name>(&object.u)) {
        
        // CHECK: Symbol must be resolved
        if (!name->symbol) {
          llvm::errs() << "ERROR: Unresolved symbol: " 
                       << name->ToString() << "\n";
          continue;  // Or report error
        }
        
        // Get symbol address from converter
        const auto *symbol = name->symbol;
        mlir::Value addr = converter.getSymbolAddress(*symbol);
        
        if (!addr) {
          llvm::errs() << "ERROR: No address for symbol: " 
                       << symbol->name().ToString() << "\n";
          continue;
        }
        
        privateVars.push_back(addr);
      }
    }
  }
}
```

**Check Symbol Table**:
```cpp
// Debugging: Print available symbols
converter.getCurrentScope().DumpSymbols(llvm::errs());
```

---

#### Error: "fir.alloca must have a constant shape"

**Symptom**:
```
error: 'fir.alloca' op result type must have constant shape
note: see current operation: %0 = fir.alloca !fir.array<?xi32>
```

**Cause**: Creating allocation for array with dynamic shape without proper shape operand.

**Fix**: Use shape operand for dynamic arrays

```cpp
// WRONG: Dynamic array without shape
auto arrType = fir::SequenceType::get({fir::SequenceType::getUnknownExtent()}, 
                                      builder.getI32Type());
auto addr = builder.create<fir::AllocaOp>(loc, arrType);  // ERROR!

// CORRECT: Provide shape for dynamic allocation
auto arrType = fir::SequenceType::get({fir::SequenceType::getUnknownExtent()}, 
                                      builder.getI32Type());
mlir::Value arraySize = builder.create<fir::LoadOp>(loc, sizeVar);
auto addr = builder.create<fir::AllocaOp>(
    loc, arrType, /*typeparams=*/{}, /*shape=*/{arraySize});
```

---

### Error Category 3: Missing Terminators

#### Error: "block operations must be terminated"

**Symptom**:
```
error: 'omp.parallel' op expects regions to end with 'omp.terminator'
note: see current operation: "omp.parallel"(...) ({
  ^bb0:
    // ... operations ...
    // Missing terminator!
})
```

**Cause**: Forgetting to add terminator operation at end of region.

**Fix**: Always terminate regions

```cpp
// Pattern for parallel regions
auto parallelOp = builder.create<omp::ParallelOp>(loc, ...);
auto &region = parallelOp.getRegion();
builder.createBlock(&region);

// Lower body statements
lowerStatements(...);

// CRITICAL: Add terminator
builder.create<omp::TerminatorOp>(loc);
```

**Region Terminator Reference**:
| **Operation** | **Terminator** |
|---------------|----------------|
| `omp.parallel` | `omp.terminator` |
| `omp.wsloop` | `omp.yield` |
| `omp.simd` | `omp.yield` |
| `omp.task` | `omp.terminator` |
| `omp.sections` | `omp.terminator` |
| `omp.section` | `omp.terminator` |
| `omp.single` | `omp.terminator` |
| `omp.master` | `omp.terminator` |
| `omp.critical` | `omp.terminator` |

**Auto-Terminator Pattern**:
```cpp
// Helper to ensure terminator is added
class RegionTerminatorGuard {
  mlir::OpBuilder &builder;
  mlir::Location loc;
  
public:
  RegionTerminatorGuard(mlir::OpBuilder &b, mlir::Location l) 
      : builder(b), loc(l) {}
  
  ~RegionTerminatorGuard() {
    // Automatically add terminator when leaving scope
    if (!builder.getBlock()->mightHaveTerminator()) {
      builder.create<omp::TerminatorOp>(loc);
    }
  }
};

// Usage:
{
  RegionTerminatorGuard guard(builder, loc);
  // ... generate region body ...
  // Terminator added automatically
}
```

---

### Error Category 4: Attribute Errors

#### Error: "requires attribute 'XXX'"

**Symptom**:
```
error: 'omp.wsloop' op requires attribute 'schedule_kind'
```

**Cause**: Missing required attribute in operation creation.

**Fix**: Add all required attributes

```cpp
// Check TableGen definition for required attributes
// In OpenMPOps.td:
// OptionalAttr<...> vs Attr<...> (required)

// WRONG: Missing schedule_kind
auto wsloopOp = builder.create<omp::WsLoopOp>(
    loc, lowerBound, upperBound, step, ...);

// CORRECT: Provide required attribute
auto scheduleKind = omp::ClauseScheduleKindAttr::get(
    builder.getContext(), omp::ClauseScheduleKind::Static);

auto wsloopOp = builder.create<omp::WsLoopOp>(
    loc, lowerBound, upperBound, step,
    /*schedule_chunk=*/nullptr,
    /*schedule_kind=*/scheduleKind,  // ← Required!
    ...);
```

---

#### Error: "attribute value does not match expected type"

**Symptom**:
```
error: 'reduction_syms' attribute should be an array of symbols
note: got: []
```

**Cause**: Wrong attribute type (e.g., array vs single value, wrong element type).

**Fix**: Match attribute type exactly

```cpp
// WRONG: Empty array when symbols expected
auto parallelOp = builder.create<omp::ParallelOp>(
    loc, ...,
    reductionVars,
    /*reduction_syms=*/builder.getArrayAttr({}));  // Wrong!

// CORRECT: Create proper symbol references
llvm::SmallVector<mlir::Attribute> reductionSyms;
for (auto &reduction : reductionList) {
  auto symRef = mlir::SymbolRefAttr::get(
      builder.getContext(), reduction.reductionFn);
  reductionSyms.push_back(symRef);
}

auto parallelOp = builder.create<omp::ParallelOp>(
    loc, ...,
    reductionVars,
    builder.getArrayAttr(reductionSyms));  // Correct type
```

---

### Error Category 5: Data-Sharing Errors

#### Error: "SSA value defined inside parallel region used outside"

**Symptom**:
```
error: using value %0 defined inside parallel region
note: %0 = fir.load %1 : !fir.ref<i32>  (inside omp.parallel)
note: %2 = arith.addi %0, %3  (outside omp.parallel)
```

**Cause**: Violating SSA form - values defined in nested regions can't be used in parent.

**Fix**: Use allocations and loads/stores instead

```cpp
// WRONG: Defining value inside, using outside
auto parallelOp = builder.create<omp::ParallelOp>(loc, ...);
builder.createBlock(&parallelOp.getRegion());

mlir::Value result = builder.create<fir::LoadOp>(loc, addr);  // Inside
builder.create<omp::TerminatorOp>(loc);

// Try to use 'result' here - ERROR! Outside region

// CORRECT: Use memory location to communicate
mlir::Value resultAddr = builder.create<fir::AllocaOp>(
    loc, builder.getI32Type());  // Outside parallel

auto parallelOp = builder.create<omp::ParallelOp>(loc, ...);
builder.createBlock(&parallelOp.getRegion());

mlir::Value computedValue = /* ... compute ... */;
builder.create<fir::StoreOp>(loc, computedValue, resultAddr);  // Store
builder.create<omp::TerminatorOp>(loc);

// Load after parallel region
mlir::Value result = builder.create<fir::LoadOp>(loc, resultAddr);
```

---

#### Error: "private variable not properly cloned"

**Symptom**:
```
error: private variable 'x' used before initialization
```

**Cause**: PRIVATE clause requires creating local copy, but address mapping not updated.

**Fix**: Update symbol address mapping for private variables

```cpp
// In DataSharingProcessor or similar
void processPrivate(const Symbol &symbol) {
  mlir::Location loc = converter.getCurrentLocation();
  auto &builder = converter.getFirOpBuilder();
  
  // Get original variable address
  mlir::Value originalAddr = converter.getSymbolAddress(symbol);
  
  // Create private copy (allocation)
  mlir::Type varType = originalAddr.getType();
  mlir::Value privateAddr = builder.create<fir::AllocaOp>(
      loc, varType.cast<fir::ReferenceType>().getEleTy());
  
  // CRITICAL: Update symbol address for this scope
  converter.bindSymbol(symbol, privateAddr);
  
  // Now uses of 'symbol' will reference privateAddr, not originalAddr
}
```

---

### Error Category 6: Debugging Tools and Techniques

#### Technique 1: Enable MLIR Verification

```bash
# Verify MLIR after lowering
flang-new -fc1 -fopenmp -emit-mlir -verify-each test.f90

# More verbose verification
flang-new -fc1 -fopenmp -emit-mlir -mlir-print-ir-after-all test.f90 2>&1 | less
```

#### Technique 2: Dump MLIR Operations

```cpp
// In lowering code (Clauses.cpp, OpenMP.cpp)
void debugOperation(mlir::Operation *op) {
  llvm::errs() << "=== Operation Dump ===\n";
  op->dump();
  llvm::errs() << "=== Operands ===\n";
  for (auto operand : op->getOperands()) {
    llvm::errs() << "  ";
    operand.dump();
  }
  llvm::errs() << "=== Attributes ===\n";
  for (auto attr : op->getAttrs()) {
    llvm::errs() << "  " << attr.getName() << " = ";
    attr.getValue().dump();
  }
}

// Usage:
auto parallelOp = builder.create<omp::ParallelOp>(...);
debugOperation(parallelOp);
```

#### Technique 3: Check Operation Verifier

```cpp
// Manually verify operation
auto parallelOp = builder.create<omp::ParallelOp>(...);

if (failed(parallelOp.verify())) {
  llvm::errs() << "Operation verification failed!\n";
  parallelOp.dump();
}
```

#### Technique 4: Print Symbol Table

```cpp
// In lowering function
void debugSymbolTable(lower::AbstractConverter &converter) {
  llvm::errs() << "=== Symbol Table ===\n";
  auto &localSymbols = converter.getLocalSymbols();
  for (auto &[symbol, addr] : localSymbols) {
    llvm::errs() << "  " << symbol->name().ToString() << " -> ";
    addr.dump();
  }
}
```

#### Technique 5: Bisect MLIR Passes

```bash
# Find which MLIR pass is failing
flang-new -fc1 -fopenmp -emit-mlir test.f90 \
  -mlir-pass-pipeline-crash-reproducer=reproducer.mlir \
  -mlir-pass-pipeline-local-reproducer
```

#### Technique 6: MLIR Textual Format Debugging

```bash
# Generate MLIR text
flang-new -fc1 -fopenmp -emit-mlir test.f90 -o test.mlir

# Manually inspect MLIR
cat test.mlir

# Re-parse MLIR to check syntax
mlir-opt test.mlir --verify-diagnostics
```

---

### Common Fix Patterns Reference

#### Pattern 1: Safe Operation Creation

```cpp
// Always use this pattern to avoid verification failures
template<typename OpTy, typename... Args>
OpTy safeCreateOp(mlir::OpBuilder &builder, mlir::Location loc, Args&&... args) {
  auto op = builder.create<OpTy>(loc, std::forward<Args>(args)...);
  
  // Verify immediately during development
  #ifdef DEBUG
  if (failed(op.verify())) {
    llvm::errs() << "ERROR: Operation failed verification\n";
    op.dump();
    llvm_unreachable("Invalid operation created");
  }
  #endif
  
  return op;
}
```

#### Pattern 2: Type-Safe Value Conversion

```cpp
// Helper to convert between FIR types
mlir::Value convertToType(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    mlir::Value value,
    mlir::Type targetType) {
  
  if (value.getType() == targetType) {
    return value;  // Already correct type
  }
  
  // Use fir.convert for compatible types
  return builder.create<fir::ConvertOp>(loc, targetType, value);
}
```

#### Pattern 3: Region Builder Helper

```cpp
// Helper to safely create and populate regions
class RegionBuilder {
  mlir::OpBuilder &builder;
  mlir::Region &region;
  mlir::Location loc;
  mlir::Block *block;
  
public:
  RegionBuilder(mlir::OpBuilder &b, mlir::Region &r, mlir::Location l,
                llvm::ArrayRef<mlir::Type> argTypes = {})
      : builder(b), region(r), loc(l) {
    
    llvm::SmallVector<mlir::Location> argLocs(argTypes.size(), loc);
    block = builder.createBlock(&region, {}, argTypes, argLocs);
  }
  
  mlir::BlockArgument getArgument(unsigned index) {
    return block->getArgument(index);
  }
  
  template<typename TerminatorOp>
  void finalize() {
    builder.create<TerminatorOp>(loc);
  }
};

// Usage:
auto parallelOp = builder.create<omp::ParallelOp>(loc, ...);
RegionBuilder regionBuilder(builder, parallelOp.getRegion(), loc);

// Generate body
// ...

// Add terminator
regionBuilder.finalize<omp::TerminatorOp>();
```

#### Pattern 4: Clause Processing Template

```cpp
// Template for processing optional clauses safely
template<typename ClauseType, typename ProcessFn>
void processClauseIfPresent(
    const parser::OmpClauseList &clauses,
    ProcessFn processFn) {
  
  for (const auto &clause : clauses) {
    if (const auto *typedClause = 
            std::get_if<ClauseType>(&clause.u)) {
      processFn(*typedClause);
    }
  }
}

// Usage:
processClauseIfPresent<parser::OmpClause::NumThreads>(
    clauseList,
    [&](const auto &clause) {
      // Process NUM_THREADS clause
      numThreadsValue = evaluateExpr(clause.v);
    });
```

---

### Quick Diagnostic Checklist

When encountering lowering errors, check:

**SSA Form**:
- [ ] All values defined before use?
- [ ] No values from nested regions used in parent?
- [ ] Proper dominance relationships?

**Operation Structure**:
- [ ] Correct number of operands?
- [ ] All required attributes present?
- [ ] Regions properly terminated?
- [ ] Block arguments match expected count and types?

**Type Consistency**:
- [ ] All operand types match TableGen definition?
- [ ] Loop bounds have consistent types?
- [ ] Reference types vs value types correct?
- [ ] Array shapes properly specified?

**Symbol Resolution**:
- [ ] All variables resolved in semantic phase?
- [ ] Symbol addresses available during lowering?
- [ ] Private variable mappings updated?
- [ ] Correct scope for symbol lookup?

**Clause Processing**:
- [ ] All clauses from parse tree handled?
- [ ] Clause processing functions exist in ClauseProcessor?
- [ ] Operands/attributes passed to operation?
- [ ] Mutual exclusivity checked?

---

## Debugging LLVM IR Generation Issues

When the generated LLVM IR for OpenMP constructs is incorrect, use these systematic debugging workflows to identify and fix the problem.

---

### Workflow 1: Compare Generated IR with Expected Pattern

#### Step 1: Generate LLVM IR with Debug Information

```bash
# Generate LLVM IR with full debug info
flang-new -fc1 -fopenmp -emit-llvm -g test.f90 -o test.ll

# Generate human-readable IR
flang-new -fc1 -fopenmp -S -emit-llvm test.f90 -o test.ll

# Include source line information
flang-new -fc1 -fopenmp -emit-llvm -debug-info-kind=line-tables-only test.f90 -o test.ll
```

#### Step 2: Identify Missing or Incorrect Runtime Calls

**Check for OpenMP Runtime Functions**:
```bash
# List all OpenMP runtime calls
grep "__kmpc" test.ll

# Search for specific runtime function patterns
grep -E "__kmpc_(fork_call|barrier|for_static_init)" test.ll

# Count runtime calls
grep -c "__kmpc_fork_call" test.ll
```

**Expected Runtime Function Patterns**:
```llvm
; Parallel region should have:
call void @__kmpc_fork_call(...)

; Worksharing loop should have:
call void @__kmpc_for_static_init_*(...)
call void @__kmpc_for_static_fini(...)

; Barrier should have:
call void @__kmpc_barrier(...)

; Critical section should have:
call void @__kmpc_critical(...)
call void @__kmpc_end_critical(...)
```

#### Step 3: Verify Function Signatures

```bash
# Extract runtime function declarations
grep "declare.*@__kmpc" test.ll

# Check function signature matches OpenMP runtime
# Compare with: openmp/runtime/src/kmp.h
```

**Common Signature Issues**:
- Wrong number of parameters
- Incorrect parameter types (i32 vs i64)
- Missing `ident_t*` location parameter
- Wrong calling convention

---

### Workflow 2: Trace MLIR-to-LLVM Translation

#### Step 1: Generate Intermediate MLIR

```bash
# Stop at MLIR level
flang-new -fc1 -fopenmp -emit-mlir test.f90 -o test.mlir

# Inspect MLIR operations
cat test.mlir | grep "omp\."
```

#### Step 2: Run MLIR-to-LLVM Translation Manually

```bash
# Translate MLIR to LLVM IR using mlir-translate
mlir-translate --mlir-to-llvmir test.mlir -o test.ll

# Compare with flang-generated IR
diff -u <(flang-new -fc1 -fopenmp -emit-llvm test.f90) test.ll
```

#### Step 3: Enable Translation Debug Output

```bash
# Set MLIR debug environment
export MLIR_ENABLE_DUMP=1

# Run with verbose output
flang-new -fc1 -fopenmp -emit-llvm test.f90 -mllvm -debug 2>&1 | tee translation.log

# Filter for OpenMP translation
grep -i "openmp\|omp\." translation.log
```

---

### Workflow 3: Bisect Translation Passes

#### Identify Which Pass Produces Incorrect IR

```bash
# Run translation with pass pipeline logging
flang-new -fc1 -fopenmp -emit-llvm test.f90 \
  -mllvm -print-before-all \
  -mllvm -print-after-all \
  2>&1 | tee passes.log

# Search for specific operation transformation
grep -A 20 "IR Dump After.*OpenMP" passes.log
```

#### Check Specific Translation Functions

**File**: `mlir/lib/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.cpp`

```cpp
// Add debug output to translation functions
LogicalResult convertOmpOperation(Operation *op, ...) {
  llvm::errs() << "Translating: ";
  op->dump();
  
  // ... translation logic ...
  
  llvm::errs() << "Generated IR:\n";
  translatedBlock->dump();
  
  return success();
}
```

**Rebuild and Test**:
```bash
ninja flang-new
flang-new -fc1 -fopenmp -emit-llvm test.f90 2>&1 | grep -A 5 "Translating"
```

---

### Workflow 4: Validate Runtime Call Arguments

#### Check ident_t Location Structure

**Expected Pattern**:
```llvm
; Source location should be initialized
@.loc.source = private unnamed_addr constant %struct.ident_t {
  i32 0,                    ; reserved_1
  i32 2,                    ; flags (2 = source location)
  i32 0,                    ; reserved_2
  i32 0,                    ; reserved_3
  i8* getelementptr inbounds ([N x i8], [N x i8]* @.str, i32 0, i32 0)
}

; Used in runtime calls
call void @__kmpc_barrier(%struct.ident_t* @.loc.source, i32 %tid)
```

**Check for Issues**:
```bash
# Verify ident_t structures exist
grep "struct.ident_t" test.ll

# Check if location is null (bug!)
grep "@__kmpc.*null" test.ll
```

#### Verify Thread ID Arguments

**Expected Pattern**:
```llvm
; Get global thread ID
%gtid = call i32 @__kmpc_global_thread_num(%struct.ident_t* @.loc)

; Use in runtime calls
call void @__kmpc_barrier(%struct.ident_t* @.loc, i32 %gtid)
```

**Common Issues**:
- Using constant 0 instead of actual thread ID
- Not calling `__kmpc_global_thread_num`
- Thread ID not propagated to nested calls

```bash
# Check thread ID usage
grep "global_thread_num" test.ll
grep -A 2 "global_thread_num" test.ll | grep "call.*@__kmpc"
```

---

### Workflow 5: Inspect Outlined Function Structure

#### Verify Function Outlining

**Expected Pattern for Parallel Region**:
```llvm
; Outlined function signature
define internal void @.omp_outlined.(
    i32* noalias %tid.addr,
    i32* noalias %bound.tid.addr,
    ...captured variables...
) {
  ; Function body with original parallel region code
}

; Fork call in parent function
call void @__kmpc_fork_call(
    %struct.ident_t* @.loc,
    i32 N,                     ; Number of captured variables
    void (i32*, i32*, ...)* @.omp_outlined.,
    ...captured variable addresses...
)
```

**Debugging Outlined Functions**:
```bash
# List all outlined functions
grep "define.*@\.omp_outlined" test.ll

# Check fork_call arguments
grep -A 5 "__kmpc_fork_call" test.ll

# Verify captured variable count matches
grep "fork_call.*i32 [0-9]" test.ll
```

#### Check Captured Variable Passing

**Common Issues**:
- Missing captured variables
- Wrong number in fork_call (second i32 parameter)
- Incorrect types for captured variables
- Variables not properly loaded before fork_call

```bash
# Extract captured variable count
grep "__kmpc_fork_call" test.ll | sed 's/.*i32 \([0-9]*\).*/\1/'

# Count actual parameters to outlined function
grep "define.*@\.omp_outlined" test.ll | grep -o "," | wc -l
# Should be: 2 (tid, bound_tid) + N (captured vars)
```

---

### Workflow 6: Verify Memory Model and Synchronization

#### Check Atomic Operations

**Expected Atomic Patterns**:
```llvm
; Atomic load
%val = load atomic i32, i32* %addr seq_cst, align 4

; Atomic store
store atomic i32 %val, i32* %addr seq_cst, align 4

; Atomic RMW
%old = atomicrmw add i32* %addr, i32 %inc seq_cst

; Compare-exchange
%success = cmpxchg i32* %addr, i32 %expected, i32 %new seq_cst seq_cst
```

**Verify Atomicity**:
```bash
# Check for atomic operations
grep "atomic" test.ll

# Verify memory ordering
grep -E "(seq_cst|acquire|release|monotonic)" test.ll

# Check for missing atomics (bug: using plain load/store)
grep -A 2 "omp.*atomic" test.mlir
grep -A 2 "load.*%.*:" test.ll  # Should be "load atomic"
```

#### Check Barrier Placement

**Expected Barrier Locations**:
- End of parallel regions (unless NOWAIT)
- End of worksharing constructs (unless NOWAIT)
- Explicit barrier directives

```bash
# Count barriers
grep -c "__kmpc_barrier" test.ll

# Check barrier location (should be before function return)
grep -B 5 "call.*__kmpc_barrier" test.ll | grep -E "(ret|br)"
```

---

### Workflow 7: Debug Data-Sharing Issues

#### Verify Variable Privatization

**Expected Pattern**:
```llvm
; In outlined function
define internal void @.omp_outlined.(...) {
  ; Private variables allocated on stack
  %private.var = alloca i32, align 4
  
  ; Original shared variable passed as parameter
  ; Use private copy in parallel region
  store i32 %init_val, i32* %private.var
  ; ... use %private.var ...
}
```

**Check for Issues**:
```bash
# Verify alloca in outlined function for private vars
grep -A 20 "define.*@\.omp_outlined" test.ll | grep "alloca"

# Check for incorrect shared variable access (bug!)
# Should use private copy, not original address
```

#### Verify Firstprivate Initialization

**Expected Pattern**:
```llvm
; Copy value from shared to private
%shared.val = load i32, i32* %shared.addr
store i32 %shared.val, i32* %private.addr
```

#### Verify Lastprivate Write-back

**Expected Pattern**:
```llvm
; After parallel region, copy back to shared
%private.val = load i32, i32* %private.addr
store i32 %private.val, i32* %shared.addr
```

---

### Workflow 8: Runtime Execution Testing

#### Use OpenMP Runtime Debug Mode

```bash
# Enable OpenMP runtime debugging
export OMP_DISPLAY_ENV=TRUE
export KMP_VERSION=1
export KMP_SETTINGS=1

# Run compiled program
./test

# Observe runtime behavior:
# - Number of threads
# - Nested parallelism
# - Thread affinity
# - Schedule types
```

#### Use Runtime Stubs for Tracing

**Create stub runtime** (`stub_runtime.c`):
```c
#include <stdio.h>

void __kmpc_fork_call(void *loc, int nargs, void *fn, ...) {
    printf("__kmpc_fork_call: loc=%p nargs=%d fn=%p\n", loc, nargs, fn);
    // Don't actually execute - just trace
}

void __kmpc_barrier(void *loc, int gtid) {
    printf("__kmpc_barrier: loc=%p gtid=%d\n", loc, gtid);
}

// Add stubs for other runtime functions...
```

**Compile and Link**:
```bash
# Compile stub runtime
clang -c stub_runtime.c -o stub_runtime.o

# Link with test program (override real runtime)
clang test.ll stub_runtime.o -o test_traced

# Run to see call sequence
./test_traced
```

---

### Workflow 9: Compare with Working Reference

#### Generate Reference IR from Clang

```c
// reference.c - Equivalent C code with OpenMP
#pragma omp parallel
{
    // Parallel region
}
```

```bash
# Generate reference LLVM IR
clang -fopenmp -S -emit-llvm reference.c -o reference.ll

# Compare structure
diff -u reference.ll test.ll

# Extract OpenMP-specific parts
grep "__kmpc" reference.ll > reference_omp.txt
grep "__kmpc" test.ll > test_omp.txt
diff -u reference_omp.txt test_omp.txt
```

#### Use LLVM IR Diff Tools

```bash
# Structural comparison
llvm-diff test.ll reference.ll

# Compare function signatures
grep "define.*@__kmpc" test.ll > test_sigs.txt
grep "define.*@__kmpc" reference.ll > ref_sigs.txt
diff test_sigs.txt ref_sigs.txt
```

---

### Workflow 10: Automated Validation Checks

#### Create IR Validation Script

**Script**: `validate_omp_ir.sh`
```bash
#!/bin/bash
IR_FILE=$1

echo "=== OpenMP LLVM IR Validation ==="

# Check 1: Runtime function declarations present
echo "[CHECK] Runtime function declarations..."
required_funcs="__kmpc_global_thread_num __kmpc_fork_call"
for func in $required_funcs; do
    if ! grep -q "declare.*@$func" "$IR_FILE"; then
        echo "  [FAIL] Missing declaration: $func"
    else
        echo "  [PASS] Found: $func"
    fi
done

# Check 2: ident_t structures initialized
echo "[CHECK] Source location structures..."
if ! grep -q "struct.ident_t" "$IR_FILE"; then
    echo "  [FAIL] No ident_t structures found"
else
    count=$(grep -c "struct.ident_t.*{" "$IR_FILE")
    echo "  [PASS] Found $count location structures"
fi

# Check 3: Thread ID usage
echo "[CHECK] Thread ID management..."
if ! grep -q "__kmpc_global_thread_num" "$IR_FILE"; then
    echo "  [WARN] No thread ID calls (may use constant)"
fi

# Check 4: Balanced calls (e.g., critical/end_critical)
echo "[CHECK] Balanced runtime calls..."
critical_start=$(grep -c "__kmpc_critical" "$IR_FILE")
critical_end=$(grep -c "__kmpc_end_critical" "$IR_FILE")
if [ "$critical_start" -ne "$critical_end" ]; then
    echo "  [FAIL] Unbalanced critical: start=$critical_start end=$critical_end"
else
    echo "  [PASS] Balanced critical sections: $critical_start"
fi

# Check 5: Outlined functions called
echo "[CHECK] Outlined functions..."
outlined=$(grep -c "define.*@\.omp_outlined" "$IR_FILE")
fork_calls=$(grep -c "__kmpc_fork_call" "$IR_FILE")
if [ "$outlined" -ne "$fork_calls" ]; then
    echo "  [WARN] Outlined functions ($outlined) != fork_calls ($fork_calls)"
else
    echo "  [PASS] Outlined functions match fork_calls: $outlined"
fi

echo "=== Validation Complete ==="
```

**Usage**:
```bash
chmod +x validate_omp_ir.sh
./validate_omp_ir.sh test.ll
```

---

### Common IR Generation Issues and Fixes

#### Issue 1: Missing Runtime Calls

**Symptom**: IR has OpenMP structure but no `__kmpc_*` calls

**Diagnosis**:
```bash
# Check MLIR has OpenMP ops
grep "omp\." test.mlir

# Check IR has runtime calls
grep "__kmpc" test.ll
```

**Cause**: Translation layer not invoking OpenMPIRBuilder

**Fix**: Ensure `convertOmpOperation` is registered
```cpp
// In OpenMPToLLVMIRTranslation.cpp
void mlir::registerOpenMPDialectTranslation(DialectRegistry &registry) {
  registry.insert<omp::OpenMPDialect>();
  registry.addExtension(+[](MLIRContext *ctx, omp::OpenMPDialect *dialect) {
    dialect->addInterfaces<OpenMPDialectLLVMIRTranslationInterface>();
  });
}
```

---

#### Issue 2: Incorrect Number of Fork Call Arguments

**Symptom**: 
```llvm
call void @__kmpc_fork_call(..., i32 2, ...)  ; Says 2 captured vars
; But outlined function has 5 parameters
```

**Diagnosis**:
```bash
# Extract fork_call argument count
grep "fork_call" test.ll | sed 's/.*i32 \([0-9]*\).*/\1/'

# Count outlined function parameters (subtract 2 for tid/bound_tid)
grep "define.*@\.omp_outlined" test.ll
```

**Cause**: Mismatch between captured variables counted and actually passed

**Fix**: Ensure all captured variables added to fork_call:
```cpp
// In translation code
unsigned numCaptured = capturedVars.size();
args.push_back(builder.getInt32(numCaptured));  // Must match!
args.push_back(outlinedFn);
for (auto var : capturedVars) {
  args.push_back(var);  // All must be passed
}
```

---

#### Issue 3: Wrong Memory Ordering on Atomics

**Symptom**: Data race despite atomic directive

**Diagnosis**:
```bash
# Check atomic memory order
grep "load atomic" test.ll
grep "store atomic" test.ll
grep "atomicrmw" test.ll
```

**Cause**: Using `monotonic` instead of `seq_cst`

**Expected**:
```llvm
load atomic i32, i32* %addr seq_cst, align 4
```

**Fix**: Use correct memory ordering in MLIR-to-LLVM translation:
```cpp
// Map OpenMP memory ordering to LLVM
auto order = llvm::AtomicOrdering::SequentiallyConsistent;
if (atomicOp.getMemoryOrder() == omp::ClauseMemoryOrderKind::AcqRel)
  order = llvm::AtomicOrdering::AcquireRelease;
```

---

### Debugging Checklist

When LLVM IR is incorrect, verify:

**Structure**:
- [ ] OpenMP runtime functions declared?
- [ ] `ident_t` location structures present?
- [ ] Outlined functions created?
- [ ] Fork calls present for parallel regions?

**Arguments**:
- [ ] Thread ID obtained via `__kmpc_global_thread_num`?
- [ ] Thread ID passed to all runtime calls?
- [ ] Captured variable count matches in fork_call?
- [ ] All captured variables passed to outlined function?

**Synchronization**:
- [ ] Barriers present where expected?
- [ ] Critical sections balanced (start/end)?
- [ ] Atomic operations have correct memory ordering?
- [ ] NOWAIT clause properly removes barriers?

**Data Sharing**:
- [ ] Private variables allocated in outlined function?
- [ ] Firstprivate variables initialized?
- [ ] Lastprivate variables written back?
- [ ] Shared variables passed as parameters?

**Memory Safety**:
- [ ] No null pointer dereferences in runtime calls?
- [ ] Proper alignment on atomic operations?
- [ ] Stack allocations for private variables?
- [ ] No dangling pointers from outlined functions?

---

# PART 5: QUICK REFERENCE

## OpenMP Directive Categories

**Parallel Constructs**: `parallel`, `parallel do`, `parallel sections`  
**Worksharing**: `do`, `sections`, `section`, `single`, `workshare`  
**Tasking**: `task`, `taskloop`, `taskgroup`, `taskwait`, `taskyield`  
**Synchronization**: `barrier`, `critical`, `atomic`, `flush`, `ordered`  
**Cancellation**: `cancel`, `cancellation point`  
**Data Environment**: `threadprivate`, `declare reduction`, `declare mapper`  
**Device**: `target`, `target data`, `target enter data`, `target exit data`, `target update`  
**Loop**: `loop`, `simd`, `distribute`, `teams distribute`  
**Miscellaneous**: `master`, `masked`, `scan`, `depobj`, `allocate`

## Important Clause Categories

**Data-Sharing**: `private`, `firstprivate`, `lastprivate`, `shared`, `default`, `threadprivate`  
**Data-Mapping**: `map`, `to`, `from`, `tofrom`, `alloc`, `release`, `delete`  
**Reduction**: `reduction`, `in_reduction`, `task_reduction`  
**Synchronization**: `nowait`, `ordered`, `collapse`, `bind`  
**Scheduling**: `schedule`, `dist_schedule`, `grainsize`, `num_tasks`  
**Device**: `device`, `num_teams`, `thread_limit`, `priority`  
**Dependencies**: `depend`, `depobj`, `affinity`

## Testing Commands

```bash
# Run OpenMP semantic tests
./bin/flang-new -fc1 -fopenmp -fsyntax-only test.f90

# Generate MLIR with OpenMP dialect
./bin/flang-new -fc1 -fopenmp -emit-mlir test.f90

# Generate LLVM IR
./bin/flang-new -fc1 -fopenmp -emit-llvm test.f90

# Run OpenMP test suite
./bin/llvm-lit -v flang/test/Semantics/OpenMP/
./bin/llvm-lit -v flang/test/Lower/OpenMP/
```

---

# PART 6: RESOURCES

## Specifications
- [OpenMP 5.2 Specification](https://www.openmp.org/specifications/)
- [OpenMP 6.0 Draft Features](https://github.com/OpenMP/spec/tree/master/openmp-6.0)

## Key Documentation
- Flang OpenMP Implementation: `flang/docs/OpenMP-semantics.md`
- MLIR OpenMP Dialect: `mlir/docs/Dialects/OpenMPDialect.md`
- OpenMP Runtime: `openmp/runtime/src/`

## Related LLVM Projects
- `openmp/` - OpenMP runtime library (libomp)
- `offload/` - Offloading support for accelerators
- `llvm/lib/Frontend/OpenMP/` - LLVM-level OpenMP utilities

---

**Last Updated**: January 26, 2026  
**Status**: Active tracking of OpenMP feature development across compiler stack
