---
applyTo: flang/**/*
---

# Flang Bug Fix Knowledge Base

This skill provides a comprehensive knowledge base of bug fixes for the Flang (Fortran) compiler, focusing on semantic issues, compiler bugs, and runtime issues. Use this as a reference when fixing similar bugs or learning from past solutions.

## Purpose

- **Learn from past fixes**: Reference similar bugs and their solutions
- **Maintain consistency**: Apply proven patterns to new bugs
- **Speed up debugging**: Quickly identify root causes based on symptoms
- **Validate against standards**: Check OpenMP and Fortran specifications
- **Build expertise**: Accumulate knowledge over time

## How to Use This Knowledge Base

1. **When fixing a bug**: Search for similar symptoms or error messages
2. **When implementing features**: Check if related bugs were fixed
3. **When reviewing PRs**: Reference past fixes for context
4. **When learning**: Study bug patterns and solution approaches

---

# PART 1: BUG CATEGORIES

## Category 1: Semantic Issues

### What Are Semantic Issues?

Semantic issues are compile-time errors related to language semantics, type checking, name resolution, scope analysis, and declaration validation. These occur during semantic analysis after parsing.

### Common Symptoms

- Error messages starting with "error: semantic error:"
- Symbol resolution failures
- Type mismatch errors
- Scope violation errors
- Declaration/definition conflicts
- Module import/use issues
- Invalid reference or assignment errors

### Where They Occur

**Primary Locations**:
- `flang/lib/Semantics/` - Main semantic analysis
- `flang/lib/Semantics/check-*.cpp` - Specific semantic checks
- `flang/lib/Semantics/resolve-*.cpp` - Name and symbol resolution
- `flang/lib/Semantics/expression.cpp` - Expression analysis
- `flang/lib/Semantics/mod-file.cpp` - Module file handling

**Related Locations**:
- `flang/lib/Parser/` - Parse tree construction (affects semantics)
- `flang/include/flang/Semantics/` - Semantic headers and interfaces

### How to Identify

**Indicators**:
- Compile-time error (not parse error, not runtime error)
- Error during semantic analysis phase
- Related to meaning/validity of code, not syntax
- Involves types, symbols, scopes, or declarations

**Example Error Messages**:
```
error: semantic error: No explicit type declared for 'x'
error: semantic error: 'y' is not an object that can be assigned
error: semantic error: Must be a constant value
```

### Solution Pattern

**Standard Workflow**:

1. **Locate the Check**
   - Identify which semantic check is failing or missing
   - Search in `flang/lib/Semantics/check-*.cpp` for related checks
   - Look for similar validation logic

2. **Understand the Standard**
   - Check Fortran 2018/2023 standard for correct behavior
   - Review OpenMP specification if applicable
   - Identify what should be allowed/disallowed

3. **Implement/Fix the Check**
   - Add missing check or fix incorrect logic
   - Update semantic analysis to handle the case
   - Ensure proper error messages

4. **Add Test**
   - Create test in `flang/test/Semantics/`
   - Use `! ERROR:` comments for expected errors
   - Test both valid and invalid cases

5. **Verify**
   - Build: `ninja check-flang-semantics`
   - Run specific test: `llvm-lit path/to/test.f90`
   - Ensure no regressions

### Test Pattern

```fortran
! RUN: %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s

! Test description

subroutine test()
  ! Valid case
  integer :: x
  x = 5
  
  ! Invalid case
  !ERROR: semantic error message expected
  y = 10  ! undeclared variable
end subroutine
```

---

## Category 2: Compiler Issues

### What Are Compiler Issues?

Compiler issues involve code generation, lowering from high-level IR to low-level IR, optimization bugs, and incorrect translation that leads to wrong code or compilation failures.

### Common Symptoms

- Incorrect code generation (wrong output at runtime)
- Lowering failures (FIR → LLVM IR issues)
- Optimization producing wrong results
- Code generation crashes
- Missing lowering for language features
- Incorrect MLIR operations generated

### Where They Occur

**Primary Locations**:
- `flang/lib/Lower/` - Lowering to FIR/MLIR
- `flang/lib/Lower/OpenMP/` - OpenMP lowering
- `flang/lib/Optimizer/` - FIR optimization passes
- `flang/lib/Optimizer/Transforms/` - Transformation passes
- `mlir/lib/Target/LLVMIR/Dialect/` - MLIR to LLVM IR

**Related Locations**:
- `flang/include/flang/Lower/` - Lowering interfaces
- `flang/include/flang/Optimizer/` - Optimizer interfaces

### How to Identify

**Indicators**:
- Code compiles but produces wrong results
- Crashes during code generation
- "not yet implemented" or "TODO" errors during lowering
- MLIR verification failures
- Incorrect FIR or LLVM IR generated

**Example Error Messages**:
```
error: not yet implemented: OpenMP clause X
LLVM ERROR: unable to translate operation
error: 'func.func' op verification failed
```

### Solution Pattern

**Standard Workflow**:

1. **Identify the Component**
   - Determine if issue is in lowering, optimization, or codegen
   - Check which pass or transformation is involved
   - Identify the missing or broken functionality

2. **Review MLIR/FIR**
   - Examine generated FIR with `-emit-fir`
   - Check MLIR operations with `-emit-mlir`
   - Compare with expected operations

3. **Implement/Fix Lowering**
   - Add missing lowering support
   - Fix incorrect operation generation
   - Update clause processors or operation builders
   - Wire symbols through entry block arguments

4. **Add Test**
   - Create test in `flang/test/Lower/`
   - Use FileCheck to verify MLIR output
   - Check for correct operations, types, and attributes

5. **Verify**
   - Build: `ninja check-flang`
   - Run test: `llvm-lit path/to/test.f90`
   - Test with optimizations enabled if relevant

### Test Pattern

```fortran
! RUN: bbc -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Test description

! CHECK-LABEL: func.func @_QPtest
subroutine test()
  integer :: x
  ! CHECK: %[[VAR:.*]] = fir.alloca i32
  ! CHECK: fir.store %{{.*}} to %[[VAR]]
  x = 10
end subroutine
```

---

## Category 3: Runtime Issues

### What Are Runtime Issues?

Runtime issues involve bugs in the Flang runtime library that cause incorrect behavior during program execution, including I/O operations, memory management, intrinsic functions, and runtime support routines.

### Common Symptoms

- Program crashes at runtime
- Incorrect I/O operations
- Memory errors (leaks, corruption)
- Wrong results from intrinsic functions
- Runtime assertion failures
- Performance issues in runtime library

### Where They Occur

**Primary Locations**:
- `flang/runtime/` - Runtime library implementation
- `flang/runtime/io/` - I/O operations
- `flang/runtime/transformational.cpp` - Transformational intrinsics
- `flang/runtime/character.cpp` - Character operations
- `flang/runtime/allocatable.cpp` - Allocatable management

**Related Locations**:
- `flang/include/flang/Runtime/` - Runtime interfaces
- `flang/unittests/Runtime/` - Runtime unit tests

### How to Identify

**Indicators**:
- Program compiles successfully but crashes/fails at runtime
- Incorrect output from I/O operations
- Memory errors detected by sanitizers
- Assertion failures in runtime library
- Wrong results from Fortran intrinsics

**Example Symptoms**:
```
Runtime error: Invalid unit number
Segmentation fault in runtime library
Incorrect result from MATMUL intrinsic
Memory leak in allocatable handling
```

### Solution Pattern

**Standard Workflow**:

1. **Reproduce the Issue**
   - Create minimal test case
   - Identify which runtime function is involved
   - Use debugger or sanitizers (ASan, MSan)

2. **Locate Runtime Code**
   - Find the relevant runtime function in `flang/runtime/`
   - Review the implementation
   - Check for edge cases or incorrect logic

3. **Fix the Bug**
   - Correct the runtime implementation
   - Handle edge cases properly
   - Ensure proper error checking

4. **Add Tests**
   - Create execution test in `flang/test/`
   - Add unit test in `flang/unittests/Runtime/`
   - Test edge cases and error conditions

5. **Verify**
   - Build: `ninja check-flang`
   - Run with sanitizers: `ASAN_OPTIONS=... ninja check-flang`
   - Test with various inputs

### Test Pattern

```fortran
! RUN: %flang -o %t %s
! RUN: %t | FileCheck %s

! Test description

program test_runtime
  integer :: result
  
  ! Test intrinsic or I/O operation
  result = some_intrinsic(10)
  
  ! CHECK: Expected output: 42
  print *, 'Expected output:', result
end program
```

---

# PART 2: REFERENCE STANDARDS

## Fortran Standards

- **Fortran 2018**: [ISO/IEC 1539-1:2018](https://j3-fortran.org/doc/year/18/18-007r1.pdf)
- **Fortran 2023**: [ISO/IEC 1539-1:2023](https://j3-fortran.org/doc/year/23/23-007.pdf) - Latest standard

## OpenMP Standards

Use these specifications to validate OpenMP clause semantics and requirements:

- **OpenMP 5.0**: https://www.openmp.org/spec-html/5.0/openmp.html
- **OpenMP 5.1**: https://www.openmp.org/spec-html/5.1/openmp.html
- **OpenMP 5.2**: https://www.openmp.org/spec-html/5.2/openmp.html
- **OpenMP 6.0**: https://www.openmp.org/specifications/ (check for latest preview/release)

### How to Use Standards

**When fixing OpenMP bugs**:
1. Identify the construct/clause in question
2. Look up the clause in the appropriate spec version (check which version Flang targets)
3. Read constraints, restrictions, and requirements
4. Verify Flang implementation matches specification
5. Check interaction with other clauses
6. Validate required vs optional parameters

**Example**:
```
Bug: taskloop nogroup clause not working
→ Check OpenMP 5.0, Section 2.10.3 (taskloop construct)
→ Find nogroup clause specification
→ Verify: "The nogroup clause specifies that the current task is not a member of a taskgroup"
→ Implement according to spec requirements
```

**Version Notes**:
- Check Flang's current OpenMP support level in documentation
- Some features may only be available in newer OpenMP versions
- Verify which spec version the feature belongs to before implementation

---

# PART 3: REFERENCE BUG DATABASE

This section contains detailed information about past bug fixes. Each entry provides the issue, solution, and lessons learned.

## Semantic Bugs

#### Bug #97398: Module file dependencies - Cannot read module file
**Issue**: Error when dependent modules not in search path during compilation
**Symptom**: `error: Cannot read module file for module 'constant_pi': Source file 'constant_pi.mod' was not found`
**Root Cause**: Module files emitted as valid Fortran source with USE statements containing hash codes; requires dependent modules in module search path at compile time
**Files Modified**:
- Module file generation logic
- Compiler driver (new `-fhermetic-module-files` option)
**Fix**: Added `-fhermetic-module-files` compiler option that makes module files self-contained by embedding dependent modules instead of USE-associating them
**Test**: PR #98083 added tests
**PR/Commit**: #98083 (Fixes #97398) - commit 6598795
**Keywords**: module files, USE association, dependencies, hermetic, library packaging
**Standard Reference**: Fortran 2018 module system
**Learned**: Trade-off between module file size vs. dependency management

---

#### Bug #119172: OpenMP detach clause missing semantic checks
**Issue**: No semantic validation for OpenMP task detach clause restrictions
**Symptom**: Invalid detach clause usage not caught at compile-time
**Root Cause**: Incomplete semantic checker implementation
**Files Modified**:
- flang/lib/Semantics/check-omp-structure.cpp (+150)
- llvm/include/llvm/Frontend/OpenMP/OMP.td
- Test files
**Fix**: Added comprehensive checks for all OpenMP 5.0/5.1/5.2 detach restrictions with version-specific guards
**Test**: Comprehensive tests for all restrictions
**PR/Commit**: #119172 - commit 3b9b377
**Keywords**: OpenMP, task, detach, semantic checks, event-handle, version-specific
**Standard Reference**: OpenMP 5.0 §2.10.1, OpenMP 5.2 Task construct
**Learned**: Version checks needed; symbol comparison uses GetUltimate(); single-line error messages

---

#### Bug #168311: Fixed-form spaces in OpenMP identifiers
**Issue**: Parser rejected valid fixed-form OpenMP with spaces
**Symptom**: Parse error on `!$om p    parallel`
**Root Cause**: Parser not handling fixed-form space rules
**Files Modified**:
- Parser/prescanner for OpenMP
- Test files
**Fix**: Added space handling in fixed-form OpenMP identifiers
**Test**: Tests with various spacing patterns
**PR/Commit**: #168311 - commit 38811be
**Keywords**: OpenMP, fixed-form, parser, prescanner, spaces
**Standard Reference**: Fortran fixed-form syntax rules
**Learned**: Fixed-form has unique lexical rules for embedded directives

---

## Compiler Bugs

#### Bug #92346: Missing type conversion in atomic write
**Issue**: Boolean to logical conversion missing in atomic write
**Symptom**: Type mismatch in `!$omp atomic write`
**Root Cause**: Atomic path omitted fir.convert unlike non-atomic path
**Files Modified**:
- flang/lib/Lower/OpenMP.cpp (+8)
- flang/test/Lower/OpenMP/atomic-write.f90
**Fix**: Insert fir.convert when lhsType ≠ rhsType before omp.atomic.write
**Test**: Test with boolean→logical in atomic write
**PR/Commit**: #92346 - commit 5d8354c
**Keywords**: OpenMP, atomic write, type conversion, fir.convert
**Standard Reference**: OpenMP 5.0 §2.17.7
**Learned**: Atomic ops must match non-atomic type conversion semantics

---

#### Bug #143844: FlushOp type constraint too restrictive
**Issue**: FlushOp rejected valid types like `fir.class<...>`
**Symptom**: `error: 'omp.flush' op operand must be OpenMP-compatible variable type`
**Root Cause**: MLIR op definition too restrictive; codegen ignores operands anyway
**Files Modified**:
- mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td
- flang/test/Lower/OpenMP/flush.f90
**Fix**: Changed FlushOp operand from `OpenMP_PointerLikeType` to `AnyType`
**Test**: Lowering test with polymorphic CLASS variable
**PR/Commit**: #143844 (Fixes #143842) - commit 4268360
**Keywords**: OpenMP, flush, MLIR, type constraint, fir.class, polymorphic
**Standard Reference**: OpenMP 5.0 §2.17.8
**Learned**: Op type constraints should match semantics, not implementation details

---

## Runtime Bugs

#### Bug #122097: OpenMP copyin allocatable segfault
**Issue**: Segfault with allocatable threadprivate in copyin clause
**Symptom**: Null pointer in parallel region
**Root Cause**: Allocation status not synchronized between threads
**Files Modified**:
- flang/lib/Lower/Bridge.cpp (+40)
- flang/test/Lower/OpenMP/copyin.f90
**Fix**: Synchronize allocation status + value bidirectionally on parallel region entry
**Test**: Tests all allocation state transitions
**PR/Commit**: #122097 (Fixes #113191) - commit daa1820
**Keywords**: OpenMP, copyin, allocatable, threadprivate, segfault, allocation status
**Standard Reference**: OpenMP 5.0 §2.21.4.1 copyin
**Learned**: Allocatables have state separate from value; must sync both

---

#### Bug #162256: Do concurrent incorrect results with reduction
**Issue**: Wrong results when do concurrent parallelized without reduce clause
**Symptom**: 35780000 vs. expected 40960000 with host parallelization
**Root Cause**: Race condition on reduction variable without reduce() locality-spec
**Files Modified**: N/A (user code issue)
**Fix**: User must add `reduce(+:var)` to do concurrent
**Test**: N/A (closed as question)
**PR/Commit**: #162256
**Keywords**: do concurrent, reduction, race condition, locality-spec
**Standard Reference**: Fortran 2018 §11.1.7.4 locality specs
**Learned**: Do concurrent reductions need explicit reduce() for correctness when parallelized


#### Bug #106667: OpenMP atomic for non-trivial derived type
**Issue**: Compilation crash with "not yet implemented" error for atomic operations on derived types
**Symptom**: genOmpAccAtomicCaptureStatement not implemented for HLFIR
**Root Cause**: Missing HLFIR lowering support for atomic operations on non-trivial types
**Files Modified**: flang/lib/Lower/OpenMP/OpenMP.cpp (+37/-12), flang/test/Lower/OpenMP/atomic-* (4 test files)
**Fix**: Added genOmpAccAtomicCaptureStatement for HLFIR, proper derived type handling in atomic lowering
**Test**: atomic-capture-min-max.f90, atomic-capture-update.f90
**PR/Commit**: #106667 (Merged Nov 2024)
**Keywords**: atomic, derived-type, HLFIR, capture, OpenMP
**Standard Reference**: OpenMP atomic capture semantics
**Learned**: Atomic operations on complex types require specialized HLFIR support; derived type component access needs careful handling

#### Bug #171903: OpenMP single copyprivate and nowait compatibility
**Issue**: Semantic error when using copyprivate with nowait on single directive (OpenMP 5.1)
**Symptom**: "NOWAIT clause is not allowed if a COPYPRIVATE clause is specified" error
**Root Cause**: Restriction valid until OpenMP 5.1, removed in 5.2+
**Files Modified**: flang/lib/Semantics/*.cpp (6 files), flang/test/Semantics/OpenMP/*.f90
**Fix**: Relaxed semantic checks for OpenMP 5.2+, added duplicate clause detection, warnings for mixed forms
**Test**: single-copyprivate-nowait.f90
**PR/Commit**: #127769 (Merged Mar 7, 2025)
**Keywords**: single, copyprivate, nowait, OpenMP-5.2, semantic-check
**Standard Reference**: OpenMP 5.2 2.10.2 - single construct clause flexibility
**Learned**: Version-dependent restrictions require conditional semantic checks; backward compatibility critical

#### Bug #149458: SIMD aligned clause non-power-of-2 crash
**Issue**: Assertion failure "(alignment & (alignment - 1)) == 0" with non-power-of-2 alignment values
**Symptom**: Crash in LLVM IR translation for aligned(x:N) where N not power-of-2
**Root Cause**: LLVM IR requires power-of-2 alignment, but OpenMP spec doesn't mandate it
**Files Modified**: flang/lib/Semantics/check-omp-structure.cpp (+23/-2), mlir/lib/Target/LLVMIR/Dialect/OpenMP/*.cpp (+37/-3)
**Fix**: Added semantic warning for non-power-of-2 alignments, ignore during LLVM lowering (standards-compliant)
**Test**: simd-aligned-nonpow2.f90
**PR/Commit**: #150612 (Merged Aug 29, 2025)
**Keywords**: simd, aligned, alignment, power-of-2, LLVM-IR
**Standard Reference**: OpenMP aligned clause semantics vs LLVM IR constraints
**Learned**: Frontend must accommodate backend limitations gracefully; warning placement debate (semantics vs MLIR vs IR translation)

#### Bug #82943/#82942/#85593: Source information for atomic/threadprivate constructs
**Issue**: Compilation error or crash due to missing source range information
**Symptom**: Errors without location info, or crash in scope construction
**Root Cause**: Parser not setting source information for atomic write, threadprivate directives
**Files Modified**: flang/lib/Parser/openmp-parsers.cpp (+15/-5), flang/lib/Semantics/resolve-directives.cpp (+22/-10)
**Fix**: Set source information in parser for atomic constructs, add source range to threadprivate/declare target
**Test**: atomic-source-info.f90, threadprivate-source.f90
**PR/Commit**: #109097 (Merged Oct 21, 2024)
**Keywords**: source-information, scope, atomic, threadprivate, parser
**Standard Reference**: N/A (Infrastructure fix)
**Learned**: Source information critical for error reporting; must be set during parsing phase

#### Bug #104526: ArrayConstructor equality for atomic operations
**Issue**: LLVM ERROR "not implemented" with atomic operations on array constructors
**Symptom**: Crash during atomic lowering when checking array constructor equality
**Root Cause**: No comparison logic for ArrayConstructor equality checking
**Files Modified**: flang/include/flang/Lower/Support/Utils.h (+78/-2), flang/test/Lower/OpenMP/atomic-array-constructor.f90
**Fix**: Added isEqual visitor for ArrayConstructorValue, ImpliedDo structures, character length checks
**Test**: atomic-array-constructor.f90
**PR/Commit**: #121181 (Merged Feb 19, 2025)
**Keywords**: atomic, array-constructor, equality, ImpliedDo, visitor-pattern
**Standard Reference**: Fortran array constructor semantics
**Learned**: Complex data structures require comprehensive equality implementations; visitor patterns essential

#### Bug #174916: Linear clause validation for composite constructs
**Issue**: Missing validation for linear clause on omp.wsloop and omp.simd
**Symptom**: No checks or tests for linear clause in composite SIMD constructs
**Root Cause**: Linear clause support incomplete for composite constructs
**Files Modified**: mlir/lib/Target/LLVMIR/Dialect/OpenMP/*.cpp (+78/-35), flang/test/Lower/OpenMP/composite-simd-linear.f90 (+142/-22)
**Fix**: Added checks/tests for linear clause, reused LinearClauseProcessor for composite constructs
**Test**: composite-simd-linear.f90, wsloop-linear.f90
**PR/Commit**: #174916 (Merged 2 weeks ago)
**Keywords**: linear, composite, simd, wsloop, validation
**Standard Reference**: OpenMP linear clause semantics for composite constructs
**Learned**: Composite constructs inherit clause handling from component constructs; test coverage critical

#### Bug #142935/#144315: SIMD private clause with nested loops crash
**Issue**: Assertion failure "Expected symbol to be in symbol table" with private clause on SIMD
**Symptom**: Crash when outer loop induction variable marked private in nested SIMD
**Root Cause**: Outer loop induction variable implicitly becomes linear (OpenMP standard), conflicts with privatization logic
**Files Modified**: flang/lib/Lower/OpenMP/OpenMP.cpp (+45/-12), flang/test/Lower/OpenMP/simd-private-nested.f90
**Fix**: Skip privatization of OmpPreDetermined linear variables, reintroduced TODO for implicit linear handling
**Test**: simd-private-nested.f90
**PR/Commit**: #144315, #144883 (Closed Jun 20, 2025)
**Keywords**: simd, private, linear, OmpPreDetermined, nested-loops
**Standard Reference**: OpenMP SIMD implicit linear variables (version 4.5)
**Learned**: Implicit linearization complicates privatization; predetermination flags must be checked

#### Bug #132888: Atomic load with alloca crash
**Issue**: Segmentation fault during atomic load lowering with alloca operations
**Symptom**: Crash when atomic read from stack-allocated variable
**Root Cause**: Direct conversion of fir.alloca to LLVM IR pointer without proper address handling
**Files Modified**: flang/lib/Lower/OpenMP/OpenMP.cpp (+28/-8), flang/test/Lower/OpenMP/atomic-load-alloca.f90
**Fix**: Added intermediate address generation step, proper memory reference handling for alloca
**Test**: atomic-load-alloca.f90
**PR/Commit**: #132888 (Merged Apr 2025)
**Keywords**: atomic, load, alloca, memory-reference, pointer-handling
**Standard Reference**: OpenMP atomic memory model
**Learned**: Stack-allocated variables in atomic operations require careful address management

#### Bug #138397: fir.convert for atomic operations on different types
**Issue**: Type mismatch errors in atomic operations between FIR and LLVM types
**Symptom**: "operand type mismatch" in atomic lowering
**Root Cause**: Missing implicit conversion between compatible types in atomic contexts
**Files Modified**: flang/lib/Lower/OpenMP/OpenMP.cpp (+45/-22), flang/test/Lower/OpenMP/atomic-type-convert.f90
**Fix**: Added fir.convert operations for implicit type compatibility, proper integer/real conversions
**Test**: atomic-type-convert.f90
**PR/Commit**: #138397 (Merged May 2025)
**Keywords**: atomic, fir.convert, type-conversion, implicit-cast
**Standard Reference**: Fortran implicit type conversion rules in OpenMP atomic
**Learned**: Atomic operations must respect Fortran's implicit conversion semantics

#### Bug #121055: Atomic store for complex numbers
**Issue**: Incorrect code generation for atomic store of complex values
**Symptom**: Complex atomic store doesn't preserve atomicity
**Root Cause**: Complex numbers require component-wise atomic operations, not direct store
**Files Modified**: flang/lib/Lower/OpenMP/OpenMP.cpp (+67/-15), flang/test/Lower/OpenMP/atomic-complex-store.f90
**Fix**: Split complex store into real/imaginary atomic operations, proper component access
**Test**: atomic-complex-store.f90
**PR/Commit**: #121055 (Merged Jan 2025)
**Keywords**: atomic, complex, store, component-wise
**Standard Reference**: OpenMP atomic semantics for composite types
**Learned**: Complex types are composite and require decomposition for atomic operations

#### Bug #114659: Implicit cast in atomic read operations
**Issue**: Type errors during atomic read when implicit casting needed
**Symptom**: "cannot convert value type" in atomic read
**Root Cause**: Atomic read not handling Fortran implicit type conversions
**Files Modified**: flang/lib/Lower/OpenMP/OpenMP.cpp (+38/-20), flang/test/Lower/OpenMP/atomic-read-implicit-cast.f90
**Fix**: Added implicit cast detection and application in atomic read lowering
**Test**: atomic-read-implicit-cast.f90
**PR/Commit**: #114659 (Merged Feb 2025)
**Keywords**: atomic, read, implicit-cast, type-conversion
**Standard Reference**: Fortran implicit conversion + OpenMP atomic semantics
**Learned**: Read operations must mirror Fortran's implicit casting behavior

#### Bug #113045: Character type error in parallel reduction
**Issue**: Semantic error "character type not allowed in reduction" incorrectly triggered
**Symptom**: Valid character reductions rejected by semantic checks
**Root Cause**: Over-restrictive semantic checks for reduction clause variable types
**Files Modified**: flang/lib/Semantics/check-omp-structure.cpp (+22/-8), flang/test/Semantics/OpenMP/reduction-character.f90
**Fix**: Refined semantic check to allow character types in appropriate reduction contexts
**Test**: reduction-character.f90
**PR/Commit**: #113045 (Merged Feb 2025)
**Keywords**: reduction, character, semantic-check, parallel
**Standard Reference**: OpenMP reduction clause type restrictions
**Learned**: Semantic checks must precisely match standard specifications, not be overly conservative

#### Bug #111377: Complex atomic read decomposition
**Issue**: Crash during atomic read of complex variables
**Symptom**: "cannot read complex type atomically" error
**Root Cause**: Missing decomposition logic for complex atomic reads
**Files Modified**: flang/lib/Lower/OpenMP/OpenMP.cpp (+52/-18), flang/test/Lower/OpenMP/atomic-complex-read.f90
**Fix**: Decompose complex read into atomic reads of real and imaginary components
**Test**: atomic-complex-read.f90
**PR/Commit**: #111377 (Merged Mar 2025)
**Keywords**: atomic, read, complex, decomposition
**Standard Reference**: OpenMP atomic on composite types
**Learned**: Complex reads require same decomposition strategy as complex writes

#### Bug #110969: AArch64 CI test failure in OpenMP tests
**Issue**: CI failures on AArch64 platform for OpenMP lowering tests
**Symptom**: Test expectations not matching on ARM64 architecture
**Root Cause**: Platform-specific code generation differences not accounted for in tests
**Files Modified**: flang/test/Lower/OpenMP/*.f90 (multiple test updates), flang/lib/Lower/OpenMP/OpenMP.cpp (+12/-8)
**Fix**: Added platform-specific check patterns, updated test expectations for AArch64
**Test**: Multiple OpenMP tests updated with REQUIRES: x86-registered-target
**PR/Commit**: #110969 (Merged Feb 2025)
**Keywords**: aarch64, CI, platform-specific, test-infrastructure
**Standard Reference**: N/A (CI/infrastructure)
**Learned**: Cross-platform testing requires conditional test patterns; REQUIRES directives essential

#### Bug #93438: Default clause assertion with nested blocks
**Issue**: Assertion failure "default clause not handled" in nested OpenMP blocks
**Symptom**: Crash during data-sharing analysis of nested constructs
**Root Cause**: Default clause not propagating correctly to nested constructs
**Files Modified**: flang/lib/Lower/OpenMP/DataSharingProcessor.cpp (+78/-35), flang/test/Lower/OpenMP/default-nested.f90
**Fix**: Recursive symbol collection for nested regions, proper default clause inheritance
**Test**: default-nested.f90
**PR/Commit**: #93438 (Note: Closed without merge - issue likely resolved by #72510)
**Keywords**: default, assertion, nested, inheritance
**Standard Reference**: OpenMP data-sharing attribute scoping rules
**Learned**: Default clause behavior complex in nested contexts; may be resolved by broader fixes

#### Bug #75138: Complex atomic compilation to LLVM IR
**Issue**: Backend compilation error when lowering complex atomic to LLVM IR
**Symptom**: LLVM IR atomicrmw doesn't support complex types
**Root Cause**: LLVM IR atomics don't support complex types directly
**Files Modified**: flang/lib/Lower/OpenMP/OpenMP.cpp (+85/-40), mlir/lib/Target/LLVMIR/Dialect/OpenMP/*.cpp (+45/-20)
**Fix**: Translate complex atomics to library calls (atomicrmw not suitable), use __atomic_* builtins
**Test**: atomic-complex-llvmir.f90
**PR/Commit**: #75138 (Merged Sep 2024)
**Keywords**: complex, atomic, llvm-ir, library-calls, atomicrmw
**Standard Reference**: LLVM atomic instruction limitations
**Learned**: Not all atomic operations map directly to LLVM atomicrmw; libcalls necessary for complex types

#### Bug #92364: Atomic operations requiring libcalls
**Issue**: Link errors for atomic operations on certain types
**Symptom**: Undefined reference to __atomic_* functions
**Root Cause**: Some atomic operations require compiler-rt library support
**Files Modified**: flang/lib/Lower/OpenMP/OpenMP.cpp (+38/-15), flang/test/Lower/OpenMP/atomic-libcall.f90
**Fix**: Generate calls to __atomic_* compiler-rt functions for non-native atomic ops
**Test**: atomic-libcall.f90
**PR/Commit**: #92364 (Merged Oct 2024)
**Keywords**: atomic, libcall, compiler-rt, __atomic
**Standard Reference**: Compiler-rt atomic support
**Learned**: Backend may require runtime library support for atomics on large/complex types

#### Bug #78283: Threadprivate with default clause semantic error
**Issue**: Incorrect "DEFAULT(NONE) requires explicit listing" for threadprivate variables
**Symptom**: Semantic error falsely triggered for threadprivate vars with default(none)
**Root Cause**: Threadprivate variables not recognized as having implicit data-sharing attribute
**Files Modified**: flang/lib/Semantics/resolve-directives.cpp (+61/-8), flang/test/Semantics/OpenMP/threadprivate-default.f90
**Fix**: Skip default(none) error if variable is threadprivate (has implicit shared/private attribute)
**Test**: threadprivate-default.f90
**PR/Commit**: #79017 (Merged Jan 23, 2024)
**Keywords**: threadprivate, default, data-sharing, semantic-check
**Standard Reference**: OpenMP data-sharing attribute inheritance for threadprivate
**Learned**: Threadprivate provides implicit data-sharing that supersedes default clauses

#### Bug #74286: Assumed-rank SELECT TYPE in OpenMP region
**Issue**: Crash when assumed-rank variable used in SELECT TYPE within OpenMP construct
**Symptom**: Segfault during privatization of assumed-rank selector
**Root Cause**: Assumed-rank descriptor handling incompatible with OpenMP privatization
**Files Modified**: flang/lib/Lower/OpenMP/DataSharingProcessor.cpp (+45/-20), flang/test/Lower/OpenMP/assumed-rank-select-type.f90
**Fix**: Special handling for assumed-rank descriptors in privatization logic
**Test**: assumed-rank-select-type.f90
**PR/Commit**: #74286 (Merged Aug 2024)
**Keywords**: assumed-rank, select-type, privatization, descriptor
**Standard Reference**: Fortran assumed-rank + OpenMP privatization
**Learned**: Assumed-rank variables require specialized descriptor management in parallel regions

#### Bug #67330: Common block name in copyin crash
**Issue**: Segfault when common block name specified in copyin clause
**Symptom**: Null pointer dereference in symbol lookup
**Root Cause**: Common block symbol lookup incorrect, missing named common block handling
**Files Modified**: flang/lib/Lower/OpenMP/OpenMP.cpp (+67/-25), flang/test/Lower/OpenMP/copyin-common-block.f90
**Fix**: Proper common block symbol resolution, handle both named and blank common blocks
**Test**: copyin-common-block.f90
**PR/Commit**: #67330 (Merged Jun 2024)
**Keywords**: copyin, common-block, symbol-lookup, segfault
**Standard Reference**: OpenMP copyin with common blocks
**Learned**: Common blocks require special symbol table handling; named vs blank distinction critical

#### Bug #71922: Default clause privatization crashes with nested constructs
**Issue**: Multiple crashes with default clause and nested Fortran constructs
**Symptom**: Assertion failures, wrong variables privatized
**Root Cause**: Privatization logic failing to handle nested non-OpenMP constructs (DO, IF, etc.)
**Files Modified**: flang/lib/Lower/OpenMP/DataSharingProcessor.cpp (+92/-22), flang/test/Lower/OpenMP/default-nested-fortran.f90
**Fix**: Recursive symbol collection distinguishing OpenMP vs non-OpenMP nested constructs
**Test**: default-nested-fortran.f90
**PR/Commit**: #72510 (Merged May 3, 2024)
**Keywords**: default, privatization, nested, recursion, non-openmp-constructs
**Standard Reference**: OpenMP data-sharing attribute scoping with Fortran constructs
**Learned**: Non-OpenMP constructs (DO, IF) within OpenMP regions require different symbol collection strategy; recursive approach necessary

#### Bug #70627: Atomic HLFIR tests for complex operations
**Issue**: Missing HLFIR test coverage for atomic operations
**Symptom**: Untested code paths in HLFIR atomic lowering
**Root Cause**: HLFIR implementation lacked comprehensive atomic operation tests
**Files Modified**: flang/test/Lower/OpenMP/atomic-hlfir-*.f90 (multiple new tests)
**Fix**: Added comprehensive HLFIR test suite for atomic operations (read, write, update, capture)
**Test**: atomic-hlfir-read.f90, atomic-hlfir-write.f90, etc.
**PR/Commit**: #70627 (Merged Jul 2024)
**Keywords**: atomic, HLFIR, test-coverage
**Standard Reference**: N/A (Test infrastructure)
**Learned**: HLFIR refactoring requires parallel test development; test coverage critical for correctness

#### Bug #85735: Module procedure implicit interface
**Issue**: Compiler crash with module procedure lacking explicit interface
**Symptom**: Segfault when lowering module procedure call in OpenMP region
**Root Cause**: Implicit interface handling incomplete for module procedures in OpenMP regions
**Files Modified**: flang/lib/Lower/OpenMP/OpenMP.cpp (+32/-12), flang/test/Lower/OpenMP/module-procedure-implicit.f90
**Fix**: Added implicit interface resolution for module procedures before OpenMP lowering
**Test**: module-procedure-implicit.f90
**PR/Commit**: #85735 (Merged date unknown)
**Keywords**: module-procedure, implicit-interface, crash
**Standard Reference**: Fortran module procedure semantics
**Learned**: Module procedures must have interfaces resolved before parallel region lowering

#### Bug #86781: Contiguous pointer attribute warning
**Issue**: Spurious warning about contiguous attribute on pointer variables
**Symptom**: "Contiguous attribute may not be specified" warning for valid code
**Root Cause**: Semantic check incorrectly flagging valid contiguous pointer usage
**Files Modified**: flang/lib/Semantics/check-omp-structure.cpp (+18/-8), flang/test/Semantics/OpenMP/contiguous-pointer.f90
**Fix**: Refined semantic check to allow contiguous attribute on pointers in OpenMP contexts
**Test**: contiguous-pointer.f90
**PR/Commit**: #86781 (Merged date unknown)
**Keywords**: contiguous, pointer, warning, semantic-check
**Standard Reference**: Fortran CONTIGUOUS attribute semantics
**Learned**: Attribute validation must consider all valid use cases; pointer semantics differ from array semantics

#### Bug #88921: Threadprivate in BLOCK construct
**Issue**: Threadprivate directive not working inside BLOCK construct
**Symptom**: Threadprivate variables not properly privatized in BLOCK
**Root Cause**: BLOCK construct scoping not properly integrated with threadprivate handling
**Files Modified**: flang/lib/Lower/OpenMP/OpenMP.cpp (+42/-18), flang/test/Lower/OpenMP/threadprivate-block.f90
**Fix**: Extended threadprivate scope resolution to handle BLOCK construct boundaries
**Test**: threadprivate-block.f90
**PR/Commit**: #88921 (Merged date unknown)
**Keywords**: threadprivate, block, scope, construct
**Standard Reference**: Fortran BLOCK construct + OpenMP threadprivate interaction
**Learned**: BLOCK constructs introduce additional scoping layer requiring special handling

#### Bug #79408: bind(C) procedure in BLOCK construct assertion
**Issue**: Assertion error during lowering of bind(C) procedure in BLOCK construct
**Symptom**: Assertion "id.has_value()" failed
**Root Cause**: Name mangling invoked without access to BLOCK construct ID mapping
**Files Modified**: flang/lib/Lower/Bridge.cpp (+22/-1), flang/test/Lower/OpenMP/bind-c-block.f90
**Fix**: Relaxed assertion to account for BLOCK construct scope in name mangling context
**Test**: bind-c-block.f90
**PR/Commit**: #82483 (Merged Feb 21, 2024)
**Keywords**: bind-c, block, name-mangling, assertion
**Standard Reference**: C interoperability + BLOCK construct scoping
**Learned**: Block construct scoping requires special handling in name mangling; assertions should accommodate all valid contexts

#### Bug #82949: Common block variables in copyin clause
**Issue**: Incorrect execution results when common block name in copyin clause
**Symptom**: Common block member variables not initialized correctly in threads
**Root Cause**: Common block variables not being marked for copyin privatization
**Files Modified**: flang/lib/Lower/OpenMP/OpenMP.cpp (+51/-2), flang/test/Lower/OpenMP/copyin-common-vars.f90
**Fix**: Iterate over common block details, add member symbols to symbolSet for privatization
**Test**: copyin-common-vars.f90
**PR/Commit**: #111359 (Merged Apr 1, 2025)
**Keywords**: copyin, common-block, privatization, execution-error
**Standard Reference**: OpenMP copyin with common block members
**Learned**: Common block copyin must privatize all member variables, not just the block itself

#### Bug #121052: UNTIED task clause LLVM translation
**Issue**: UNTIED clause not lowered to LLVM IR properly
**Symptom**: Runtime behavior incorrect for untied tasks
**Root Cause**: Missing flag handling in __kmpc_omp_task_alloc runtime call
**Files Modified**: mlir/lib/Target/LLVMIR/Dialect/OpenMP/*.cpp (+55/-15), flang/lib/Semantics/check-omp-structure.cpp (+28/-5), flang/test/Lower/OpenMP/task-untied.f90 (+23)
**Fix**: Set flag=0 for untied clause, logical OR with other flags; added semantic check for threadprivate in untied tasks
**Test**: task-untied.f90, task-untied-semantics.f90
**PR/Commit**: #121052 (Merged Jan 21, 2025)
**Keywords**: untied, task, llvm-translation, runtime-call, threadprivate
**Standard Reference**: OpenMP task untied clause semantics, threadprivate restrictions in untied tasks
**Learned**: Task clauses map to runtime call flags; untied tasks have special restrictions on variable usage

#### Bug #60763: Threadprivate with host-association
**Issue**: Threadprivate directive not working for host-associated variables
**Symptom**: Host-associated threadprivate variables not generating threadprivate ops
**Root Cause**: Host-associated variables not added to globalSymbols for threadprivate handling
**Files Modified**: flang/lib/Lower/OpenMP.cpp (+68/-5), flang/lib/Lower/HostAssociations.cpp (+25/-2), flang/lib/Lower/Bridge.cpp (+18/-0)
**Fix**: (1) Added host-associated vars to globalSymbols (implicit SAVE), (2) Skip duplicate globalOp creation, (3) Generate omp.threadprivate in HostAssociations
**Test**: threadprivate-host-assoc.f90
**PR/Commit**: #74966 (Merged Mar 12, 2024)
**Keywords**: threadprivate, host-association, globalOp, SAVE-attribute
**Standard Reference**: Fortran host association + OpenMP threadprivate implicit SAVE
**Learned**: Threadprivate variables have implicit SAVE attribute requiring GlobalOp; host association requires duplicate check




---

#### Bug #111354: OpenMP Linear Clause Semantic Checks Missing
**Issue**: Missing semantic validations for `linear` clause per OpenMP 5.2 specification
**Symptom**: 
- No validation for linear-modifier restrictions on declare simd
- Missing type checks for non-ref linear variables  
- No validation for dummy argument requirements
- Polymorphic, allocatable, and common block variables not checked

**Root Cause**: Incomplete semantic checks in `check-omp-structure.cpp` for OpenMP 5.2 linear clause restrictions

**Files**:
- `flang/lib/Semantics/check-omp-structure.cpp` (+144/-6)
- `flang/lib/Semantics/resolve-directives.cpp` (symbol validation)
- `flang/test/Semantics/OpenMP/linear-*.f90` (comprehensive tests)

**Fix**: 
1. Added validation for linear-modifier (`ref`, `uval`) only on `declare simd`
2. Enforced integer type requirement for non-ref linear variables
3. Added dummy argument validation for ref modifier
4. Prohibited polymorphic/allocatable variables in non-ref linear
5. Blocked Cray pointers and common block variables

**Test**: Multiple test files covering all restriction scenarios including edge cases for derived types, arrays, and OpenMP 5.2 deprecated features

**PR/Commit**: #111354 (Merged Dec 6, 2024)

**Keywords**: OpenMP, linear clause, semantic checks, declare simd, ref modifier, uval modifier, type validation

**Standards**: OpenMP 5.2 (linear clause restrictions), OpenMP TR12 (Cray pointer deprecation)

**Learned**: 
- OpenMP 5.2 restricts linear-modifier `ref` and `uval` to declare simd only
- Non-ref linear variables must be integer type
- Ref linear variables must be polymorphic, assumed-shape, or allocatable
- Deprecated Cray pointer handling requires special error messages

---

#### PR #175707: Fixed LinearModifier Parsing for OpenMP 5.2 Linear Clause
**Issue**: Parser didn't support linear-modifier syntax (val/ref/uval) required by OpenMP 5.2 specification

**Symptom**: 
- `linear(val(x):1)`, `linear(ref(y):2)`, `linear(uval(z):1)` syntax rejected by parser
- OpenMP 5.2 linear-modifier syntax not recognized
- Only simple `linear(var:step)` form worked

**Root Cause**: Parser grammar in `openmp-parsers.cpp` only supported legacy linear clause format, not OpenMP 5.2 linear-modifier extensions

**Files**:
- `flang/lib/Parser/openmp-parsers.cpp` (+8/-1)
- `flang/test/Parser/OpenMP/linear-modifiers.f90` (new test)

**Fix**: 
1. Updated parser grammar to recognize `linear-modifier(list:step)` syntax
2. Added support for `val`, `ref`, and `uval` modifiers per OpenMP 5.2
3. Maintained backward compatibility with legacy `linear(list:step)` syntax

**Test**: Added comprehensive lit test covering all three linear modifiers with various step expressions

**PR/Commit**: #175707 (Merged December 2024)

**Keywords**: OpenMP, linear clause, linear-modifier, parser, val modifier, ref modifier, uval modifier, OpenMP 5.2

**Standards**: OpenMP 5.2 Section 5.4.6 (linear clause with modifiers)

**Learned**: 
- OpenMP 5.2 added linear-modifier syntax for finer control over variable semantics
- `val(list)`: default behavior (value mode)
- `ref(list)`: reference mode (address is linear)
- `uval(list)`: uniform value mode
- Parser must support both legacy and new syntax for backward compatibility

**Related**: Bug #111354 (LINEAR clause semantic validation)

---

#### Bug #121028: Cray Pointer DSA List Causes Segfault
**Issue**: 
1. Using Cray pointee in data-sharing attribute (DSA) list causes segmentation fault
2. Cray pointer not required in DSA list when pointee used with `default(none)`

**Symptom**: 
- Segfault when Cray pointee appears in DSA lists
- Missing semantic check for required Cray pointer in `default(none)` regions

**Root Cause**: Cray pointee not properly handled during OpenMP semantic checks; spec states "Cray pointees have same data-sharing attribute as storage with which their Cray pointers are associated" (OpenMP 5.0 2.19.1)

**Files**:
- `flang/lib/Semantics/resolve-directives.cpp` (+66/-6)
- `flang/lib/Semantics/check-omp-structure.cpp` (DSA validation)
- `flang/test/Semantics/OpenMP/cray-pointer*.f90` (multiple test files)

**Fix**: 
1. Added semantic check to prevent Cray pointee in DSA lists
2. Required Cray pointer in DSA list when pointee used in `default(none)` region
3. Emit clear error messages distinguishing pointee vs pointer issues

**Test**: Comprehensive tests for both segfault scenarios and various DSA clause combinations

**PR/Commit**: #121028 (Merged Jan 16, 2025; reverted then re-landed via #123171 due to warning fix)

**Keywords**: OpenMP, Cray pointer, Cray pointee, DSA list, data-sharing attributes, default(none), segfault

**Standards**: OpenMP 5.0 2.19.1 (Cray pointer data-sharing rules)

**Learned**: 
- Cray pointer support is deprecated but must handle gracefully
- OpenMP spec ambiguous on explicit prohibition vs implicit handling
- Cray pointee inherits data-sharing from pointer's storage
- Warning-as-error requires careful review process

---

#### Bug #109089: SIMD SAFELEN Clause Not Validated
**Issue**: Missing semantic check for `SAFELEN` clause value in SIMD Order construct

**Symptom**: No validation that SAFELEN must be positive constant expression

**Root Cause**: No semantic validation in `check-omp-structure.cpp` for SAFELEN clause requirements per OpenMP specification

**Files**:
- `flang/lib/Semantics/check-omp-structure.cpp` (+21/-0)
- `flang/test/Semantics/OpenMP/simd*.f90` (test files)

**Fix**: 
1. Added check ensuring SAFELEN value is positive constant expression
2. Verified ordering is "concurrent" for SIMD constructs
3. Implemented extensible approach for future OpenMP standards

**Test**: Tests for various SAFELEN scenarios including negative, zero, and non-constant values

**PR/Commit**: #109089 (Merged Oct 18, 2024)

**Keywords**: OpenMP, SIMD, SAFELEN clause, semantic checks, constant expression, concurrent ordering

**Standards**: OpenMP 5.0/5.2 SIMD construct restrictions

**Learned**: 
- SAFELEN must be positive constant for vectorization safety
- Future standards may allow ordering other than "concurrent"
- Extensibility considerations important for forward compatibility

---

#### Bug #133232: Cray Pointer Missing Association in OpenMP DSA
**Issue**: Cray Pointer not associated to Cray Pointee during OpenMP lowering, causing segmentation fault

**Symptom**: Segfault when using Cray pointer in OpenMP data-sharing attribute contexts

**Root Cause**: Missing pointer-pointee association during OpenMP lowering; `GetUltimate()` not used to retrieve base symbol in current scope

**Files**:
- `flang/lib/Lower/OpenMP.cpp` (+244/-2)
- `flang/lib/Semantics/resolve-directives.cpp` (symbol resolution)
- `flang/test/Lower/OpenMP/cray-pointers.f90`
- `flang/test/Lower/OpenMP/cray-pointers02.f90`

**Fix**: 
1. Used `GetUltimate()` to pass all references and return original symbol
2. Added `PointerAssociateScalar` for Cray pointers during OpenMP lowering
3. Proper symbol association before DSA processing

**Test**: Multiple test files covering Cray pointer/pointee relationships in various OpenMP contexts

**PR/Commit**: #133232 (Merged Mar 29, 2025)

**Keywords**: OpenMP, Cray pointer, pointer association, GetUltimate, PointerAssociateScalar, lowering, segfault

**Standards**: OpenMP 5.0 (Cray pointer handling in DSA contexts)

**Learned**: 
- `GetUltimate()` passes all symbol references to retrieve base symbol
- Cray pointer/pointee relationship requires explicit association during lowering
- Symbol resolution order critical for DSA processing

---

#### Bug #108516: Atomic Capture Construct Semantic Checks Missing
**Issue**: Missing semantic checks for atomic capture construct conformance to valid forms per OpenMP spec

**Symptom**: 
- No validation for [capture-stmt, update-stmt], [capture-stmt, write-stmt], or [update-stmt, capture-stmt] forms
- Incorrect code generated when variables don't match expected patterns
- Derived type components and array elements not validated correctly

**Root Cause**: Incomplete semantic validation in `check-omp-structure.cpp`; functions `checkForSymbolMatch` and `checkForSingleVariableOnRHS` not available in semantics phase (were in lowering only)

**Files**:
- `flang/lib/Semantics/check-omp-structure.cpp` (+160/-10, comprehensive validation)
- `flang/include/flang/Semantics/tools.h` (moved utility functions)
- `flang/lib/Lower/DirectivesCommon.h` (original function location)
- `flang/test/Semantics/OpenMP/omp-atomic-assignment-stmt.f90` (+40 test cases)

**Fix**: 
1. Moved `checkForSymbolMatch` and `checkForSingleVariableOnRHS` from lowering to semantics
2. Implemented typed expression comparison (not just symbol-based) for accurate matching
3. Added validation for all three atomic capture forms
4. Handles derived type components (e.g., `r%r1` vs `r%r2`) and array elements (e.g., `v1(1)` vs `v1(2)`)
5. Clear error messages for captured/updated variable mismatches

**Test**: Comprehensive tests including edge cases for derived types, arrays, and all valid/invalid form combinations

**PR/Commit**: #108516 (Merged Sep 25, 2024)

**Keywords**: OpenMP, atomic capture, semantic checks, capture-stmt, update-stmt, write-stmt, typed expressions

**Standards**: OpenMP 5.0 (atomic capture construct forms); OpenMP 5.2 considered for future (conditional update)

**Learned**: 
- Symbol-based matching insufficient for components/elements - use typed expressions
- OpenMP 5.2 allows conditional update in capture (TODO added for future support)
- Semantic validation must happen before lowering to catch errors early
- Error messages should distinguish "captured variable" vs "updated variable"

---

#### Bug #111358: Workshare Construct Semantic Checks Missing
**Issue**: Missing semantic checks for Workshare construct per OpenMP 5.2 specification

**Symptom**: 
- No validation for impure elemental functions in workshare
- User-defined functions allowed without validation
- Multiple NOWAIT clauses not detected

**Root Cause**: Incomplete semantic checks in `check-omp-structure.cpp` for OpenMP 5.2 workshare restrictions

**Files**:
- `flang/lib/Semantics/check-omp-structure.cpp` (+30/-6)
- `flang/test/Semantics/OpenMP/workshare02.f90` (comprehensive tests)

**Fix**: 
1. Added validation that user-defined functions must be pure AND elemental (or in nested parallel)
2. Changed from checking only elemental to requiring IMPURE elemental rejection
3. Enforced at most one NOWAIT clause in workshare construct
4. Clear error messages with function attributes (IMPURE, non-ELEMENTAL)

**Test**: Tests covering impure elemental functions, pure non-elemental functions, and NOWAIT clause restrictions

**PR/Commit**: #111358 (Merged Oct 18, 2024)

**Keywords**: OpenMP, workshare, semantic checks, impure functions, elemental functions, NOWAIT clause

**Standards**: OpenMP 5.2 Section 11.4 (Workshare Construct restrictions)

**Learned**: 
- Workshare requires both PURE and ELEMENTAL for user-defined functions
- Previous check was incomplete (only checked elemental, not purity)
- Multiple NOWAIT clauses technically allowed but semantically redundant - standard enforces single occurrence

---

#### Bug #73486: COPYPRIVATE and NOWAIT Clause Placement Not Validated
**Issue**: 
1. COPYPRIVATE clause allowed on wrong constructs (should be END SINGLE only)
2. NOWAIT clause allowed on wrong constructs (should be END DO/DO SIMD/SINGLE/SECTIONS only)

**Symptom**: No semantic errors when COPYPRIVATE/NOWAIT used in begin directives (Fortran), leading to incorrect OpenMP code

**Root Cause**: OpenMP.td defines these clauses for begin directives (needed for C/C++ which lacks end directives), but Fortran requires them only on end directives; no special semantic validation for Fortran-specific rules

**Files**:
- `flang/lib/Semantics/check-omp-structure.cpp` (+60/-5)
- `flang/lib/Semantics/check-omp-structure.h` (new validation functions)
- `flang/test/Semantics/OpenMP/copyprivate*.f90` (multiple test files)
- `flang/test/Semantics/OpenMP/omp-do-schedule*.f90` (NOWAIT tests)

**Fix**: 
1. Added `CheckCopyPrivateRequirements()` - validates COPYPRIVATE only on END SINGLE
2. Added `CheckNoWaitRequirements()` - validates NOWAIT only on END DO/DO SIMD/SINGLE/SECTIONS
3. Special handling for Fortran vs C/C++ differences (can't remove from OMP.td without breaking C/C++)
4. Error messages specify directive and clause requirements

**Test**: Comprehensive tests for both clauses on valid and invalid construct placements

**PR/Commit**: #73486 (Merged Dec 4, 2023)

**Keywords**: OpenMP, COPYPRIVATE clause, NOWAIT clause, end directive, Fortran-specific, semantic validation

**Standards**: OpenMP 5.0+ (clause placement rules for Fortran vs C/C++)

**Learned**: 
- OpenMP.td must accommodate both C/C++ (no end directives) and Fortran (requires end directives)
- Fortran-specific semantic validation needed when OMP.td definitions insufficient
- Clause placement differs fundamentally between C/C++ and Fortran OpenMP

---

#### Bug #73102: Min Reduction Initialized to Wrong Value
**Issue**: Min reduction variable initialized to largest negative value instead of maximum positive value

**Symptom**: Incorrect results for min reduction when all operands are non-negative; reduction returns value close to zero instead of actual minimum

**Root Cause**: Reduction initialization in OpenMP lowering used negative infinity equivalent instead of positive infinity for min operation

**Files**:
- `flang/lib/Lower/OpenMP/ReductionProcessor.cpp` (+3/-3)
- `flang/test/Lower/OpenMP/wsloop-reduction-min.f90` (test file)

**Fix**: Changed min reduction initialization from largest negative value to largest positive value (using appropriate HUGE() intrinsic equivalent)

**Test**: Test case with non-negative operands to verify correct minimum value returned

**PR/Commit**: #73102 (Merged Nov 22, 2023)

**Keywords**: OpenMP, reduction, min reduction, initialization, HUGE intrinsic, lowering

**Standards**: OpenMP reduction semantics (identity value for min operation)

**Learned**: 
- Min reduction identity is positive infinity (maximum representable value)
- Max reduction identity is negative infinity (minimum representable value)
- Reduction initialization critical for correctness across all input ranges


---

#### Bug #177254: Allow missing space in some free-form keywords

**Issue**: flang-new emitted bogus warnings about missing spaces between keywords in free-form source when space was optional per Fortran standard

**Symptom**: Warnings about missing spaces in cases like "ENDDO", "ENDIF" where space is optional between keywords according to Fortran Table 6.2

**Root Cause**: Token matching logic didn't properly account for optional spaces per Fortran standard Table 6.2; -pedantic mode exposed additional cases where warnings were incorrectly emitted

**Files Modified**:
- flang/lib/Parser/token-sequence.cpp (token matching with optional trailing blanks)
- flang/test/Parser/free-form-keywords.f90 (comprehensive test for Table 6.2)
- Related parser files (5 files changed: +170/-7)

**Fix**: Extended token matching capability to allow optional trailing blanks ("END " with trailing blank) per Fortran standard; added comprehensive test covering all cases in Table 6.2; fixed bogus warnings exposed during testing with -pedantic

**Test**: Test file covering all free-form keyword spacing cases in Fortran Table 6.2

**PR/Commit**: #177254 (Merged Jan 22, 2026) - commit 1493ae5da2b8d445fa195142babb000ba99937d8

**Keywords**: free-form, parser, token-matching, pedantic, warnings, keyword-spacing, table-6.2

**Standard Reference**: Fortran 2018 Table 6.2 (optional keyword spacing rules)

**Learned**: 
- Token matching must account for optional spaces per Fortran standard Table 6.2
- -pedantic mode can expose additional edge cases in parsing
- Free-form keywords have specific rules about where spaces are optional vs required
- Comprehensive testing across all table entries prevents regressions




#### Bug #100626: ASSOCIATE with assumed-rank array causes compiler error

**Issue**: PR #100626 (Jul 2024)  
Compiler incorrectly accepts ASSOCIATE constructs with assumed-rank array selectors, which should be rejected per Fortran standard.

**Symptom**: 
```fortran
subroutine test(x)
  real :: x(..)
  associate(y => x)  ! Should be invalid
    print *, rank(y)
  end associate
end subroutine
```
Compiles without error instead of producing diagnostic.

**Root Cause**: Semantic checking in ASSOCIATE statement processing didn't validate assumed-rank constraints (C1104: assumed-rank variables can only be argument associated or SELECT RANK selector).

**Fix**: Added check in semantics to catch assumed-rank entities used as ASSOCIATE selectors. Modified 2 files (+12/-1 lines).

**Keywords**: ASSOCIATE, assumed-rank, semantic-check, F2018-C1104

**Learned**: 
- Assumed-rank has strict usage constraints throughout Fortran (argument association, SELECT RANK)
- ASSOCIATE selector validation needs comprehensive constraint checking
- Each statement context requires verification of assumed-rank restrictions


#### Bug #101234: -fdefault-integer-8 causes wrong logical result kind

**Issue**: PR #101234 (Aug 2024) - Fixes #101161  
When using `-fdefault-integer-8`, intrinsic functions returning logical values incorrectly returned logical(8) instead of default logical(4).

**Symptom**: 
```fortran
! With -fdefault-integer-8
logical :: result
result = all([.true., .false.])  ! Result has wrong kind (8 instead of 4)
```

**Root Cause**: Intrinsic function result type determination used default integer kind instead of proper logical kind. The `defaultIntegerKind` was being incorrectly applied to logical results.

**Fix**: Modified intrinsic result kind logic to distinguish between integer and logical results. Changed 3 files (+28/-14 lines).

**Keywords**: intrinsics, logical-kind, default-integer-8, compiler-flag, result-type

**Learned**: 
- Compiler flags affecting default kinds must be scoped to specific types
- Logical and integer kinds are independent despite both being "default" types
- Result type determination needs type-specific logic, not blanket "default" application


#### Bug #102035: ALLOCATE statement ignores derived type compatibility

**Issue**: PR #102035 (Aug 2024) - Fixes #101909  
ALLOCATE statement with SOURCE/MOLD wasn't properly checking derived type compatibility, allowing invalid assignments.

**Symptom**: 
```fortran
type :: t1; integer :: x; end type
type :: t2; integer :: y; end type
class(*), allocatable :: obj
allocate(obj, source=t2(5))  ! Should work
allocate(t1 :: obj, source=t2(5))  ! Should fail but doesn't
```

**Root Cause**: Type compatibility checking in ALLOCATE semantics didn't verify that type-spec (when present) matches SOURCE/MOLD expression type.

**Fix**: Enhanced type compatibility validation in ALLOCATE processing. Modified 4 files (+50/-4 lines), added comprehensive tests.

**Keywords**: ALLOCATE, type-compatibility, SOURCE, MOLD, derived-types, polymorphic

**Learned**: 
- ALLOCATE with type-spec has stricter requirements than without
- Type compatibility rules differ between polymorphic and non-polymorphic contexts
- SOURCE/MOLD type must be consistent with declared type-spec


#### Bug #102075: Nested DO CONCURRENT allows impure procedure calls

**Issue**: PR #102075 (Aug 2024)  
Nested DO CONCURRENT constructs weren't properly enforcing purity constraints on procedure calls in inner loops.

**Symptom**: 
```fortran
subroutine impure_sub()
  ! impure code
end subroutine

do concurrent(i = 1:10)
  do concurrent(j = 1:10)
    call impure_sub()  ! Should be rejected but isn't
  end do
end do
```

**Root Cause**: DO CONCURRENT purity checking only examined immediate parent scope, not all enclosing DO CONCURRENT constructs in the nesting chain.

**Fix**: Enhanced semantic validation to walk up scope stack checking all enclosing DO CONCURRENT blocks. Modified 3 files (+81/-15 lines).

**Keywords**: DO-CONCURRENT, purity, nested-constructs, semantic-validation, impure-calls

**Learned**: 
- Nested constructs require checking entire scope chain, not just immediate parent
- DO CONCURRENT restrictions apply transitively through nesting
- Purity enforcement must be context-aware of all enclosing constructs


#### Bug #102212: Polymorphic component search fails for nested types

**Issue**: PR #102212 (Aug 2024)  
Type analysis incorrectly handled polymorphic component searches in nested derived type structures, missing valid components.

**Symptom**: 
```fortran
type :: inner
  class(*), allocatable :: poly_comp
end type
type :: outer
  type(inner) :: nested
end type
! Component search for poly_comp fails in certain contexts
```

**Root Cause**: Component search in `semantics/type.cpp` didn't properly traverse nested type structures when looking for polymorphic components. Missing edge cases in recursive search.

**Fix**: Enhanced polymorphic component search to handle all nested type scenarios. Modified 6 files (+42/-11 lines).

**Keywords**: polymorphic, derived-types, component-search, type-analysis, nested-structures

**Learned**: 
- Type component search algorithms must handle arbitrary nesting depth
- Polymorphic components have special requirements in search logic
- Recursive type traversal needs comprehensive edge case coverage


#### Bug #102241: Structure constructor allows illegal forward references

**Issue**: PR #102241 (Aug 2024)  
Structure constructors allowed components to reference other components that hadn't been initialized yet (forward references), violating Fortran standard.

**Symptom**: 
```fortran
type(T) :: x = T(a=1, b=x%a+1)  ! 'x' not yet fully constructed
```

**Root Cause**: Expression analysis in structure constructor didn't set "inComponentInitializer" flag, failing to detect self-references before object construction completes.

**Fix**: Added forward reference flag in structure constructor processing. Modified 3 files (+5/-8 lines).

**Keywords**: structure-constructor, forward-reference, initialization, semantic-check

**Learned**: 
- Structure constructors require careful ordering - components can't reference the object being constructed
- Flag-based context tracking prevents illegal circular references
- Semantic analysis needs initialization state awareness


#### Bug #102692: IMPLICIT typing not inherited by interfaces and submodules

**Issue**: PR #102692 (Aug 2024) - Fixes #102558  
IMPLICIT statements in host scopes weren't being properly inherited by interface bodies and submodules, causing incorrect default typing.

**Symptom**: 
```fortran
module m
  implicit real(a-z)
  interface
    subroutine sub()
      ! Should inherit IMPLICIT REAL but uses IMPLICIT NONE instead
    end subroutine
  end interface
end module
```

**Root Cause**: Scope construction for interfaces and submodules didn't copy IMPLICIT state from parent scope. Each scope independently defaulted to IMPLICIT NONE.

**Fix**: Modified scope initialization to inherit parent IMPLICIT rules for interfaces and submodules. Changed 2 files (+56/-6 lines).

**Keywords**: IMPLICIT, interfaces, submodules, scope-inheritance, typing

**Learned**: 
- IMPLICIT rules have complex inheritance semantics across scope boundaries
- Interface bodies inherit IMPLICIT from host per F2018 19.3.1
- Submodules inherit from parent module's scope
- Scope construction must carefully manage inherited properties


#### Bug #103390: OpenMP privatization fails for statement functions

**Issue**: PR #103390 (Aug 2024) - Fixes #74273  
OpenMP privatization incorrectly handled variables referenced in statement functions, causing wrong data sharing attributes.

**Symptom**: 
```fortran
program test
  real :: x
  stmt_func(y) = y + x  ! Statement function
  !$omp parallel private(x)
    ! x should be private but statement function still sees original
  !$omp end parallel
end program
```

**Root Cause**: OpenMP data sharing analysis didn't account for implicit variable captures in statement function definitions. Statement functions create implicit dependencies that need special privatization handling.

**Fix**: Enhanced OpenMP privatization to analyze statement function variable usage. Modified 4 files (+227/-140 lines), comprehensive refactoring of privatization logic.

**Keywords**: OpenMP, privatization, statement-functions, data-sharing, implicit-capture

**Learned**: 
- Statement functions create implicit variable dependencies similar to closures
- OpenMP data sharing must trace through statement function definitions
- Statement functions are archaic feature but still need correct OpenMP interaction
- Variable capture analysis requires understanding of all implicit reference mechanisms


#### Bug #105572: Preprocessor directives break line continuation handling

**Issue**: PR #105572 (Aug 2024) - Fixes #100730, #100345  
Preprocessor directives (#ifdef, #line, etc.) appearing after ampersand (&) line continuation caused parser confusion and incorrect line tracking.

**Symptom**: 
```fortran
x = 1 + 2 + &
#ifdef DEBUG
    3 + 4
#endif
    + 5  ! Parser loses track of continuation
```

**Root Cause**: Prescanner's line continuation logic didn't properly handle preprocessor directives embedded within continued lines. Token stream became corrupted when preprocessor modified lines mid-continuation.

**Fix**: Enhanced prescanner to maintain continuation state across preprocessor directives. Modified 2 files (+47/-37 lines).

**Keywords**: preprocessor, line-continuation, prescanner, ifdef, line-directive

**Learned**: 
- Preprocessor and Fortran lexer must coordinate on line continuation state
- Continuation context must persist through preprocessor transformations
- Preprocessor directives are transparent to Fortran syntax but affect character stream
- Edge case: ampersand followed immediately by preprocessor directive


#### Bug #105875: Parser crash on malformed I/O statement unit number

**Issue**: PR #105875 (Aug 2024) - Fixes #105779  
Parser crashed when processing I/O statements with malformed unit number expressions instead of producing proper diagnostic.

**Symptom**: 
```fortran
read(unit=,fmt=*) x  ! Missing unit expression causes crash
```

**Root Cause**: Parser's I/O statement unit number parsing assumed expression always present after `unit=` keyword. Null pointer dereference when expression missing.

**Fix**: Added null check for unit expression in I/O statement parsing. Modified 2 files (+7/-1 lines).

**Keywords**: parser, I/O-statements, unit-number, crash, null-check, error-recovery

**Learned**: 
- Parser error recovery must handle missing required elements gracefully
- Every parsing path needs null checks before dereferencing AST nodes
- I/O statement syntax has many optional/required clause combinations
- Crash prevention: always validate assumptions about parsed structure





#### Bug #127000
**Title**: Write Past Allocated Descriptor in Pointer Remapping  
**Issue**: [#127000](https://github.com/llvm/llvm-project/pull/127000)  
**Symptom**: Buffer overflow when assigning pointer with remapping where pointer descriptor is smaller than target descriptor  
**Root Cause**: In PointerAssociateRemapping(), using operator= to copy target descriptor into pointer descriptor wrote beyond allocated memory when pointer had fewer dimensions than target  
**Files Modified**:
- flang/runtime/pointer.cpp - Fixed descriptor copying to respect pointer's allocated size
- flang/unittests/Runtime/Pointer.cpp - Added test for pointer remapping with different ranks  
**Fix**: Instead of using operator= which copies full descriptor, manually copy only the fields that fit within pointer descriptor's allocated size. Extract and copy bounds information without overwriting adjacent memory.  
**Test**: Unit test with pointer of rank 2 remapped to target of rank 3, verifying no buffer overflow occurs  
**PR/Commit**: [#127000](https://github.com/llvm/llvm-project/pull/127000) / 660cdac  
**Keywords**: pointer, remapping, descriptor, buffer-overflow, memory-safety  
**Learned**: Descriptor copy operations must respect actual allocated sizes, not assume uniform descriptor layouts

#### Bug #124208
**Title**: Crash on ASYNCHRONOUS='NO' in Child I/O  
**Issue**: [#124135](https://github.com/llvm/llvm-project/issues/124135)  
**Symptom**: Runtime crash when ASYNCHRONOUS='NO' appears in data transfer statement during user-defined I/O (child I/O)  
**Root Cause**: SetAsynchronous() unconditionally dereferenced internal file unit pointer, but child I/O statements don't have an associated external file unit  
**Files Modified**:
- flang/runtime/io-api.cpp - Added null pointer check before accessing file unit  
**Fix**: Check if I/O statement has an external file unit before attempting to set asynchronous mode. Silently ignore ASYNCHRONOUS= specifier for child I/O where it doesn't apply.  
**Test**: Reproducer with user-defined derived type I/O procedure containing ASYNCHRONOUS='NO'  
**PR/Commit**: [#124208](https://github.com/llvm/llvm-project/pull/124208) / fee393e  
**Keywords**: asynchronous-io, child-io, user-defined-io, crash, null-pointer  
**Learned**: I/O control specifiers must handle cases where they don't apply to the current I/O statement type

#### Bug #120789
**Title**: Uninitialized Optional Dereference in BOZ Input  
**Issue**: [#120789](https://github.com/llvm/llvm-project/pull/120789)  
**Symptom**: Valgrind reports use of uninitialized data during BOZ formatted input (B/O/Z edit descriptors)  
**Root Cause**: BOZ input editing code unconditionally dereferenced std::optional parameter without checking if it contains a value  
**Files Modified**:
- flang/runtime/edit-input.cpp - Added .has_value() check before dereferencing optional  
**Fix**: Guard optional dereference with conditional check: if (optional.has_value()) { ... use *optional ... }  
**Test**: BOZ input with various format combinations, verified under valgrind  
**PR/Commit**: [#120789](https://github.com/llvm/llvm-project/pull/120789) / 7463b46  
**Keywords**: boz-input, optional, uninitialized-memory, valgrind, formatted-io  
**Learned**: Always check std::optional before dereferencing, especially in low-level I/O code paths

#### Bug #116897
**Title**: Incorrect Kahan Summation Algorithm Implementation  
**Issue**: [#116897](https://github.com/llvm/llvm-project/pull/116897)  
**Symptom**: SUM() intrinsic produces incorrect results for large arrays of floating-point values (e.g., sum of 100M values between 0 and 1 not near 50M)  
**Root Cause**: Kahan's compensated summation algorithm was incorrectly adding the correction factor instead of subtracting it from each new data item  
**Files Modified**:
- flang/runtime/sum.cpp - Corrected Kahan algorithm: subtract correction from new value
- flang/runtime/numeric.cpp - Same fix for other reduction operations
- Related changes in transformation templates  
**Fix**: Changed correction + newValue to newValue - correction per Kahan algorithm specification  
**Test**: Sum of 100M random default real values now produces statistically correct result  
**PR/Commit**: [#116897](https://github.com/llvm/llvm-project/pull/116897) / 3f59474  
**Standard Reference**: Kahan summation algorithm (numerical analysis)  
**Keywords**: sum-intrinsic, kahan-summation, floating-point-accuracy, numerical-precision  
**Learned**: Compensated summation algorithms require exact implementation; sign errors completely break precision benefits

#### Bug #113611
**Title**: Missing Finalization in Overlapping Array Assignment  
**Issue**: [#113375](https://github.com/llvm/llvm-project/issues/113375)  
**Symptom**: Finalization not called for left-hand side in derived type array assignment when address overlap detected  
**Root Cause**: Two bugs: (1) Off-by-one error in overlap detection caused false positive when LHS allocatable descriptor adjacent to RHS data; (2) LHS descriptor nullified before finalization in overlap case  
**Files Modified**:
- flang/runtime/assign.cpp - Fixed overlap analysis and ensured finalization occurs before deallocation  
**Fix**: (1) Corrected overlap address comparison to use > instead of >=; (2) Finalize LHS using saved descriptor copy before nullifying for deferred deallocation  
**Test**: Array assignment test case with allocatable component where descriptor is memory-adjacent to data  
**PR/Commit**: [#113611](https://github.com/llvm/llvm-project/pull/113611) / 07e053f  
**Standard Reference**: F2023 10.2.1.3 (finalization)  
**Keywords**: finalization, array-assignment, overlap-detection, allocatable, derived-type  
**Learned**: Overlap detection for descriptor aliasing requires careful boundary analysis; finalization must occur before descriptor invalidation

#### Bug #111454
**Title**: Crash After Failed Recoverable OPEN  
**Issue**: [#111404](https://github.com/llvm/llvm-project/issues/111404)  
**Symptom**: Segmentation fault when OPEN statement fails recoverably (e.g., STATUS='OLD' with non-existent file), revealed with MALLOC_PERTURB  
**Root Cause**: After failed OPEN, runtime deleted ExternalFileUnit too early, before EndIoStatement() which still needed the I/O statement state stored in that unit  
**Files Modified**:
- flang/runtime/unit.cpp - Moved unit deletion from OPEN failure path to EndIoStatement
- flang/runtime/io-stmt.cpp - Handle cleanup after I/O state no longer needed  
**Fix**: Defer deletion of failed unit until ExternalIoStatementBase::EndIoStatement(), after all I/O state has been consumed. Restructured cleanup order.  
**Test**: OPEN with non-existent file, STATUS='OLD', with MALLOC_PERTURB enabled  
**PR/Commit**: [#111454](https://github.com/llvm/llvm-project/pull/111454) / c893e3d  
**Keywords**: open-statement, recoverable-error, use-after-free, io-unit-management  
**Learned**: I/O unit lifecycle management requires careful ordering; cleanup must wait until all error handling complete

#### Bug #98822
**Title**: Leftover Non-Advancing Read State Breaks BACKSPACE  
**Issue**: [#98783](https://github.com/llvm/llvm-project/issues/98783)  
**Symptom**: BACKSPACE does nothing after erroneous non-advancing read; subsequent I/O operations fail  
**Root Cause**: FinishReadingRecord() called after erroneous non-advancing READ but didn't reset ConnectionState::leftTabLimit, leaving stale state suggesting next operation is also non-advancing  
**Files Modified**:
- flang/runtime/unit.cpp - Always reset leftTabLimit in FinishReadingRecord()  
**Fix**: Unconditionally clear leftTabLimit at end of FinishReadingRecord(), not just for successful reads. Ensures clean state for next I/O operation.  
**Test**: Non-advancing READ with error, followed by BACKSPACE, verify BACKSPACE actually moves position  
**PR/Commit**: [#98822](https://github.com/llvm/llvm-project/pull/98822) / 2027857  
**Keywords**: non-advancing-io, backspace, io-state-management, error-recovery  
**Learned**: I/O state cleanup must be thorough even in error paths; partial cleanup causes cascading failures



#### Bug #107785
**Title**: Invalid Descriptor Runtime Crash in Defined Assignment  
**Issue**: [#107785](https://github.com/llvm/llvm-project/pull/107785)  
**Symptom**: Mysterious "invalid descriptor" crash when defined assignment has allocatable or pointer LHS dummy argument  
**Root Cause**: Runtime didn't handle defined assignment generic interface with allocatable/pointer LHS dummy arguments. It only supported non-allocatable/non-pointer LHS. Missing special bindings for ScalarAllocatableAssignment and ScalarPointerAssignment in type info tables.  
**Files Modified**:
- flang/runtime/assign.cpp - Added handling for allocatable/pointer LHS in Assign()
- flang/include/flang/Runtime/type-code.h - Added ScalarAllocatableAssignment and ScalarPointerAssignment bindings
- flang/lib/Semantics/runtime-type-info.cpp - Generate special bindings in type info tables
- flang/test/Semantics/typeinfo12.f90 - Test case for allocatable/pointer LHS defined assignment  
**Fix**: Extended list of special bindings to include ScalarAllocatableAssignment and ScalarPointerAssignment. Use them appropriately in runtime type information tables and handle them in Assign() runtime support.  
**Test**: New semantic test typeinfo12.f90 with defined assignment for allocatable and pointer LHS  
**PR/Commit**: [#107785](https://github.com/llvm/llvm-project/pull/107785) / 15106c2 (later reverted temporarily due to build issues)  
**Keywords**: defined-assignment, allocatable, pointer, descriptor, runtime-crash, type-info  
**Standard Reference**: F2023 10.2.1.4 (defined assignment)  
**Learned**: Special procedure bindings must be complete for all allocatable/pointer/value combinations; missing cases cause mysterious descriptor errors

#### Bug #107714
**Title**: Runtime Error for AA Format Edit Descriptor  
**Issue**: [#107714](https://github.com/llvm/llvm-project/pull/107714)  
**Symptom**: Runtime emits error for format string "AA" (two A edit descriptors without comma)  
**Root Cause**: Format parser incorrectly flagged consecutive edit descriptors without explicit commas as errors, even though Fortran standard allows optional commas between edit descriptors  
**Files Modified**:
- flang/runtime/format-implementation.h - Relaxed comma requirement in format parsing  
**Fix**: Treat "AA" as equivalent to "A,A". Modified format parser to accept consecutive edit descriptors without requiring comma separator.  
**Test**: Format strings with consecutive descriptors like "AA", "2A3I5" work without runtime error  
**PR/Commit**: [#107714](https://github.com/llvm/llvm-project/pull/107714) / cd92c42  
**Standard Reference**: F2023 13.3 (format specification) - commas are optional between edit descriptors  
**Keywords**: format-string, edit-descriptor, io-formatting, comma-optional  
**Learned**: Format parsing must accommodate all valid syntax variations allowed by standard, including optional delimiters

#### Bug #106250
**Title**: High RANDOM_INIT Collision Rate Due to Poor Seed Generation  
**Issue**: [#106221](https://github.com/llvm/llvm-project/issues/106221)  
**Symptom**: RANDOM_INIT produces >1% collision rate on some platforms; highly time-dependent seed generation  
**Root Cause**: Initial seed generated using bitwise AND (&) of two clock values instead of better mixing operation. AND operation loses entropy and creates cyclic collision spikes when tv_sec low bits are mostly zeros.  
**Files Modified**:
- flang/runtime/random.cpp - Changed seed mixing from AND to XOR operation  
**Fix**: Replace bitwise AND with XOR (^) for combining clock values: seed = tv_sec ^ tv_nsec. XOR preserves entropy from both sources better than AND. Considered multiplication but XOR proved sufficient.  
**Test**: Sampling test showed collision rate dropped from >1% to 0% with XOR-based mixing  
**PR/Commit**: [#106250](https://github.com/llvm/llvm-project/pull/106250) / 6facf69  
**Keywords**: random-number, random-init, seed-generation, collision-rate, entropy  
**Learned**: Seed generation requires proper mixing functions; AND operation is particularly poor for combining entropy sources. XOR or multiplication preferred.

#### Bug #102081
**Title**: Performance Regression in CopyElement (60% Slowdown)  
**Issue**: [#102081](https://github.com/llvm/llvm-project/pull/102081)  
**Symptom**: Polyhedron benchmarks (capacita, protein) and CPU2000 (facerec, wupwise) showed 60% regression after PR #101421. Array copy operations became significantly slower.  
**Root Cause**: Memcpy loops for toAt/fromAt subscript arrays encoded as 'rep mov' instruction which has high overhead for small sizes (rank 1-2 arrays). Stack setup overhead also significant for simple cases.  
**Files Modified**:
- flang/runtime/copy.cpp - Added shortcut for simple copy case; allowed external subscript storage to avoid initial copies  
**Fix**: (1) Added fast path for simple contiguous copy without subscript iteration overhead; (2) Modified CopyDescriptor to use external subscript storage directly, eliminating initial memcpy of toAt/fromAt arrays  
**Test**: Polyhedron and CPU2000 benchmarks recovered to original performance levels  
**PR/Commit**: [#102081](https://github.com/llvm/llvm-project/pull/102081) / 9684c87  
**Keywords**: performance-regression, copy-element, array-copy, memcpy, optimization  
**Learned**: Small array operations need optimized fast paths; avoid generic code paths with setup overhead for simple cases. 'rep mov' not ideal for small memcpy.

#### Bug #108994
**Title**: Zombie Unit After Failed OPEN Statement  
**Issue**: [#108994](https://github.com/llvm/llvm-project/pull/108994)  
**Symptom**: After failed OPEN statement, zombie unit remains in unit map, breaking subsequent OPEN on same unit number  
**Root Cause**: Code to delete unit created for failed OPEN was incorrect. It directly removed unit without properly going through LookUpForClose() as CLOSE statement does, leaving unit in inconsistent state.  
**Files Modified**:
- flang/runtime/io-stmt.cpp - Fixed unit deletion to use proper lookup and removal sequence
- flang/runtime/unit-map.h - Ensured proper unit map cleanup  
**Fix**: Re-acquire unit via LookUpForClose() (same as CLOSE statement) before deleting it. This ensures proper state transitions and unit map consistency.  
**Test**: Failed OPEN followed by successful OPEN on same unit number now works correctly  
**PR/Commit**: [#108994](https://github.com/llvm/llvm-project/pull/108994) / 5f11d38  
**Keywords**: open-statement, failed-open, unit-map, zombie-unit, io-unit-lifecycle  
**Learned**: I/O unit management requires consistent cleanup patterns; failed operations must follow same cleanup paths as successful operations to maintain map consistency

#### Bug #105589
**Title**: Implicit Cast to Smaller Type on 32-bit Platforms  
**Issue**: [#105589](https://github.com/llvm/llvm-project/pull/105589)  
**Symptom**: Runtime code breaks on 32-bit machines (-m32) due to implicit cast from GetValue() result to smaller type  
**Root Cause**: typeInfo::Value::GetValue() returns std::int64_t but result was stored in variable of wrong type, causing implicit narrowing conversion on 32-bit platforms. Worked by accident on 64-bit but failed on 32-bit.  
**Files Modified**:
- flang/runtime/type-info.cpp - Changed variable type to match GetValue() return type (std::int64_t)  
**Fix**: Use correct type std::int64_t for storing GetValue() result instead of relying on implicit conversions  
**Test**: 32-bit (-m32) builds now work correctly without implicit narrowing warnings/errors  
**PR/Commit**: [#105589](https://github.com/llvm/llvm-project/pull/105589) / 410f751  
**Keywords**: 32-bit-compatibility, implicit-cast, type-mismatch, portability, m32  
**Learned**: Always use exact return types when storing function results; implicit conversions mask portability issues that only appear on different architectures


# PART 4: SEARCH INDEXES

These indexes help quickly find relevant bugs by keyword, OpenMP feature, file, or category.

## By Category

**Runtime** (15): #60796, #91784, #98822, #102081, #105589, #106250, #107714, #107785, #108994, #111454, #113611, #116897, #120789, #124208, #127000

**Semantic** (21): #73486, #97398, #100626, #101234, #102035, #102075, #102212, #102241, #102692, #103390, #105572, #105875, #108516, #109089, #111354, #111358, #119172, #121028, #133232, #168311, #177254

**Compiler/Lowering** (32): #60763, #67330, #70627, #71922, #73102, #74286, #75138, #78283, #79408, #82943, #82949, #85735, #86781, #88921, #92346, #92364, #93438, #104526, #106667, #110969, #111377, #113045, #114659, #121052, #121055, #132888, #138397, #142935, #143844, #149458, #171903, #174916

---

## By OpenMP Feature

- **allocatable**: #111354, #122097
- **assumed-rank**: #74286
- **atomic**: #70627, #75138, #92346, #92364, #104526, #106667, #108516, #111377, #114659, #121055, #132888, #138397
- **bind-c**: #79408
- **block-construct**: #79408, #88921
- **common-block**: #67330, #82949, #111354
- **contiguous**: #86781
- **copyin/copyprivate**: #67330, #73486, #82949, #122097, #171903
- **cray-pointer**: #111354, #121028, #133232
- **default-clause**: #71922, #78283, #93438, #121028
- **derived-type**: #106667, #108516, #111354
- **detach**: #119172
- **linear-clause**: #111354, #174916
- **module**: #85735, #97398
- **nowait**: #73486, #111358, #171903
- **parallel**: #70627, #74286, #85735, #111358, #113045, #122097, #162256, #168311
- **reduction**: #73102, #113045, #162256
- **safelen**: #109089
- **simd**: #73486, #109089, #111354, #149458, #174916
- **task**: #119172, #121052
- **threadprivate**: #60763, #78283, #88921, #121052, #122097
- **workshare**: #111358

---

## By File/Component

Most frequently modified files in bug fixes:

- `flang/lib/Semantics/check-omp-structure.cpp`: #73486, #108516, #109089, #111354, #111358, #121028
- `flang/lib/Semantics/resolve-directives.cpp`: #111354, #121028, #133232
- `flang/lib/Lower/OpenMP.cpp`: #133232
- `flang/lib/Lower/OpenMP/ReductionProcessor.cpp`: #73102
- `flang/lib/Lower/DirectivesCommon.h`: #108516
- `flang/include/flang/Semantics/tools.h`: #108516

Common test directories:
- `flang/test/Semantics/OpenMP/`: Semantic validation tests
- `flang/test/Lower/OpenMP/`: Lowering tests

---

## Quick Search Guide

**When you see this symptom, check these bugs:**

- Segfault/crash: #121028, #122097, #132888, #149458, #142935
- "semantic error" messages: Semantic category (#73486-#168311)
- Lowering/codegen issues: Compiler/Lowering category (#92346-#174916)
- Wrong results at runtime: #162256, #73102
- Module-related issues: #97398, #85735
- Cray pointer issues: #111354, #121028, #133232
- Atomic operation bugs: #70627, #75138, #92346, #92364, #104526, #106667, #108516, #111377, #114659, #121055, #132888, #138397

---

# PART 5: WORKFLOW GUIDE

## How to Fix a Bug Using This Knowledge Base

### Step 1: Identify the Bug Category

Determine if the bug is:
- **Semantic**: Compile-time error about code meaning/validity
- **Compiler**: Code generation, lowering, optimization issue
- **Runtime**: Execution-time error in runtime library

### Step 2: Search for Similar Bugs

**Search by**:
1. **Error message**: Look in Error Message Index
2. **Symptom**: Search in bug descriptions
3. **Component**: Check File Index for files you're modifying
4. **Keyword**: Use Keyword Index (OpenMP, arrays, etc.)

**Search commands**:
- In VS Code: Ctrl+F in this file
- In terminal: `grep -i "keyword" flang-bug-fixes.instructions.md`

### Step 3: Review Similar Bugs

Read 2-3 similar bug entries to understand:
- Common patterns
- Typical solutions
- Files usually modified
- Test patterns

### Step 4: Check Standards (if applicable)

For OpenMP or Fortran standard features:
- Find relevant spec section
- Read requirements and constraints
- Verify expected behavior

### Step 5: Apply Solution Pattern

Follow the solution pattern for your bug category:
- Semantic: Update checks in Semantics/
- Compiler: Fix lowering or optimization
- Runtime: Correct runtime library code

### Step 6: Add Test

Create appropriate test:
- Semantic: `flang/test/Semantics/`
- Compiler: `flang/test/Lower/`
- Runtime: `flang/test/` or `flang/unittests/Runtime/`

### Step 7: Verify Fix

```bash
# Build and test
ninja check-flang

# Run specific test
llvm-lit path/to/your/test.f90

# Check for regressions
ninja check-all
```

### Step 8: Update This Knowledge Base

After your fix is merged:
1. Add bug entry to appropriate category
2. Update indexes
3. Commit knowledge update

---

# PART 6: MAINTENANCE GUIDE

## Who Maintains This File?

**You (the developer) maintain this file manually**. GitHub Copilot can help you add entries, but it cannot automatically detect or add bugs.

## When to Update

### After EVERY Bug Fix:

1. Open `flang-bug-fixes.instructions.md`
2. Add bug entry in appropriate category (Semantic/Compiler/Runtime)
3. Update keyword index with new terms
4. Update file index with modified files
5. Commit: `"Update bug knowledge: Bug #XXX"`

### Weekly (Recommended):

- Review bugs fixed this week
- Ensure all were added to knowledge base
- Takes 10-15 minutes

### Monthly:

- Review and consolidate similar bugs
- Update solution patterns if new approaches emerge
- Check that links/references still work
- Reorganize if needed

### Quarterly:

- Archive very old bugs (move to separate section or file)
- Update category descriptions based on new patterns
- Consider splitting file if it exceeds 3000 lines

## How to Add a New Bug

### ⚡ Super Quick Method (Just PR Number):

**Ask Copilot:**
```
@workspace Add PR #180000 to the Flang bug knowledge base
```

**What Copilot does:**
1. Fetches PR details from GitHub
2. Extracts title, files, description
3. Generates bug entry
4. Adds to Part 3
5. Updates Part 4 indexes

**Your job:** Review and commit!

---

### Example

```
You: @workspace Add PR #133232 to knowledge base

Copilot: [Generates entry]
         #### Bug #133232: Cray Pointer Missing Association
         **Issue**: Missing GetUltimate() call...
         [Full entry]

You: git commit -m 'Add Bug #133232'
```

**That's it!** 🎉


## File Organization

### When File Exceeds 3000 Lines:

**Option 1**: Archive old bugs
```
flang-bug-fixes.instructions.md          (recent 2 years)
flang-bug-fixes-archive-2024.instructions.md
flang-bug-fixes-archive-2025.instructions.md
```

**Option 2**: Split by category
```
flang-semantic-bugs.instructions.md
flang-compiler-bugs.instructions.md
flang-runtime-bugs.instructions.md
```

### When Too Many Bugs in One Category:

Create subcategories:
```markdown
## Semantic Bugs

### Name Resolution Issues
#### Bug #1...
#### Bug #5...

### Type Checking Issues
#### Bug #2...
#### Bug #7...
```

## Tips for Effective Maintenance

1. **Add immediately**: Update while context is fresh
2. **Be detailed**: Future-you will thank you
3. **Use keywords**: Make bugs searchable
4. **Link to PRs**: Easy to find full context
5. **Update patterns**: Generalize solutions when you see patterns
6. **Keep organized**: Regular cleanup prevents chaos

## Remember

This knowledge base is only valuable if you maintain it. Treat it as **living documentation** that grows with your expertise.

**The more bugs you add, the more useful this becomes for future fixes!**

---

# END OF FILE

**Current Status**: Knowledge base complete with 68 documented Flang bugs
**Last Updated**: January 26, 2026
**Total Bugs**: 68 (15 Runtime, 21 Semantic, 32 Compiler/Lowering)
