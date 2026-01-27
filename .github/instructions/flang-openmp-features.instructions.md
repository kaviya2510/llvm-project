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

# PART 3: IMPLEMENTATION PATTERNS

## Adding a New OpenMP Clause

### Step 1: Parser Support
```fortran
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

# PART 4: QUICK REFERENCE

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

# PART 5: RESOURCES

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
