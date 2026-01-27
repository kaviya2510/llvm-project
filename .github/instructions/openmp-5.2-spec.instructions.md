---
applyTo: **/*
description: OpenMP 5.2 Specification Reference - Directive and Clause Compatibility
---

# OpenMP 5.2 Specification Reference

This file contains extracted information from the OpenMP 5.2 specification to assist with understanding directive/clause compatibility and restrictions.

**Source**: OpenMP API Specification Version 5.2 (November 2021)

---

## CHAPTER 1: OVERVIEW OF THE OPENMP API

### 1.1 Introduction to OpenMP

OpenMP is an API that supports multi-platform shared-memory parallel programming in C/C++ and Fortran. It provides:
- **Compiler directives** for parallelism
- **Runtime library routines** for execution control
- **Environment variables** for runtime behavior control

**Key Philosophy:**
- **Incremental parallelism**: Start with sequential code, add parallelism gradually
- **Performance portability**: Same code runs on different platforms
- **Ease of use**: High-level abstractions for common parallel patterns

---

### 1.2 OpenMP Terminology

#### 1.2.1 Threading Concepts

| Term | Definition | Example |
|------|------------|---------|
| **OpenMP thread** | An execution entity with its own stack and local variables | Each thread in a parallel region |
| **Thread number** | Unique identifier (0 to N-1) within a team | Thread 0 is the primary thread |
| **Primary thread** | Thread with number 0 in a team | Creates team, coordinates work |
| **Worker thread** | Non-primary thread (numbers 1 to N-1) | Executes parallel work |
| **Parent thread** | Thread that encounters `parallel` construct | Creates child threads |
| **Child thread** | Thread created by a `parallel` construct | Part of the new team |
| **Ancestor thread** | Parent thread or parent's ancestor | Hierarchical relationship |
| **Descendent thread** | Child thread or child's descendent | Hierarchical relationship |

#### 1.2.2 Directive Types

| Type | Description | Examples |
|------|-------------|----------|
| **Declarative directive** | Declares properties, no executable code | `threadprivate`, `declare reduction`, `declare simd` |
| **Executable directive** | Affects execution flow | `parallel`, `task`, `for`/`do` |
| **Informational directive** | Conveys properties to compiler | `assume`, attributes |
| **Utility directive** | Compiler interaction, readability | `requires`, `nothing` |
| **Stand-alone directive** | No associated structured block | `barrier`, `taskwait`, `taskyield` |
| **Subsidiary directive** | Part of a construct, not standalone | `master`, `default`, clauses |

#### 1.2.3 Construct Types

| Type | Definition | Example |
|------|------------|---------|
| **Construct** | Directive + end directive + structured block | `!$omp parallel ... !$omp end parallel` |
| **Combined construct** | Shortcut for nesting two constructs | `parallel do` = `parallel` + `do` |
| **Composite construct** | Adds semantics beyond simple nesting | `target teams` (special device semantics) |
| **Constituent construct** | Building block of combined/composite | `parallel` is constituent of `parallel do` |
| **Leaf construct** | Cannot be further decomposed | `simd`, `for`, `parallel` (when alone) |

**Example - Combined Construct:**
```fortran
! Combined: parallel do
!$omp parallel do private(i)
do i = 1, n
  a(i) = b(i) + c(i)
end do
!$omp end parallel do

! Equivalent to nested:
!$omp parallel
!$omp do private(i)
do i = 1, n
  a(i) = b(i) + c(i)
end do
!$omp end do
!$omp end parallel
```

#### 1.2.4 Region Concepts

| Term | Definition | Includes |
|------|------------|----------|
| **Region** | Dynamic extent of a construct during execution | Code in construct + called routines + implementation code |
| **Structured block** | Single entry, single exit block of code | Loop body, if block, procedure body |
| **Parallel region** | Region executed by a team of threads | From `parallel` to end of construct |
| **Active parallel region** | Executed by multiple threads | Team size > 1 |
| **Inactive parallel region** | Executed by single thread | Team size = 1 (serialized) |
| **Sequential part** | Code outside parallel regions | Initial task region, not in parallel/task |

**Example:**
```fortran
! Sequential part
integer :: x = 10

!$omp parallel  ! Parallel region starts
  x = x + omp_get_thread_num()  ! Region includes this code
  call subroutine()  ! Region includes called routines
!$omp end parallel  ! Parallel region ends

! Sequential part continues
```

#### 1.2.5 Task Concepts

| Term | Definition |
|------|------------|
| **Task** | Independent unit of work (code + data environment) |
| **Explicit task** | Created by `task` or `taskloop` construct |
| **Implicit task** | Created by `parallel` construct (one per thread) |
| **Initial task** | Task executing sequential part of program |
| **Included task** | Task executed immediately (not deferred) |
| **Undeferred task** | Task with `if(false)` or inside `final` task |
| **Merged task** | Task that shares data environment with parent |
| **Untied task** | Task that can be suspended and resumed by different threads |
| **Tied task** | Task bound to the thread that started it (default) |
| **Task region** | Code executed for a specific task |
| **Task scheduling point** | Point where scheduler may switch tasks |

**Example - Task Scheduling:**
```fortran
!$omp parallel
!$omp single
  do i = 1, 10
    !$omp task  ! Create explicit tasks
      call work(i)
    !$omp end task
  end do
  !$omp taskwait  ! Task scheduling point - wait for all tasks
!$omp end single
!$omp end parallel
```

#### 1.2.6 Data Environment Concepts

| Term | Definition |
|------|------------|
| **Data environment** | Variables accessible in a region |
| **Shared variable** | Same memory location for all threads |
| **Private variable** | Separate copy per thread/task |
| **Threadprivate variable** | Persistent private copy per thread across regions |
| **Predetermined data-sharing** | Sharing attribute determined by declaration |
| **Explicitly determined** | Sharing specified in clause |
| **Implicitly determined** | Sharing derived from context/default |

#### 1.2.7 Synchronization Concepts

| Term | Definition | Construct |
|------|------------|-----------|
| **Barrier** | All threads wait until all reach the barrier | `barrier`, implicit at end of worksharing |
| **Nowait** | Removes implicit barrier | `nowait` clause |
| **Atomic operation** | Indivisible memory update | `atomic` |
| **Critical section** | Only one thread at a time | `critical` |
| **Lock** | Mutual exclusion mechanism | `omp_set_lock()` |
| **Ordered** | Sequential execution order preserved | `ordered` |
| **Flush** | Memory consistency operation | `flush` |

**Example - Barrier Synchronization:**
```fortran
!$omp parallel
  call phase1()
  !$omp barrier  ! All threads wait here
  call phase2()  ! All threads proceed together
!$omp end parallel  ! Implicit barrier
```

---

### 1.3 Execution Model

**Fork-Join Parallelism:**
1. **Fork**: Primary thread creates team of threads at `parallel` construct
2. **Work**: All threads execute the parallel region
3. **Join**: Threads synchronize at end, only primary continues

**Thread Management:**
- Number of threads controlled by `num_threads` clause or `OMP_NUM_THREADS`
- Thread pool maintained by runtime (threads created once, reused)
- Nested parallelism supported (teams within teams)

**Example:**
```fortran
program main
  ! Sequential execution - initial task
  
  !$omp parallel num_threads(4)  ! FORK: Create 4 threads
    ! Parallel region - all 4 threads execute
    call work()
  !$omp end parallel  ! JOIN: Synchronize, primary continues
  
  ! Sequential execution continues
end program
```

**Task Execution Model:**
- **Task creation**: Encountering thread creates task (code + data)
- **Task scheduling**: Runtime decides when/where to execute tasks
- **Task completion**: Task finishes when structured block completes
- **Task synchronization**: `taskwait`, `taskgroup`, barriers

**Example:**
```fortran
!$omp parallel
!$omp single
  ! Create tasks
  !$omp task
    call process_a()
  !$omp end task
  
  !$omp task
    call process_b()
  !$omp end task
  
  !$omp taskwait  ! Wait for both tasks to complete
!$omp end single
!$omp end parallel
```

---

### 1.4 Memory Model

#### 1.4.1 Structure of the OpenMP Memory Model

**Memory Architecture:**
- **Thread Memory**: Each thread has its own temporary view of memory
- **Shared Memory**: All threads can access shared variables
- **Flush Operation**: Synchronizes thread's view with shared memory

**Key Concepts:**
- **Memory consistency**: When one thread's writes become visible to others
- **Relaxed consistency**: Compiler/hardware can reorder operations
- **Explicit synchronization**: Required to ensure visibility (`flush`, barriers)

#### 1.4.2 Device Data Environments

**For target constructs:**
- **Host device**: Where program starts (CPU)
- **Target device**: Accelerator/GPU where code executes
- **Data mapping**: Transfer between host and device (`map` clause)
- **Data persistence**: `target data` keeps data on device across constructs

**Example:**
```fortran
!$omp target data map(to:a) map(from:b)
  !$omp target
    ! a is on device, b will be copied back
    b = compute(a)
  !$omp end target
!$omp end target data
```

#### 1.4.3 Memory Consistency

**Sequential Consistency for Data-Race-Free Programs:**
- If no data races exist, memory operations appear in sequential order
- **Data race**: Two threads access same memory, at least one writes, no synchronization

**Avoiding Data Races:**
1. Use `atomic` for shared variable updates
2. Use `critical` sections for complex operations
3. Use `reduction` for combining values
4. Use proper synchronization (barriers, locks)

**Example - Data Race:**
```fortran
! ❌ DATA RACE
integer :: counter = 0
!$omp parallel
  counter = counter + 1  ! RACE: Multiple threads modify without sync
!$omp end parallel

! ✅ CORRECT
integer :: counter = 0
!$omp parallel
  !$omp atomic
  counter = counter + 1  ! SAFE: Atomic operation
!$omp end parallel
```

#### 1.4.4 The Flush Operation

**Purpose**: Ensure memory consistency across threads

**Implicit Flush Points:**
- At barriers
- At entry/exit of `parallel` regions
- At entry/exit of `critical` regions
- At `atomic` operations
- At `ordered` regions
- At `omp_set_lock`/`omp_unset_lock`

**Explicit Flush:**
```fortran
!$omp flush(variable_list)  ! Synchronize specific variables
!$omp flush                 ! Synchronize all shared variables
```

**Example:**
```fortran
integer :: flag = 0, data = 0

! Thread 1
data = 42
!$omp flush(data, flag)
flag = 1

! Thread 2
do while (flag == 0)
  !$omp flush(flag)
end do
!$omp flush(data)
! data is now guaranteed to be 42
```

---

### 1.5 Tool Interfaces

#### 1.5.1 OMPT (OpenMP Tools Interface)

**Purpose**: Performance measurement and analysis
- **Events**: Parallel region entry/exit, task creation, synchronization
- **Callbacks**: Tool registers handlers for events
- **Use cases**: Profilers, debuggers, performance analyzers

#### 1.5.2 OMPD (OpenMP Debugger Interface)

**Purpose**: Third-party debugger support
- **Introspection**: Examine OpenMP state (threads, tasks, variables)
- **Process control**: Not provided (debugger's responsibility)
- **Use cases**: GDB, LLDB extensions for OpenMP

---

### 1.6 OpenMP Compliance

**Conformance Requirements:**
- Implementation must support all features marked as required
- May provide additional features beyond specification
- Must document implementation-defined behaviors
- Must respect semantic constraints and restrictions

**Conforming Programs:**
- Use only features defined in specification
- Follow syntax rules for base language (C/C++/Fortran)
- Respect OpenMP restrictions and constraints
- Portable across conforming implementations

---

## Key OpenMP Concepts Summary

### Programming Model Hierarchy

```
Program
  └─ Sequential Part (Initial Task)
       └─ Parallel Region (Team of threads)
            ├─ Implicit Tasks (one per thread)
            │    ├─ Work-sharing Construct (distribute work)
            │    └─ Explicit Tasks (dynamic work)
            └─ Synchronization (barriers, atomics, critical)
```

### Execution Flow

1. **Initial Task**: Program starts with single thread
2. **Fork**: `parallel` creates team of threads
3. **Work-Sharing**: Distribute loop iterations, sections, or tasks
4. **Synchronization**: Barriers ensure all threads complete phase
5. **Join**: Threads synchronize at end of parallel region
6. **Continue**: Primary thread continues execution

### Data Sharing Hierarchy

1. **Predetermined**: By variable declaration (static → shared, iteration vars → private)
2. **Explicit**: Specified in clauses (`private`, `shared`, `firstprivate`, etc.)
3. **Implicit**: Derived from `default` clause or construct defaults
4. **Threadprivate**: Persistent thread-local storage across regions

### Synchronization Hierarchy

- **Coarse-grained**: Barriers (all threads wait)
- **Work-sharing**: Implicit barriers at end of constructs
- **Fine-grained**: Atomics (single variable), Critical (code section)
- **Task-based**: Taskwait (explicit tasks), Taskgroup (task tree)
- **Ordering**: Ordered (preserve sequential order in parallel loop)

---

## Quick Reference: Common Directive-Clause Compatibility

### PARALLEL Construct (Section 10.1)

**Name**: parallel  
**Category**: executable  
**Association**: block  
**Properties**: parallelism-generating, cancellable, thread-limiting, context-matching

**Allowed Clauses**:
- ✅ `allocate`
- ✅ `copyin`
- ✅ `default`
- ✅ `firstprivate`
- ✅ `if`
- ✅ `num_threads`
- ✅ `private`
- ✅ `proc_bind`
- ✅ `reduction`
- ✅ `shared`

**Binding**:
The binding thread set for a parallel region is the encountering thread. The encountering thread becomes the primary thread of the new team.

**Semantics**:
- When a thread encounters a parallel construct, a team of threads is created to execute the parallel region
- The thread that encountered the construct becomes the primary thread with thread number zero
- All threads in the team execute the region
- Thread numbers are consecutive whole numbers from 0 to (num_threads - 1)
- A set of implicit tasks equal to the number of threads is generated
- An implicit barrier occurs at the end of the parallel region
- After the parallel region, only the primary thread resumes execution

**Key Points**:
- Creates team of threads for parallel execution
- Supports data-sharing clauses (private, shared, firstprivate)
- Can be nested (creates new teams)
- Thread count controlled by `num_threads` clause or runtime

---

### TASK Construct (Section 12.5)

**Name**: task  
**Category**: executable  
**Association**: block  
**Properties**: parallelism-generating, thread-limiting, task-generating

**Allowed Clauses**:
- ✅ `affinity`
- ✅ `allocate`
- ✅ `default`
- ✅ `depend`
- ✅ `detach`
- ✅ `final`
- ✅ `firstprivate`
- ✅ `if`
- ✅ `in_reduction`
- ✅ `mergeable`
- ✅ `priority`
- ✅ `private`
- ✅ `shared`
- ✅ `untied`

**Clause Set** (exclusive properties):
- `detach` and `mergeable` are mutually exclusive

**Binding**:
The binding thread set of the task region is the current team. A task region binds to the innermost enclosing parallel region.

**Semantics**:
- Generates an explicit task from the associated structured block
- Data environment created according to data-sharing clauses
- Encountering thread may immediately execute or defer the task
- Any thread in the team may be assigned a deferred task
- Task completed when structured block execution completes (or detach event fulfilled)
- Includes task scheduling points

**Special Behaviors**:
- **Undeferred task**: When `if` clause evaluates to `false`, task is executed immediately
- **Detachable task**: With `detach` clause, completion requires allow-completion event
- **Final task**: With `final` clause, child tasks become included tasks
- **Untied task**: Can be suspended and resumed by different threads

**Common Use Cases**:
```fortran
! Basic task
!$omp task private(x)
  ! work
!$omp end task

! Task with dependencies
!$omp task depend(in: a) depend(out: b) private(x) allocate(x)
  ! work that depends on a, produces b
!$omp end task

! Conditional task
!$omp task if(n > threshold) private(x)
  ! Only create task if condition true
!$omp end task
```

---

### TASKLOOP Construct (Section 12.6)

**Name**: taskloop  
**Category**: executable  
**Association**: loop  
**Properties**: parallelism-generating, task-generating

**Allowed Clauses**:
- ✅ `allocate`
- ✅ `collapse`
- ✅ `default`
- ✅ `final`
- ✅ `firstprivate`
- ✅ `grainsize`
- ✅ `if`
- ✅ `in_reduction`
- ✅ `lastprivate`
- ✅ `mergeable`
- ✅ `nogroup`
- ✅ `num_tasks`
- ✅ `priority`
- ✅ `private`
- ✅ `reduction`
- ✅ `shared`
- ✅ `untied`

**Clause Sets** (exclusive groups):
- **synchronization-clause**: `nogroup` OR `reduction` (exclusive)
- **granularity-clause**: `grainsize` OR `num_tasks` (exclusive)

**Binding**:
The binding thread set of the taskloop region is the current team. A taskloop region binds to the innermost enclosing parallel region.

**Semantics**:
- Partitions loop iterations into chunks, each assigned to an explicit task
- Iteration count computed before loop entry
- Data environment created per data-sharing clauses
- Order of loop task creation is unspecified
- Unless `nogroup` present, executes as if enclosed in implicit `taskgroup`
- With `reduction`: implicit `task_reduction` on enclosing taskgroup, tasks get `in_reduction`

**Granularity Control**:
- `grainsize(n)`: Approximately `n` iterations per task
- `num_tasks(n)`: Create approximately `n` tasks
- Neither specified: implementation-defined task count

**Common Use Cases**:
```fortran
! Basic taskloop
!$omp taskloop private(i) allocate(i)
do i = 1, n
  ! work
end do
!$omp end taskloop

! Taskloop with grainsize
!$omp taskloop grainsize(100) private(i)
do i = 1, n
  ! work with ~100 iterations per task
end do
!$omp end taskloop

! Taskloop with reduction
!$omp taskloop reduction(+:sum) private(i)
do i = 1, n
  sum = sum + a(i)
end do
!$omp end taskloop

! Taskloop without implicit taskgroup
!$omp taskloop nogroup private(i)
do i = 1, n
  ! work without waiting at end
end do
!$omp end taskloop
```

---

### DEPOBJ Construct (Section 15.9.4)

**Name**: depobj  
**Category**: executable  
**Association**: none  
**Properties**: default

**Arguments**:
- `depobj(depend-object)` - Variable of OpenMP depend type

**Allowed Clauses** (one required, mutually exclusive):
- ✅ `depend` - Initialize depend object
- ✅ `destroy` - Uninitialize depend object
- ✅ `update` - Update dependence type

**Clause Set**:
- Properties: unique, required, exclusive
- Exactly one of `depend`, `destroy`, or `update` must be specified

**Binding**:
The binding thread set for a depobj region is the encountering thread.

**Semantics**:
- **With `depend` clause**: Sets depend-object state to initialized, represents specified dependence
- **With `update` clause**: Updates initialized depend-object to new dependence type
- **With `destroy` clause**: Sets depend-object state to uninitialized

**Restrictions**:
- `depend` clause must specify only ONE locator
- depend-object must be UNINITIALIZED when using `depend` clause
- depend-object must be INITIALIZED when using `destroy` or `update` clause

**Common Use Cases**:
```fortran
integer(omp_depend_kind) :: depobj_var

! Initialize depend object
!$omp depobj(depobj_var) depend(in: x)

! Use in task dependencies
!$omp task depend(depobj_var)
  ! work
!$omp end task

! Update dependence type
!$omp depobj(depobj_var) update(inout)

! Destroy depend object
!$omp depobj(depobj_var) destroy
```

**Why Use depobj**:
- Separate dependence specification from task creation
- Reuse same dependence across multiple tasks
- Dynamic dependence patterns
- Pass dependencies as first-class objects

---

### SIMD Construct (Section 10.4)

**Name**: simd  
**Category**: executable  
**Association**: loop  
**Properties**: parallelism-generating, context-matching, simdizable, pure

**Allowed Clauses**:
- ✅ `aligned`
- ✅ `collapse`
- ✅ `if`
- ✅ `lastprivate`
- ✅ `linear`
- ✅ `nontemporal`
- ✅ `order`
- ✅ `private`
- ✅ `reduction`
- ✅ `safelen`
- ✅ `simdlen`

**Separating Directives**:
- `scan` - Can be used within simd regions

**Binding**:
A simd region binds to the current task region. The binding thread set of the simd region is the current team.

**Semantics**:
- Enables execution of multiple loop iterations concurrently using SIMD instructions
- Each concurrent iteration executed by a different SIMD lane
- Number of concurrent iterations is implementation-defined
- Each set of concurrent iterations is a SIMD chunk
- Lexical forward dependencies preserved within each SIMD chunk (unless `order(concurrent)`)
- When `if` clause is false, preferred number of concurrent iterations is one

**Key Restrictions**:
- If both `simdlen` and `safelen` specified, `simdlen` value ≤ `safelen` value
- Only simdizable constructs can be encountered during execution
- If `order(concurrent)`, cannot also have `safelen` clause
- Cannot contain calls to `longjmp` or `setjmp`
- No exceptions can be raised in simd region (C++)

**Common Use Cases**:
```fortran
! Basic SIMD loop
!$omp simd
do i = 1, n
  a(i) = b(i) + c(i)
end do
!$omp end simd

! SIMD with private variables
!$omp simd private(temp)
do i = 1, n
  temp = b(i) * 2
  a(i) = temp + c(i)
end do
!$omp end simd

! SIMD with reduction
!$omp simd reduction(+:sum)
do i = 1, n
  sum = sum + a(i)
end do
!$omp end simd

! SIMD with safelen (max dependency distance)
!$omp simd safelen(8)
do i = 9, n
  a(i) = a(i-8) + b(i)
end do
!$omp end simd

! SIMD with aligned clause (for vectorization optimization)
!$omp simd aligned(a,b:64)
do i = 1, n
  a(i) = b(i) + c(i)
end do
!$omp end simd
```

---

### DECLARE SIMD Directive (Section 7.7)

**Name**: declare simd  
**Category**: declarative  
**Association**: declaration  
**Properties**: pure

**Arguments**:
- `declare simd[(proc-name)]` - Optional procedure name

**Allowed Clauses**:
- ✅ `aligned`
- ✅ `linear`
- ✅ `simdlen`
- ✅ `uniform`

**Clause Groups**:
- `branch` - Optional clause group for branch hints

**Semantics**:
- Enables creation of SIMD versions of associated function
- SIMD versions can process multiple arguments from single invocation in SIMD loop concurrently
- If `simdlen` not specified, number of concurrent arguments is implementation-defined
- For `linear` clause, uniform parameters are considered constant

**Key Restrictions**:
- Function/subroutine body must be structured block
- Cannot execute OpenMP constructs except `ordered simd` or `atomic`
- No side effects that alter execution for concurrent SIMD chunk iterations
- Cannot contain `longjmp`/`setjmp` calls (C/C++)
- Cannot contain `throw` statements (C++)
- In Fortran: proc-name cannot be generic name, procedure pointer, or entry name

**Common Use Cases**:
```fortran
! Declare SIMD function
!$omp declare simd(compute) uniform(n) linear(i)
function compute(i, n) result(val)
  integer :: i, n, val
  val = i * n
end function

! Use in SIMD loop
!$omp simd
do i = 1, n
  a(i) = compute(i, n)  ! Calls SIMD version
end do
!$omp end simd

! Declare SIMD with aligned
!$omp declare simd aligned(a,b:64)
subroutine vector_add(a, b, n)
  real(8), dimension(*) :: a, b
  integer :: n
  ! ...
end subroutine
```

---

### Combined SIMD Constructs

OpenMP provides several combined constructs that pair SIMD with other directives:

#### FOR SIMD / DO SIMD
```fortran
! Combines worksharing loop with SIMD
!$omp do simd private(i)
do i = 1, n
  a(i) = b(i) + c(i)
end do
!$omp end do simd
```
- Distributes iterations across threads, each thread executes its chunk with SIMD
- Allowed clauses: union of `do` and `simd` clauses

#### PARALLEL FOR SIMD / PARALLEL DO SIMD
```fortran
! Combines parallel, worksharing, and SIMD
!$omp parallel do simd private(i) reduction(+:sum)
do i = 1, n
  sum = sum + a(i)
end do
!$omp end parallel do simd
```
- Creates parallel region + worksharing + SIMD in one directive
- Most common for parallelizing vectorizable loops

#### DISTRIBUTE SIMD
```fortran
! For use in teams constructs
!$omp teams distribute simd
do i = 1, n
  a(i) = b(i) + c(i)
end do
!$omp end teams distribute simd
```
- Distributes iterations across teams, each team uses SIMD

#### TASKLOOP SIMD
```fortran
! Task-based loop with SIMD
!$omp taskloop simd private(i) grainsize(100)
do i = 1, n
  a(i) = b(i) * c(i)
end do
!$omp end taskloop simd
```
- Partitions iterations into tasks, each task executes with SIMD
- Allowed clauses: union of `taskloop` and `simd` clauses

---

### TASKGROUP Construct (Section 15.4)

**Name**: taskgroup  
**Category**: executable  
**Association**: block  
**Properties**: cancellable

**Allowed Clauses**:
- ✅ `allocate` (only with task_reduction variables - see restrictions)
- ✅ `task_reduction`

**NOT Allowed**:
- ❌ `private`, `firstprivate`, `lastprivate` (no data-sharing attribute clauses)
- ❌ `shared`
- ❌ Any other data-sharing clauses

**Binding**:
The binding task set of a taskgroup region is all tasks of the current team that are generated in the region. A taskgroup region binds to the innermost enclosing parallel region.

**Semantics**:
- Specifies a wait on completion of the taskgroup set associated with the taskgroup region
- When a thread encounters a taskgroup construct, it starts executing the region
- An implicit task scheduling point occurs at the end of the taskgroup region
- The current task is suspended until all tasks in the taskgroup set complete execution

---

## ALLOCATE Clause (Section 6.6)

**Name**: allocate  
**Properties**: default

**Allowed on Directives**:
- ✅ allocators
- ✅ distribute
- ✅ do
- ✅ for
- ✅ parallel
- ✅ scope
- ✅ sections
- ✅ single
- ✅ target
- ✅ task
- ✅ **taskgroup** (⚠️ with special restrictions - see below)
- ✅ taskloop
- ✅ teams

### Modifiers

**allocator-simple-modifier**: Expression of `omp_allocator_handle_t` type

**allocator-complex-modifier**: 
- Name: `allocator`
- Arguments: `allocator` expression of `allocator_handle` type (default)
- Properties: exclusive, unique

**align-modifier**:
- Name: `align`
- Arguments: `alignment` expression of integer type (constant, positive)
- Properties: unique

### Syntax Examples

```fortran
! Simple form
!$omp parallel private(x) allocate(x)

! With allocator modifier
!$omp parallel private(x) allocate(omp_high_bw_mem_alloc: x)

! With align modifier
!$omp parallel private(x) allocate(align(64): x)
```

### Semantics

The `allocate` clause specifies the memory allocator to be used to obtain storage for a list of variables. 

**Key behavior**:
- If a list item in the clause also appears in a **data-sharing attribute clause on the same directive** that privatizes the list item, allocations that arise from that list item will be provided by the memory allocator
- If the allocator-simple-modifier is specified, the behavior is as if the allocator-complex-modifier is instead specified with allocator-simple-modifier as its allocator argument
- For allocations that arise from this clause, the `null_fb` value of the fallback allocator trait behaves as if the `abort_fb` had been specified

### **CRITICAL RESTRICTIONS**

#### Restriction 1: Must Have Data-Sharing Attribute Clause
> **For any list item that is specified in the `allocate` clause on a directive other than the `allocators` directive, a data-sharing attribute clause that may create a private copy of that list item MUST be specified on the SAME directive.**

**Examples**:

✅ **Valid** (allocate with private on same directive):
```fortran
!$omp parallel private(x) allocate(x)
!$omp task private(y) allocate(y)
!$omp taskloop private(z) allocate(z)
```

❌ **INVALID** (allocate without data-sharing clause on same directive):
```fortran
!$omp parallel private(x)
!$omp taskgroup allocate(x)  ! ERROR: No private clause on taskgroup
!$omp end taskgroup
!$omp end parallel
```

❌ **INVALID** (private on different directive):
```fortran
!$omp parallel private(x)
!$omp task allocate(x)  ! ERROR: x is private on parallel, not on task
!$omp end task
!$omp end parallel
```

✅ **Valid** (each directive has its own data-sharing clause):
```fortran
!$omp parallel private(x) allocate(x)
!$omp task private(y) allocate(y)
!$omp end task
!$omp end parallel
```

#### Restriction 2: Thread Access for Task/Taskloop/Target
> **For task, taskloop or target directives, allocation requests to memory allocators with the trait `access` set to `thread` result in unspecified behavior.**

#### Restriction 3: Target Region Allocators
> **`allocate` clauses that appear on a `target` construct or on constructs in a target region must specify an `allocator-simple-modifier` or `allocator-complex-modifier` unless a `requires` directive with the `dynamic_allocators` clause is present in the same compilation unit.**

---

## TASKGROUP + ALLOCATE: Special Case

### Why You Cannot Use allocate(x) on taskgroup Without task_reduction

**Scenario**:
```fortran
integer :: x
!$omp parallel private(x)
!$omp taskgroup allocate(x)  ! ❌ ERROR
!$omp end taskgroup
!$omp end parallel
```

**Why this fails**:
1. `taskgroup` does NOT support `private`, `firstprivate`, `lastprivate`, or any data-sharing attribute clauses
2. The `allocate` clause REQUIRES a data-sharing attribute clause on the **SAME directive** (per Restriction 1)
3. Since `taskgroup` cannot have `private(x)`, you cannot have `allocate(x)` either

**Valid alternatives**:

✅ **Option 1**: Move allocate to the directive that has the private clause
```fortran
integer :: x
!$omp parallel private(x) allocate(x)  ! allocate on parallel where private is
!$omp taskgroup
!$omp end taskgroup
!$omp end parallel
```

✅ **Option 2**: Use allocate with task_reduction on taskgroup
```fortran
integer :: x
!$omp parallel
!$omp taskgroup task_reduction(+:x) allocate(x)  ! Valid: task_reduction is a data-sharing clause
!$omp end taskgroup
!$omp end parallel
```

✅ **Option 3**: Use allocate on task directive inside taskgroup
```fortran
integer :: x
!$omp parallel
!$omp taskgroup
!$omp task private(x) allocate(x)  ! Valid: private and allocate on same task directive
!$omp end task
!$omp end taskgroup
!$omp end parallel
```

---

## Common Error Messages and Solutions

### Error: "The ALLOCATE clause requires that 'x' must be listed in a private data-sharing attribute clause on the same directive"

**Cause**: The variable appears in `allocate()` clause but not in a data-sharing attribute clause (`private`, `firstprivate`, `lastprivate`, `linear`, `task_reduction`, `in_reduction`) on the **same directive**.

**Solution**: Add the appropriate data-sharing clause to the same directive, or move the `allocate` clause to a directive that has the data-sharing clause.

**Example Fix**:
```fortran
! WRONG:
!$omp taskgroup allocate(x)

! RIGHT (Option 1 - move allocate to where private is):
!$omp parallel private(x) allocate(x)
!$omp taskgroup
!$omp end taskgroup
!$omp end parallel

! RIGHT (Option 2 - use with task_reduction):
!$omp taskgroup task_reduction(+:x) allocate(x)
```

---

## Data-Sharing Attribute Clauses

These clauses create private copies and can be used with `allocate` on the same directive:

- `private(list)` - Each thread gets its own uninitialized copy
- `firstprivate(list)` - Each thread gets its own copy initialized from master
- `lastprivate(list)` - Last iteration value copied to original variable
- `linear(list[:step])` - Values increment linearly across iterations
- `task_reduction(operator:list)` - Task-based reduction
- `in_reduction(operator:list)` - Participating in enclosing reduction

---

## Directive Categories and Common Clauses

### Work-Sharing Constructs
**Directives**: `for`, `do`, `sections`, `single`, `workshare`  
**Common Clauses**: `private`, `firstprivate`, `lastprivate`, `allocate`, `nowait`

### Tasking Constructs  
**Directives**: `task`, `taskloop`, `taskgroup`  
**`task` Clauses**: `private`, `firstprivate`, `allocate`, `depend`, `priority`, `if`, `final`  
**`taskloop` Clauses**: `private`, `firstprivate`, `lastprivate`, `allocate`, `grainsize`, `num_tasks`, `reduction`  
**`taskgroup` Clauses**: `task_reduction`, `allocate` (only with task_reduction)

### Parallel Constructs
**Directives**: `parallel`, `teams`  
**Common Clauses**: `private`, `firstprivate`, `shared`, `allocate`, `if`, `num_threads`

### Target Constructs
**Directives**: `target`, `target data`, `target enter data`, `target exit data`, `target update`  
**Common Clauses**: `map`, `private`, `firstprivate`, `allocate`, `device`, `if`, `nowait`

---

## Cross-References

- Memory Allocators: Section 6.2
- align clause: Section 6.3
- allocator clause: Section 6.4
- allocators directive: Section 6.7
- taskgroup directive: Section 15.4
- task directive: Section 12.5
- taskloop directive: Section 12.6
- parallel directive: Section 10.1

---

## Quick Lookup: "Can I use allocate() on this directive?"

| Directive      | allocate Clause | Requires Data-Sharing Clause? | Valid Data-Sharing Clauses                    |
|----------------|-----------------|-------------------------------|-----------------------------------------------|
| `parallel`     | ✅ Yes          | Yes                           | `private`, `firstprivate`, `shared`           |
| `for`/`do`     | ✅ Yes          | Yes                           | `private`, `firstprivate`, `lastprivate`      |
| `sections`     | ✅ Yes          | Yes                           | `private`, `firstprivate`, `lastprivate`      |
| `single`       | ✅ Yes          | Yes                           | `private`, `firstprivate`                     |
| `task`         | ✅ Yes          | Yes                           | `private`, `firstprivate`                     |
| `taskloop`     | ✅ Yes          | Yes                           | `private`, `firstprivate`, `lastprivate`      |
| `taskgroup`    | ⚠️ Limited     | Yes                           | `task_reduction` ONLY                         |
| `target`       | ✅ Yes          | Yes (+ allocator required)    | `private`, `firstprivate`, `map`              |
| `teams`        | ✅ Yes          | Yes                           | `private`, `firstprivate`, `shared`           |
| `distribute`   | ✅ Yes          | Yes                           | `private`, `firstprivate`, `lastprivate`      |
| `scope`        | ✅ Yes          | Yes                           | `private`                                     |

## Quick Lookup: "Which clauses are allowed on this directive?"

| Directive   | Common Clauses | Data-Sharing | Special Clauses |
|-------------|---------------|--------------|-----------------|
| `parallel`  | if, num_threads, proc_bind, reduction, copyin, allocate | private, firstprivate, shared, default | - |
| `task`      | if, final, priority, depend, detach, mergeable, untied, affinity, allocate | private, firstprivate, shared, default, in_reduction | - |
| `taskloop`  | if, final, priority, mergeable, untied, nogroup, grainsize, num_tasks, collapse, allocate | private, firstprivate, lastprivate, shared, default, reduction, in_reduction | grainsize/num_tasks exclusive |
| `taskgroup` | allocate | task_reduction | allocate only with task_reduction |
| `depobj`    | depend, update, destroy | - | Requires exactly one: depend/update/destroy |
| `simd`      | if, collapse, safelen, simdlen, aligned, linear, nontemporal, order | private, lastprivate, reduction | safelen/order(concurrent) exclusive |
| `declare simd` | simdlen, aligned, linear, uniform, branch | - | Declarative (not executable) |

## SIMD Clause Compatibility

| Combined Construct | Allowed Clauses | Notes |
|--------------------|----------------|-------|
| `do simd` / `for simd` | Union of `do`/`for` + `simd` clauses | private, lastprivate, linear, reduction, schedule, collapse, ordered, nowait, if, safelen, simdlen, aligned |
| `parallel do simd` / `parallel for simd` | Union of `parallel` + `do`/`for` + `simd` clauses | Combines parallel region, worksharing, and SIMD vectorization |
| `distribute simd` | Union of `distribute` + `simd` clauses | Used inside `teams` constructs |
| `taskloop simd` | Union of `taskloop` + `simd` clauses | Task-based loops with SIMD vectorization |

---

## Remember

1. **Golden Rule**: `allocate(var)` requires a data-sharing attribute clause for `var` on the **SAME directive**
2. **taskgroup limitation**: Only supports `task_reduction` as a data-sharing clause, so `allocate` is limited
3. **Common mistake**: Putting `allocate` on a directive where the variable is private on a different (parent) directive
4. **Fix strategy**: Either move `allocate` to where `private` is, or add the data-sharing clause to the same directive

---

## CHAPTER 5: DATA ENVIRONMENT

This chapter covers the fundamental rules for how variables are accessed in OpenMP constructs - one of the most critical aspects of correct OpenMP programming.

### 5.1 Data-Sharing Attribute Rules

OpenMP classifies variables into three categories:

1. **Predetermined data-sharing attributes**: Variables whose sharing is determined by their declaration
2. **Explicitly determined**: Variables listed in data-sharing attribute clauses
3. **Implicitly determined**: Variables not in the above categories (determined by context/default clause)

#### Predetermined Data-Sharing Attributes

Variables with predetermined attributes **CANNOT** be listed in data-sharing clauses (except for specific exceptions).

**C/C++ Rules:**
- ✅ **Shared by default**:
  - Variables with static storage duration declared in construct scope
  - Static data members
  - `__func__` and similar predefined variables
- ✅ **Private by default**:
  - Base pointers in `map` clause on target construct (if not in map clause themselves)
  - Base pointers in `reduction`/`in_reduction` clauses
- ✅ **Loop iteration variables**: Private for associated loops

**Fortran Rules:**
- ✅ **Shared by default**:
  - Cray pointees (follow pointer's data-sharing)
  - Assumed-size arrays
  - Named constants
  - Associate names (if association occurs outside construct)
- ✅ **Loop iteration variables**: Private for associated loops

**Exceptions** (when predetermined vars CAN be listed in clauses):
```fortran
! Loop iteration variables can be private or lastprivate
!$omp parallel do private(i)
do i = 1, n
  ! valid
end do

! SIMD loop iteration can be linear
!$omp simd linear(i:1)
do i = 1, n
  ! valid if step matches loop increment
end do

! Const-qualified can be firstprivate (C++)
!$omp parallel firstprivate(const_var)

! Assumed-size arrays can be shared (Fortran)
!$omp parallel shared(assumed_size_array)
```

#### Explicitly Determined Data-Sharing

Variables referenced in a construct and **listed in a data-sharing attribute clause** on that construct have explicitly determined attributes.

Example:
```fortran
!$omp parallel private(x) shared(y) firstprivate(z)
  ! x is explicitly private
  ! y is explicitly shared
  ! z is explicitly firstprivate
!$omp end parallel
```

#### Implicitly Determined Data-Sharing

Variables referenced in a construct **without predetermined or explicit attributes** are implicitly determined by:

**For `parallel`, `teams`, `task` constructs:**
- If `default` clause present → follows its specification
- If no `default` clause:
  - `parallel`: Variables are **shared**
  - `task` (orphaned): Dummy arguments/formal arguments passed by reference are **firstprivate**
  - `task` (non-orphaned): Variables shared in enclosing team are **shared**, others are **firstprivate**
  - `target`: Unmapped variables are **firstprivate**

**For other constructs:**
- If no `default` clause → reference enclosing context variables

**Example:**
```fortran
integer :: shared_var, private_var

!$omp parallel private(private_var)  ! explicit
  ! shared_var is implicitly shared (no default, parallel construct)
  !$omp task  ! orphaned task
    ! shared_var is implicitly shared (shared in enclosing team)
    ! Any new variable here would be implicitly firstprivate
  !$omp end task
!$omp end parallel
```

---

### 5.2 threadprivate Directive

**Name**: threadprivate  
**Category**: declarative  
**Association**: declaration

**Purpose**: Make global/static variables private to each thread, with persistent values across parallel regions

**Syntax:**
```fortran
!$omp threadprivate(list)
```

**Semantics:**
- Each thread gets its own copy of threadprivate variables
- Values persist across parallel regions (unlike `private`)
- Initial values are undefined unless `copyin` clause is used
- Each copy is allocated once per thread and exists for the thread's lifetime

**Common Use Cases:**
```fortran
! Module-level threadprivate
module data_mod
  integer, save :: thread_id
  !$omp threadprivate(thread_id)
end module

program main
  use data_mod
  !$omp parallel copyin(thread_id)
    thread_id = omp_get_thread_num()
    call work()  ! thread_id accessible here
  !$omp end parallel
  
  !$omp parallel  ! No copyin
    ! thread_id retains value from previous parallel region
    call more_work()
  !$omp end parallel
end program
```

---

### 5.3 List Item Privatization

When a variable is privatized (via `private`, `firstprivate`, `lastprivate`, `linear`, etc.), OpenMP creates new instances of the variable.

**Privatization Semantics:**
- Each task or SIMD lane gets its own copy of the variable
- Original variable is preserved (except for `lastprivate`)
- New copies are uninitialized (for `private`) or initialized (for `firstprivate`)
- Storage for private copies is allocated automatically

**Example:**
```fortran
integer :: x = 10

!$omp parallel private(x) num_threads(4)
  ! Each thread has its own uninitialized x
  x = omp_get_thread_num()
  print *, "Thread", omp_get_thread_num(), "x =", x
!$omp end parallel

! Original x is still 10 (not modified by parallel region)
print *, "Original x =", x  ! Output: 10
```

---

### 5.4 Data-Sharing Attribute Clauses

These clauses explicitly control how variables are shared or privatized in OpenMP constructs.

#### 5.4.1 default Clause

**Name**: default  
**Arguments**: `data-sharing-attribute` (Keyword: `firstprivate`, `none`, `private`, `shared`)  
**Properties**: unique  
**Allowed on**: `parallel`, `task`, `taskloop`, `teams`

**Purpose**: Set implicit data-sharing for all variables without predetermined or explicit attributes

**Semantics:**
- `default(shared)`: Implicitly determined variables are shared
- `default(private)`: Implicitly determined variables are private
- `default(firstprivate)`: Implicitly determined variables are firstprivate
- `default(none)`: **ALL** variables must be explicitly listed in clauses (prevents accidents)

**Best Practice:** Use `default(none)` to avoid unintended sharing bugs!

**Examples:**
```fortran
! Require explicit specification (recommended)
integer :: x, y, z
!$omp parallel default(none) private(x) shared(y)  ! ERROR: z not listed
  x = 1
  y = 2
  z = 3  ! Compile error: z has no data-sharing attribute
!$omp end parallel

! Allow implicit sharing
!$omp parallel default(shared) private(x)
  ! x is explicitly private
  ! y and z are implicitly shared
  x = 1
  y = 2  ! Modifies shared y (potential race condition!)
!$omp end parallel
```

**Restrictions:**
- If `default(none)`, every referenced variable without predetermined attribute must be explicitly listed
- If `default(firstprivate)` or `default(private)` in C/C++, static storage duration variables must still be explicit

---

#### 5.4.2 shared Clause

**Name**: shared  
**Arguments**: `list` (variable list item type)  
**Allowed on**: `parallel`, `task`, `taskloop`, `teams`

**Purpose**: Specify that all threads/tasks access the **same** memory location

**Semantics:**
- All threads reference the original variable
- No copying occurs
- **Requires synchronization** to avoid data races
- Modifications by one thread are visible to others (after appropriate barriers/flushes)

**Examples:**
```fortran
integer :: shared_counter
shared_counter = 0

!$omp parallel shared(shared_counter) num_threads(4)
  !$omp atomic
  shared_counter = shared_counter + 1  ! Safe with atomic
!$omp end parallel

print *, "Counter =", shared_counter  ! Output: 4
```

**Common Errors:**
```fortran
! ❌ RACE CONDITION - Missing synchronization
integer :: sum
sum = 0
!$omp parallel shared(sum) num_threads(4)
  sum = sum + omp_get_thread_num()  ! RACE CONDITION!
!$omp end parallel

! ✅ CORRECT - Use reduction instead
!$omp parallel reduction(+:sum) num_threads(4)
  sum = sum + omp_get_thread_num()
!$omp end parallel
```

---

#### 5.4.3 private Clause

**Name**: private  
**Arguments**: `list` (variable list item type)  
**Properties**: data-environment attribute, data-sharing attribute, privatization  
**Allowed on**: `distribute`, `do`, `for`, `loop`, `parallel`, `scope`, `sections`, `simd`, `single`, `target`, `task`, `taskloop`, `teams`

**Purpose**: Each thread/task gets its own **uninitialized** copy of the variable

**Semantics:**
- New list item created for each task/thread
- Original value is **NOT** copied (unlike `firstprivate`)
- Original variable is **NOT** updated (unlike `lastprivate`)
- Value after construct is unspecified

**Examples:**
```fortran
integer :: x
x = 100

!$omp parallel private(x) num_threads(2)
  ! x is uninitialized here - value is undefined!
  x = omp_get_thread_num()
  print *, "Thread", omp_get_thread_num(), "x =", x
!$omp end parallel

! Original x is still 100 (not affected)
print *, "After parallel, x =", x  ! Output: 100
```

**Common Use - Loop Iteration:**
```fortran
!$omp parallel do private(i, temp)
do i = 1, n
  temp = a(i) * 2  ! Each thread has its own temp
  b(i) = temp + c(i)
end do
!$omp end parallel do
```

**Restrictions:** See Section 5.3 for privatization restrictions

---

#### 5.4.4 firstprivate Clause

**Name**: firstprivate  
**Arguments**: `list` (variable list item type)  
**Properties**: data-environment attribute, data-sharing attribute, privatization  
**Allowed on**: `distribute`, `do`, `for`, `parallel`, `scope`, `sections`, `single`, `target`, `task`, `taskloop`, `teams`

**Purpose**: Each thread/task gets its own copy **initialized** from the original variable

**Semantics:**
- Provides superset of `private` functionality
- New list item created AND initialized with original value
- Initialization occurs **before** construct execution
- For `parallel`: initialization from value immediately before construct
- For work-distribution: initialization from value in enclosing implicit task

**Examples:**
```fortran
integer :: x
x = 100

!$omp parallel firstprivate(x) num_threads(2)
  ! x is initialized to 100 for each thread
  print *, "Thread", omp_get_thread_num(), "initial x =", x  ! Output: 100
  x = omp_get_thread_num()
  print *, "Thread", omp_get_thread_num(), "modified x =", x
!$omp end parallel

! Original x is still 100 (not affected by modifications)
print *, "After parallel, x =", x  ! Output: 100
```

**Common Use - Passing Values to Tasks:**
```fortran
do i = 1, n
  !$omp task firstprivate(i)
    ! Each task gets its own copy of current i value
    call process(a(i))
  !$omp end task
end do
!$omp taskwait
```

**Initialization Details:**
- **C/C++**: Copy assignment (non-array), element-wise copy (arrays)
- **C++**: Copy constructor invoked (except on target constructs - unspecified)
- **Fortran**: Intrinsic assignment or type-bound defined assignment

**Key Difference from private:**
```fortran
integer :: x = 50

! private: x is UNINITIALIZED
!$omp parallel private(x)
  ! x has garbage value here!
  
! firstprivate: x is INITIALIZED to 50
!$omp parallel firstprivate(x)
  ! x is 50 here
```

---

#### 5.4.5 lastprivate Clause

**Name**: lastprivate  
**Arguments**: `list` (variable list item type)  
**Modifiers**: `conditional` (optional)  
**Properties**: data-environment attribute, data-sharing attribute, privatization  
**Allowed on**: `distribute`, `do`, `for`, `loop`, `sections`, `simd`, `taskloop`

**Purpose**: Each thread gets private copy, **AND** last iteration's value is copied back to original

**Semantics:**
- Provides superset of `private` functionality
- Each thread/task gets private copy (uninitialized)
- After construct, original variable is assigned value from:
  - **Without `conditional`**: Sequentially last iteration (or lexically last section)
  - **With `conditional`**: Last iteration that would assign in sequential execution

**Examples:**
```fortran
integer :: x, last_i

! Without conditional
!$omp parallel do lastprivate(last_i)
do i = 1, 100
  last_i = i
end do
!$omp end parallel do
print *, "last_i =", last_i  ! Output: 100

! With conditional (only if assigned)
x = 0
!$omp parallel do lastprivate(conditional: x)
do i = 1, 100
  if (a(i) > threshold) then
    x = i  ! Only last such assignment survives
  end if
end do
!$omp end parallel do
! x contains the last i where a(i) > threshold, or 0 if none
```

**Common Use - Loop Termination Values:**
```fortran
integer :: i, n
real(8) :: sum

sum = 0.0
!$omp parallel do reduction(+:sum) lastprivate(i)
do i = 1, 1000
  sum = sum + a(i)
end do
!$omp end parallel do

print *, "Processed", i-1, "elements, sum =", sum
! i is now 1001 (value after last iteration)
```

**Assignment Details:**
- **C/C++**: Copy assignment operator (class types), element-wise (arrays)
- **Fortran**: Intrinsic assignment or pointer assignment (for pointer attributes)

**Restrictions:**
- Cannot use `lastprivate` on work-distribution construct if variable is private in enclosing parallel region
- With `conditional`: modifying variable outside the construct leads to unspecified behavior

---

#### 5.4.6 linear Clause

**Name**: linear  
**Arguments**: `list` (variable list item type)  
**Modifiers**: `step-simple-modifier` or `linear-modifier` (optional)  
**Properties**: data-environment attribute, data-sharing attribute, privatization  
**Allowed on**: `distribute`, `do`, `for`, `simd`, `declare simd`

**Purpose**: Variable has linear relationship with iteration number (used for vectorization)

**Syntax:**
```fortran
! Simple step
!$omp simd linear(x:2)

! With modifier
!$omp simd linear(val(x):1)
```

**Modifiers:**
- `val(list:step)`: Value mode (default for scalars)
- `ref(list:step)`: Reference mode (address is linear)
- `uval(list:step)`: Uniform value mode

**Semantics:**
- Variable value is `original_value + iteration * step`
- Privatized similar to `private` clause
- At end of construct, original variable is assigned value from last iteration

**Examples:**
```fortran
! Linear loop index
integer :: i
real(8), dimension(100) :: a, b

!$omp simd linear(i:1)
do i = 1, 100
  a(i) = b(i) * 2.0
end do
! i is linear with step 1

! Linear pointer arithmetic
integer :: i, ptr_offset
ptr_offset = 0
!$omp simd linear(ptr_offset:1)
do i = 1, 100
  c(ptr_offset) = a(i) + b(i)
  ptr_offset = ptr_offset + 1
end do
```

**Use with declare simd:**
```fortran
!$omp declare simd linear(i:1) uniform(n)
function compute(i, n) result(val)
  integer :: i, n, val
  val = i * n
end function
```

**Restrictions:**
- Step must be loop invariant integer
- If both `simdlen` and `safelen` are specified with `linear`, restrictions apply

---

#### 5.4.7 is_device_ptr Clause

**Name**: is_device_ptr  
**Arguments**: `list` (variable list item type)  
**Allowed on**: `target`, `dispatch`

**Purpose**: Indicate that list items are device pointers (for target constructs)

**Use Case:**
```fortran
! When pointer is already a device pointer
type(c_ptr) :: device_ptr

!$omp target is_device_ptr(device_ptr)
  ! Use device_ptr directly without mapping
!$omp end target
```

---

#### 5.4.8 use_device_ptr Clause

**Name**: use_device_ptr  
**Arguments**: `list` (variable list item type)  
**Allowed on**: `target data`

**Purpose**: Make device pointer addresses available in host code

**Example:**
```fortran
real, pointer :: d(:)

!$omp target data map(d) use_device_ptr(d)
  ! d now contains device pointer address
  call cuda_kernel(d)  ! Pass device pointer to CUDA
!$omp end target data
```

---

#### 5.4.9 has_device_addr Clause

**Name**: has_device_addr  
**Arguments**: `list` (variable list item type)  
**Allowed on**: `target`

**Purpose**: Indicate list items have device address (no mapping needed)

---

#### 5.4.10 use_device_addr Clause

**Name**: use_device_addr  
**Arguments**: `list` (variable list item type)  
**Allowed on**: `target data`

**Purpose**: Similar to `use_device_ptr` but for addresses

---

### 5.5 Reduction Clauses and Directives

Reductions are operations that combine values from all threads/tasks into a single result (e.g., sum, max, min).

#### 5.5.1 OpenMP Reduction Identifiers

**Predefined Reduction Operators:**

**Fortran:**
- Numeric: `+`, `*`, `-`, `min`, `max`
- Logical: `.and.`, `.or.`, `.eqv.`, `.neqv.`
- Bitwise: `.iand.`, `.ior.`, `.ieor.`

**C/C++:**
- Numeric: `+`, `*`, `-`, `min`, `max`
- Logical: `&&`, `||`
- Bitwise: `&`, `|`, `^`

**Example:**
```fortran
!$omp parallel do reduction(+:sum) reduction(max:max_val)
do i = 1, n
  sum = sum + a(i)
  max_val = max(max_val, a(i))
end do
```

#### 5.5.2 OpenMP Reduction Expressions

The reduction identifier determines how partial results are combined:

- `+`: `x = x + expr` (sum)
- `*`: `x = x * expr` (product)
- `max`: `x = max(x, expr)` (maximum)
- `min`: `x = min(x, expr)` (minimum)
- `.and.`: `x = x .and. expr` (logical AND)
- `.or.`: `x = x .or. expr` (logical OR)

**Initial Values:**
- `+`: 0
- `*`: 1
- `max`: Smallest representable value
- `min`: Largest representable value
- `.and.`: `.true.`
- `.or.`: `.false.`

#### 5.5.5 Properties Common to All Reduction Clauses

**All reduction clauses:**
- Create private copies initialized with operator's identity value
- Each thread updates its private copy
- At end of construct, private copies are reduced to original variable
- Requires synchronization to avoid races

**Example:**
```fortran
integer :: sum, product
real(8) :: max_val

sum = 0
product = 1
max_val = -huge(max_val)

!$omp parallel reduction(+:sum) reduction(*:product) reduction(max:max_val)
  !$omp do
  do i = 1, n
    sum = sum + a(i)
    product = product * b(i)
    max_val = max(max_val, c(i))
  end do
!$omp end parallel

print *, "Sum:", sum, "Product:", product, "Max:", max_val
```

#### 5.5.8 reduction Clause

**Name**: reduction  
**Arguments**: `reduction-identifier`, `list`  
**Allowed on**: `parallel`, `do`, `for`, `loop`, `sections`, `scope`, `simd`, `taskloop`, `teams`

**Purpose**: Standard reduction across threads in a parallel region or loop

**Syntax:**
```fortran
!$omp parallel do reduction(+:sum)
!$omp parallel do reduction(max:max_val)
```

**Common Use Cases:**
```fortran
! Sum reduction
real(8) :: sum
sum = 0.0
!$omp parallel do reduction(+:sum)
do i = 1, n
  sum = sum + array(i)
end do

! Multiple reductions
integer :: count
real(8) :: sum, avg
count = 0
sum = 0.0
!$omp parallel do reduction(+:count) reduction(+:sum)
do i = 1, n
  if (array(i) > threshold) then
    count = count + 1
    sum = sum + array(i)
  end if
end do
avg = sum / count

! Logical reduction
logical :: all_positive
all_positive = .true.
!$omp parallel do reduction(.and.:all_positive)
do i = 1, n
  all_positive = all_positive .and. (array(i) > 0.0)
end do
```

#### 5.5.9 task_reduction Clause

**Name**: task_reduction  
**Arguments**: `reduction-identifier`, `list`  
**Allowed on**: `taskgroup`

**Purpose**: Reduction across explicit tasks in a taskgroup

**Use with in_reduction:**
```fortran
integer :: sum
sum = 0

!$omp taskgroup task_reduction(+:sum)
  do i = 1, n
    !$omp task in_reduction(+:sum)
      sum = sum + compute(i)
    !$omp end task
  end do
!$omp end taskgroup

print *, "Total sum:", sum
```

#### 5.5.10 in_reduction Clause

**Name**: in_reduction  
**Arguments**: `reduction-identifier`, `list`  
**Allowed on**: `task`, `taskloop`, `target`

**Purpose**: Participate in an enclosing reduction

**Example:**
```fortran
integer :: result
result = 0

!$omp parallel
!$omp single
  !$omp taskgroup task_reduction(+:result)
    do i = 1, 100
      !$omp task in_reduction(+:result)
        result = result + a(i)
      !$omp end task
    end do
  !$omp end taskgroup
!$omp end single
!$omp end parallel

print *, "Result:", result
```

#### 5.5.11 declare reduction Directive

**Name**: declare reduction  
**Category**: declarative

**Purpose**: Define custom reduction operations

**Syntax:**
```fortran
!$omp declare reduction(reduction-identifier : type : combiner) [initializer(init-expr)]
```

**Example - Custom Reduction:**
```fortran
! Define custom max-abs reduction
!$omp declare reduction(maxabs: real(8): &
!$omp&  omp_out = max(abs(omp_out), abs(omp_in))) &
!$omp&  initializer(omp_priv = 0.0_8)

real(8) :: max_abs
max_abs = 0.0

!$omp parallel do reduction(maxabs:max_abs)
do i = 1, n
  max_abs = a(i)  ! Custom reduction applies max(abs(...))
end do

print *, "Maximum absolute value:", max_abs
```

**Example - Derived Type Reduction:**
```fortran
type :: statistics
  real(8) :: sum
  real(8) :: sum_squares
  integer :: count
end type

! Define reduction for statistics type
!$omp declare reduction(combine_stats: statistics: &
!$omp&  omp_out%sum = omp_out%sum + omp_in%sum, &
!$omp&  omp_out%sum_squares = omp_out%sum_squares + omp_in%sum_squares, &
!$omp&  omp_out%count = omp_out%count + omp_in%count) &
!$omp&  initializer(omp_priv = statistics(0.0_8, 0.0_8, 0))

type(statistics) :: stats
stats = statistics(0.0, 0.0, 0)

!$omp parallel do reduction(combine_stats:stats)
do i = 1, n
  stats%sum = stats%sum + a(i)
  stats%sum_squares = stats%sum_squares + a(i)**2
  stats%count = stats%count + 1
end do

print *, "Mean:", stats%sum / stats%count
print *, "Std dev:", sqrt(stats%sum_squares/stats%count - (stats%sum/stats%count)**2)
```

---

### 5.6 scan Directive

**Name**: scan  
**Category**: executable (separating directive)  
**Allowed in**: `do`, `for`, `simd` constructs  
**Properties**: Must have `reduction` clause on enclosing directive

**Purpose**: Implement scan/prefix operations (cumulative sum, product, etc.)

**Clauses:**
- `inclusive`: Include current iteration in scan result
- `exclusive`: Exclude current iteration from scan result

**Syntax:**
```fortran
!$omp parallel do simd reduction(inscan,+:sum)
do i = 1, n
  sum = sum + a(i)
  !$omp scan inclusive(sum)
  b(i) = sum  ! Cumulative sum up to and including i
end do
```

**Example - Inclusive Scan (Cumulative Sum):**
```fortran
real(8), dimension(10) :: a, b
real(8) :: sum

a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
sum = 0.0

!$omp parallel do simd reduction(inscan,+:sum)
do i = 1, 10
  sum = sum + a(i)
  !$omp scan inclusive(sum)
  b(i) = sum
end do
!$omp end parallel do simd

! Result: b = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55]
```

**Example - Exclusive Scan:**
```fortran
sum = 0.0

!$omp parallel do simd reduction(inscan,+:sum)
do i = 1, 10
  !$omp scan exclusive(sum)
  b(i) = sum
  sum = sum + a(i)
end do

! Result: b = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]
```

---

### 5.7 Data Copying Clauses

#### 5.7.1 copyin Clause

**Name**: copyin  
**Arguments**: `list` (threadprivate variables)  
**Allowed on**: `parallel`

**Purpose**: Copy master thread's threadprivate variable value to all other threads

**Example:**
```fortran
integer, save :: thread_data
!$omp threadprivate(thread_data)

thread_data = 100  ! Set in master thread

!$omp parallel copyin(thread_data)
  ! All threads now have thread_data = 100
  thread_data = thread_data + omp_get_thread_num()
!$omp end parallel

!$omp parallel
  ! Each thread retains its modified value (no copyin)
  print *, "Thread", omp_get_thread_num(), ":", thread_data
!$omp end parallel
```

#### 5.7.2 copyprivate Clause

**Name**: copyprivate  
**Arguments**: `list` (variable list items)  
**Allowed on**: `single`

**Purpose**: Broadcast private variable values from one thread to all others after `single` region

**Example:**
```fortran
integer :: input_value

!$omp parallel private(input_value)
  !$omp single
    read(*,*) input_value  ! Only one thread reads
  !$omp end single copyprivate(input_value)
  ! Now all threads have the input value
  call process(input_value)
!$omp end parallel
```

---

## Data-Sharing Quick Reference Tables

### Variables by Default Data-Sharing

| Variable Type | Default in `parallel` | Default in `task` | Default in `target` |
|---------------|----------------------|-------------------|---------------------|
| Automatic variables (local) | Shared | Firstprivate (orphaned) | Firstprivate (unmapped) |
| Static variables | Shared (predetermined) | Shared | Shared |
| Loop iteration variables | Private (predetermined) | Private (predetermined) | Private (predetermined) |
| Threadprivate variables | Threadprivate | Threadprivate | Threadprivate |
| Formal arguments (by reference) | Shared | Firstprivate (orphaned) | Shared |

### Clause Compatibility Matrix

| Clause | parallel | do/for | task | taskloop | taskgroup | simd | target | teams |
|--------|----------|--------|------|----------|-----------|------|--------|-------|
| `default` | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| `shared` | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| `private` | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| `firstprivate` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| `lastprivate` | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |
| `linear` | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| `reduction` | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ |
| `task_reduction` | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| `in_reduction` | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| `copyin` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `copyprivate` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

### Choosing the Right Data-Sharing Clause

| Situation | Use This Clause | Example |
|-----------|-----------------|---------|
| Variable should be shared by all threads | `shared` | Loop-independent shared data |
| Each thread needs uninitialized scratch variable | `private` | Temporary computation variables |
| Each thread needs initialized copy | `firstprivate` | Loop-independent input parameters |
| Need final iteration value after loop | `lastprivate` | Loop exit index |
| Combining values from all threads (sum, max, etc.) | `reduction` | Computing totals, finding maximum |
| Linear relationship with iteration | `linear` | Pointer arithmetic in SIMD |
| Tasks contributing to reduction | `task_reduction` + `in_reduction` | Tree-based reductions |
| Thread-specific persistent storage | `threadprivate` | Thread-local state across regions |

---

**Last Updated**: Based on OpenMP 5.2 Specification (November 2021)  
**Usage**: This file is automatically loaded by GitHub Copilot for all files in the workspace (`applyTo: **/*`)
