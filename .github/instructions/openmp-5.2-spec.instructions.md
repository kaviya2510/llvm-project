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


---

## CHAPTER 2: INTERNAL CONTROL VARIABLES

An OpenMP implementation must act as if internal control variables (ICVs) control the behavior of an OpenMP program. These ICVs store information such as the number of threads to use for future parallel regions. One copy exists of each ICV per instance of its scope. Possible ICV scopes are: **global**, **device**, **implicit task**, and **data environment**. If an ICV has global scope then one copy exists for the whole program. The ICVs are given values at various times (described below) during the execution of the program. They are initialized by the implementation itself and may be given values through OpenMP environment variables and through calls to OpenMP API routines. The program can retrieve the values of these ICVs only through OpenMP API routines.

For purposes of exposition, this document refers to the ICVs by certain names, but an implementation is not required to use these names or to offer any way to access the variables other than through the ways shown in Section 2.2.

### 2.1 ICV Descriptions

Table 2.1 shows the scope and description of each ICV.

**TABLE 2.1: ICV Scopes and Descriptions**

| ICV | Scope | Description |
|-----|-------|-------------|
| `active-levels-var` | data environment | Number of nested active parallel regions such that all parallel regions are enclosed by the outermost initial task region on the device |
| `affinity-format-var` | device | Controls the thread affinity format when displaying thread affinity |
| `bind-var` | data environment | Controls the binding of OpenMP threads to places; when binding is requested, indicates that the execution environment is advised not to move threads between places; can also provide default thread affinity policies |
| `cancel-var` | global | Controls the desired behavior of the cancel construct and cancellation points |
| `debug-var` | global | Controls whether an OpenMP implementation will collect information that an OMPD library can access to satisfy requests from a tool |
| `def-allocator-var` | implicit task | Controls the memory allocator used by memory allocation routines, directives and clauses that do not specify one explicitly |
| `default-device-var` | data environment | Controls the default target device |
| `display-affinity-var` | global | Controls the display of thread affinity |
| `dyn-var` | data environment | Enables dynamic adjustment of the number of threads used for encountered parallel regions |
| `explicit-task-var` | data environment | Whether a given task is an explicit task |
| `final-task-var` | data environment | Whether a given task is a final task |
| `levels-var` | data environment | Number of nested parallel regions such that all parallel regions are enclosed by the outermost initial task region on the device |
| `max-active-levels-var` | data environment | Controls the maximum number of nested active parallel regions when the innermost parallel region is generated by a given task |
| `max-task-priority-var` | global | Controls the maximum value that can be specified in the priority clause |
| `nteams-var` | device | Controls the number of teams requested for encountered teams regions |
| `nthreads-var` | data environment | Controls the number of threads requested for encountered parallel regions |
| `num-procs-var` | device | The number of processors available on the device |
| `place-partition-var` | implicit task | Controls the place partition available for encountered parallel regions |
| `run-sched-var` | data environment | Controls the schedule used for worksharing-loop regions that specify the runtime schedule kind |
| `stacksize-var` | device | Controls the stack size for threads that the OpenMP implementation creates |
| `target-offload-var` | global | Controls the offloading behavior |
| `team-size-var` | data environment | Size of the current team |
| `teams-thread-limit-var` | device | Controls the maximum number of threads in each contention group that a teams construct creates |
| `thread-limit-var` | data environment | Controls the maximum number of threads that participate in the contention group |
| `thread-num-var` | data environment | Thread number of an implicit task within its binding team |
| `tool-libraries-var` | global | List of absolute paths to tool libraries |
| `tool-var` | global | Indicates that a tool will be registered |
| `tool-verbose-init-var` | global | Controls whether an OpenMP implementation will verbosely log the registration of a tool |
| `wait-policy-var` | device | Controls the desired behavior of waiting threads |

---

### 2.2 ICV Initialization

Table 2.2 shows the ICVs, associated environment variables, and initial values.

**TABLE 2.2: ICV Initial Values**

| ICV | Environment Variable | Initial Value |
|-----|---------------------|---------------|
| `active-levels-var` | (none) | Zero |
| `affinity-format-var` | `OMP_AFFINITY_FORMAT` | Implementation defined |
| `bind-var` | `OMP_PROC_BIND` | Implementation defined |
| `cancel-var` | `OMP_CANCELLATION` | False |
| `debug-var` | `OMP_DEBUG` | disabled |
| `def-allocator-var` | `OMP_ALLOCATOR` | Implementation defined |
| `default-device-var` | `OMP_DEFAULT_DEVICE` | See below |
| `display-affinity-var` | `OMP_DISPLAY_AFFINITY` | False |
| `dyn-var` | `OMP_DYNAMIC` | Implementation defined |
| `explicit-task-var` | (none) | False |
| `final-task-var` | (none) | False |
| `levels-var` | (none) | Zero |
| `max-active-levels-var` | `OMP_MAX_ACTIVE_LEVELS`, `OMP_NESTED`, `OMP_NUM_THREADS`, `OMP_PROC_BIND` | Implementation defined |
| `max-task-priority-var` | `OMP_MAX_TASK_PRIORITY` | Zero |
| `nteams-var` | `OMP_NUM_TEAMS` | Zero |
| `nthreads-var` | `OMP_NUM_THREADS` | Implementation defined |
| `num-procs-var` | (none) | Implementation defined |
| `place-partition-var` | `OMP_PLACES` | Implementation defined |
| `run-sched-var` | `OMP_SCHEDULE` | Implementation defined |
| `stacksize-var` | `OMP_STACKSIZE` | Implementation defined |
| `target-offload-var` | `OMP_TARGET_OFFLOAD` | default |
| `team-size-var` | (none) | One |
| `teams-thread-limit-var` | `OMP_TEAMS_THREAD_LIMIT` | Zero |
| `thread-limit-var` | `OMP_THREAD_LIMIT` | Implementation defined |
| `thread-num-var` | (none) | Zero |
| `tool-libraries-var` | `OMP_TOOL_LIBRARIES` | empty string |
| `tool-var` | `OMP_TOOL` | enabled |
| `tool-verbose-init-var` | `OMP_TOOL_VERBOSE_INIT` | disabled |
| `wait-policy-var` | `OMP_WAIT_POLICY` | Implementation defined |

#### Device-Specific Environment Variables

If an ICV has an associated environment variable and that ICV does not have global scope, then the ICV has a set of associated device-specific environment variables that extend the associated environment variable with the following syntax:

```
<ENVIRONMENT VARIABLE>_DEV[_<device>]
```

where `<ENVIRONMENT VARIABLE>` is the associated environment variable and `<device>` is the device number as specified in the device clause (see Section 13.2).

#### Semantics

- The initial value of `dyn-var` is implementation defined if the implementation supports dynamic adjustment of the number of threads; otherwise, the initial value is false.
- If `target-offload-var` is mandatory and the number of non-host devices is zero then the `default-device-var` is initialized to `omp_invalid_device`. Otherwise, the initial value is an implementation-defined non-negative integer that is less than or, if `target-offload-var` is not mandatory, equal to `omp_get_initial_device()`.
- The value of the `nthreads-var` ICV is a list.
- The value of the `bind-var` ICV is a list.

The host and non-host device ICVs are initialized before any OpenMP API construct or OpenMP API routine executes. After the initial values are assigned, the values of any OpenMP environment variables that were set by the user are read and the associated ICVs are modified accordingly. If no `<device>` number is specified on the device-specific environment variable then the value is applied to all non-host devices.

#### Cross References
- `OMP_AFFINITY_FORMAT`, see Section 21.2.5
- `OMP_ALLOCATOR`, see Section 21.5.1
- `OMP_CANCELLATION`, see Section 21.2.6
- `OMP_DEBUG`, see Section 21.4.1
- `OMP_DEFAULT_DEVICE`, see Section 21.2.7
- `OMP_DISPLAY_AFFINITY`, see Section 21.2.4
- `OMP_DYNAMIC`, see Section 21.1.1
- `OMP_MAX_ACTIVE_LEVELS`, see Section 21.1.4
- `OMP_MAX_TASK_PRIORITY`, see Section 21.2.9
- `OMP_NESTED` (Deprecated), see Section 21.1.5
- `OMP_NUM_TEAMS`, see Section 21.6.1
- `OMP_NUM_THREADS`, see Section 21.1.2
- `OMP_PLACES`, see Section 21.1.6
- `OMP_PROC_BIND`, see Section 21.1.7
- `OMP_SCHEDULE`, see Section 21.2.1
- `OMP_STACKSIZE`, see Section 21.2.2
- `OMP_TARGET_OFFLOAD`, see Section 21.2.8
- `OMP_TEAMS_THREAD_LIMIT`, see Section 21.6.2
- `OMP_THREAD_LIMIT`, see Section 21.1.3
- `OMP_TOOL`, see Section 21.3.1
- `OMP_TOOL_LIBRARIES`, see Section 21.3.2
- `OMP_WAIT_POLICY`, see Section 21.2.3

---

### 2.3 Modifying and Retrieving ICV Values

Table 2.3 shows methods for modifying and retrieving the ICV values. If (none) is listed for an ICV, the OpenMP API does not support its modification or retrieval. Calls to OpenMP API routines retrieve or modify data environment scoped ICVs in the data environment of their binding tasks.

**TABLE 2.3: Ways to Modify and to Retrieve ICV Values**

| ICV | Ways to Modify Value | Ways to Retrieve Value |
|-----|---------------------|------------------------|
| `active-levels-var` | (none) | `omp_get_active_level` |
| `affinity-format-var` | `omp_set_affinity_format` | `omp_get_affinity_format` |
| `bind-var` | (none) | `omp_get_proc_bind` |
| `cancel-var` | (none) | `omp_get_cancellation` |
| `debug-var` | (none) | (none) |
| `def-allocator-var` | `omp_set_default_allocator` | `omp_get_default_allocator` |
| `default-device-var` | `omp_set_default_device` | `omp_get_default_device` |
| `display-affinity-var` | (none) | (none) |
| `dyn-var` | `omp_set_dynamic` | `omp_get_dynamic` |
| `explicit-task-var` | (none) | `omp_in_explicit_task` |
| `final-task-var` | (none) | `omp_in_final` |
| `levels-var` | (none) | `omp_get_level` |
| `max-active-levels-var` | `omp_set_max_active_levels`, `omp_set_nested` | `omp_get_max_active_levels` |
| `max-task-priority-var` | (none) | `omp_get_max_task_priority` |
| `nteams-var` | `omp_set_num_teams` | `omp_get_max_teams` |
| `nthreads-var` | `omp_set_num_threads` | `omp_get_max_threads` |
| `num-procs-var` | (none) | `omp_get_num_procs` |
| `place-partition-var` | (none) | `omp_get_partition_num_places`, `omp_get_partition_place_nums`, `omp_get_place_num_procs`, `omp_get_place_proc_ids` |
| `run-sched-var` | `omp_set_schedule` | `omp_get_schedule` |
| `stacksize-var` | (none) | (none) |
| `target-offload-var` | (none) | (none) |
| `team-size-var` | (none) | `omp_get_num_threads` |
| `teams-thread-limit-var` | `omp_set_teams_thread_limit` | `omp_get_teams_thread_limit` |
| `thread-limit-var` | `thread_limit` clause | `omp_get_thread_limit` |
| `thread-num-var` | (none) | `omp_get_thread_num` |
| `tool-libraries-var` | (none) | (none) |
| `tool-var` | (none) | (none) |
| `tool-verbose-init-var` | (none) | (none) |
| `wait-policy-var` | (none) | (none) |

#### Semantics

- The value of the `bind-var` ICV is a list. The runtime call `omp_get_proc_bind` retrieves the value of the first element of this list.
- The value of the `nthreads-var` ICV is a list. The runtime call `omp_set_num_threads` sets the value of the first element of this list, and `omp_get_max_threads` retrieves the value of the first element of this list.
- Detailed values in the `place-partition-var` ICV are retrieved using the listed runtime calls.
- The `thread_limit` clause sets the `thread-limit-var` ICV for the region of the construct on which it appears.

#### Cross References
- `omp_get_active_level`, see Section 18.2.20
- `omp_get_affinity_format`, see Section 18.3.9
- `omp_get_cancellation`, see Section 18.2.8
- `omp_get_default_allocator`, see Section 18.13.5
- `omp_get_default_device`, see Section 18.7.3
- `omp_get_dynamic`, see Section 18.2.7
- `omp_get_level`, see Section 18.2.17
- `omp_get_max_active_levels`, see Section 18.2.16
- `omp_get_max_task_priority`, see Section 18.5.1
- `omp_get_max_teams`, see Section 18.4.4
- `omp_get_max_threads`, see Section 18.2.3
- `omp_get_num_procs`, see Section 18.7.1
- `omp_get_num_threads`, see Section 18.2.2
- `omp_get_partition_num_places`, see Section 18.3.6
- `omp_get_partition_place_nums`, see Section 18.3.7
- `omp_get_place_num_procs`, see Section 18.3.3
- `omp_get_place_proc_ids`, see Section 18.3.4
- `omp_get_proc_bind`, see Section 18.3.1
- `omp_get_schedule`, see Section 18.2.12
- `omp_get_supported_active_levels`, see Section 18.2.14
- `omp_get_teams_thread_limit`, see Section 18.4.6
- `omp_get_thread_limit`, see Section 18.2.13
- `omp_get_thread_num`, see Section 18.2.4
- `omp_in_final`, see Section 18.5.3
- `omp_set_affinity_format`, see Section 18.3.8
- `omp_set_default_allocator`, see Section 18.13.4
- `omp_set_default_device`, see Section 18.7.2
- `omp_set_dynamic`, see Section 18.2.6
- `omp_set_max_active_levels`, see Section 18.2.15
- `omp_set_nested` (Deprecated), see Section 18.2.9
- `omp_set_num_teams`, see Section 18.4.3
- `omp_set_num_threads`, see Section 18.2.1
- `omp_set_schedule`, see Section 18.2.11
- `omp_set_teams_thread_limit`, see Section 18.4.5
- `thread_limit` clause, see Section 13.3

---

### 2.4 How the Per-Data Environment ICVs Work

When a `task` construct, a `parallel` construct or a `teams` construct is encountered, each generated task inherits the values of the data environment scoped ICVs from each generating task's ICV values.

When a `parallel` construct is encountered, the value of each ICV with implicit task scope is inherited from the implicit binding task of the generating task unless otherwise specified.

When a `task` construct is encountered, the generated task inherits the value of `nthreads-var` from the generating task's `nthreads-var` value. When a `parallel` construct is encountered, and the generating task's `nthreads-var` list contains a single element, the generated implicit tasks inherit that list as the value of `nthreads-var`. When a `parallel` construct is encountered, and the generating task's `nthreads-var` list contains multiple elements, the generated implicit tasks inherit the value of `nthreads-var` as the list obtained by deletion of the first element from the generating task's `nthreads-var` value. The `bind-var` ICV is handled in the same way as the `nthreads-var` ICV.

When a target task executes an active target region, the generated initial task uses the values of the data environment scoped ICVs from the device data environment ICV values of the device that will execute the region.

When a target task executes an inactive target region, the generated initial task uses the values of the data environment scoped ICVs from the data environment of the task that encountered the target construct.

If a `target` construct with a `thread_limit` clause is encountered, the `thread-limit-var` ICV from the data environment of the generated initial task is instead set to an implementation defined value between one and the value specified in the clause.

If a `target` construct with no `thread_limit` clause is encountered, the `thread-limit-var` ICV from the data environment of the generated initial task is set to an implementation defined value that is greater than zero.

If a `teams` construct with a `thread_limit` clause is encountered, the `thread-limit-var` ICV from the data environment of the initial task for each team is instead set to an implementation defined value between one and the value specified in the clause.

If a `teams` construct with no `thread_limit` clause is encountered, the `thread-limit-var` ICV from the data environment of the initial task of each team is set to an implementation defined value that is greater than zero and does not exceed `teams-thread-limit-var`, if `teams-thread-limit-var` is greater than zero.

When encountering a worksharing-loop region for which the runtime schedule kind is specified, all implicit task regions that constitute the binding parallel region must have the same value for `run-sched-var` in their data environments. Otherwise, the behavior is unspecified.

---

### 2.5 ICV Override Relationships

Table 2.4 shows the override relationships among construct clauses and ICVs. The table only lists ICVs that can be overridden by a clause.

**TABLE 2.4: ICV Override Relationships**

| ICV | construct clause, if used |
|-----|---------------------------|
| `bind-var` | `proc_bind` |
| `def-allocator-var` | `allocate`, `allocator` |
| `nteams-var` | `num_teams` |
| `nthreads-var` | `num_threads` |
| `run-sched-var` | `schedule` |
| `teams-thread-limit-var` | `thread_limit` |

#### Semantics

- The `num_threads` clause overrides the value of the first element of the `nthreads-var` ICV.
- If a `schedule` clause specifies a modifier then that modifier overrides any modifier that is specified in the `run-sched-var` ICV.
- If `bind-var` is not set to false then the `proc_bind` clause overrides the value of the first element of the `bind-var` ICV; otherwise, the `proc_bind` clause has no effect.

#### Cross References
- `allocate` clause, see Section 6.6
- `allocator` clause, see Section 6.4
- `num_teams` clause, see Section 10.2.1
- `num_threads` clause, see Section 10.1.2
- `proc_bind` clause, see Section 10.1.4
- `schedule` clause, see Section 11.5.3
- `thread_limit` clause, see Section 13.3

---


---

## CHAPTER 3: DIRECTIVE AND CONSTRUCT SYNTAX

This chapter describes the syntax of OpenMP directives, clauses and any related base language code. OpenMP directives are specified with various base-language mechanisms that allow compilers to ignore OpenMP directives and conditionally compiled code if support of the OpenMP API is not provided or enabled. A compliant implementation must provide an option or interface that ensures that underlying support of all OpenMP directives and OpenMP conditional compilation mechanisms is enabled. In the remainder of this document, the phrase **OpenMP compilation** is used to mean a compilation with these OpenMP features enabled.

### General Restrictions

The following restrictions apply to OpenMP directives:

- Unless otherwise specified, a program must not depend on any ordering of the evaluations of the expressions that appear in the clauses specified on a directive.
- Unless otherwise specified, a program must not depend on any side effects of the evaluations of the expressions that appear in the clauses specified on a directive.

**Restrictions on explicit OpenMP regions** (that arise from executable directives):

**C/C++:**
- A throw executed inside a region that arises from a thread-limiting directive must cause execution to resume within the same region, and the same thread that threw the exception must catch it. If the directive is also exception-aborting then whether the exception is caught or the throw results in runtime error termination is implementation defined.

**Fortran:**
- A directive may not appear in a pure procedure unless it is pure.
- A directive may not appear in a WHERE, FORALL or DO CONCURRENT construct.
- If more than one image is executing the program, any image control statement, ERROR STOP statement, FAIL IMAGE statement, collective subroutine call or access to a coindexed object that appears in an explicit OpenMP region will result in unspecified behavior.

---

### 3.1 Directive Format

An OpenMP directive is specified with a **directive-specification** that consists of the directive-specifier and any clauses that may optionally be associated with the OpenMP directive:

```
directive-specifier [[,] clause[ [,] clause] ... ]
```

The **directive-specifier** is:
- `directive-name` (for basic directives)
- `directive-name(directive-arguments)` (for argument-modified directives)

#### Directive Categories

OpenMP directives are categorized as follows:

1. **Declarative directives**: Specify variables, procedures, or other entities that should have certain properties
2. **Executable directives**: Appear in executable regions and affect execution behavior
3. **Utility directives**: Provide additional functionality (e.g., error handling)
4. **Informational directives**: Provide hints to the implementation

#### Syntax Forms

**C/C++ Syntax:**
```c
#pragma omp directive-specification new-line
```

**Fortran Free-Form Syntax:**
```fortran
!$omp directive-specification
```

**Fortran Fixed-Form Syntax:**
```fortran
!$omp directive-specification
c$omp directive-specification
*$omp directive-specification
```

#### Stand-Alone Directives

A **stand-alone directive** is an executable directive that has no associated executable user code. The following are stand-alone directives:

- `barrier`
- `taskwait`
- `taskyield`
- `flush`
- `cancel`
- `cancellation point`
- `interop`

**Syntax:**
```c
// C/C++
#pragma omp directive-specification new-line
```

```fortran
! Fortran
!$omp directive-specification
```

#### Constructs

A **construct** is an OpenMP directive (and for some directives, the associated statement, loop or structured block, if any). Most constructs can be classified into categories:

- **Loop constructs**: Apply to loops (`for`, `do`, `simd`, `distribute`, etc.)
- **Block constructs**: Apply to structured blocks (`parallel`, `task`, `target`, `teams`, etc.)
- **Combined/Composite constructs**: Combine multiple directives (`parallel for`, `target teams`, etc.)

**Example - Loop Construct:**
```fortran
!$omp parallel do private(i, temp)
do i = 1, n
  temp = compute(a(i))
  b(i) = temp * 2
end do
!$omp end parallel do
```

**Example - Block Construct:**
```c
#pragma omp parallel private(tid)
{
  tid = omp_get_thread_num();
  printf("Hello from thread %d\n", tid);
}
```

#### Combined and Composite Constructs

- **Combined construct**: Shorthand for specifying one construct immediately nested inside another construct
  - Examples: `parallel for`, `parallel sections`, `target teams`
  
- **Composite construct**: Combined construct where some clauses are separated and applied to two or more constructs
  - Example: `target teams distribute parallel for` can have clauses for `target`, `teams`, `distribute`, and `for`

**Common Combined Constructs:**

| Combined Construct | Equivalent To |
|-------------------|---------------|
| `parallel for` / `parallel do` | `parallel` + `for`/`do` |
| `parallel sections` | `parallel` + `sections` |
| `parallel workshare` | `parallel` + `workshare` (Fortran only) |
| `target teams` | `target` + `teams` |
| `target parallel` | `target` + `parallel` |
| `target parallel for` / `target parallel do` | `target` + `parallel` + `for`/`do` |
| `teams distribute` | `teams` + `distribute` |
| `distribute parallel for` / `distribute parallel do` | `distribute` + `parallel` + `for`/`do` |

---

### 3.1.1 Fixed Source Form Directives (Fortran)

In Fortran fixed source form, OpenMP directives have the following syntax:

```fortran
sentinel directive-specification
```

Where **sentinel** is one of:
- `!$omp`
- `c$omp`
- `*$omp`

**Rules:**
- The sentinel must start in column 1
- The sentinel must appear as a single word with no intervening characters
- Fortran fixed form line length restrictions apply
- Initial directive lines must have a space or zero in column 6
- Continuation lines must have a character other than space or zero in column 6

**Example:**
```fortran
c$omp parallel do private(i, j, temp)
      do i = 1, n
        do j = 1, m
          temp = a(i,j) + b(i,j)
          c(i,j) = temp * 2.0
        end do
      end do
c$omp end parallel do
```

---

### 3.1.2 Free Source Form Directives (Fortran)

In Fortran free source form, OpenMP directives have the following syntax:

```fortran
sentinel directive-specification
```

Where **sentinel** is:
- `!$omp`

**Rules:**
- The sentinel can appear in any column but must be preceded only by white space
- The sentinel must appear as a single word
- Fortran free form line length restrictions apply
- Continuation is indicated by an ampersand (&) at the end of the line
- Continued lines can have an optional ampersand at the beginning after the sentinel

**Example:**
```fortran
!$omp parallel do private(i, j, temp) &
!$omp&            shared(a, b, c) &
!$omp&            reduction(+:sum)
do i = 1, n
  do j = 1, m
    temp = a(i,j) + b(i,j)
    c(i,j) = temp * 2.0
    sum = sum + c(i,j)
  end do
end do
!$omp end parallel do
```

---

### 3.2 Clause Format

Clauses modify the behavior of directives. A clause consists of a clause name and, optionally, arguments:

```
clause-name
clause-name(argument-list)
clause-name(argument-list : argument-list)
```

#### Clause Properties

Clauses can have various properties:

**Uniqueness:**
- **unique**: Can appear at most once on a directive
- **non-unique**: Can appear multiple times

**Category:**
- **data-environment**: Affects how variables are accessed
- **data-sharing**: Controls whether variables are shared or private
- **synchronization**: Controls synchronization behavior
- **execution**: Controls how execution proceeds

**Clause Modifiers:**

Some clauses accept modifiers that change their behavior:

```fortran
clause-name(modifier, modifier : argument-list)
```

**Examples:**
```fortran
! reduction with task modifier
!$omp task reduction(task, +:sum)

! schedule with modifier
!$omp do schedule(static, 16)

! map with modifier
!$omp target map(to: a) map(from: b)
```

---

### 3.2.1 OpenMP Argument Lists

Many clauses accept argument lists that specify variables or expressions. The syntax varies by clause type.

**List Item Types:**

1. **Variable names**: Simple variables
   ```fortran
   private(x, y, z)
   ```

2. **Array elements**: Individual array elements or array sections
   ```fortran
   map(a(1:10))
   ```

3. **Derived-type components**: Structure members
   ```c
   map(obj.field)
   ```

4. **Array sections**: Contiguous or strided sections
   ```fortran
   map(array(1:n:2))  ! Elements 1, 3, 5, ..., n
   ```

**C/C++ Examples:**
```c
#pragma omp parallel private(x, y) shared(a, b) reduction(+:sum)
#pragma omp target map(to: input[0:size]) map(from: output[0:size])
```

**Fortran Examples:**
```fortran
!$omp parallel private(x, y) shared(a, b) reduction(+:sum)
!$omp target map(to: input(1:size)) map(from: output(1:size))
```

---

### 3.2.2 Reserved Locators

OpenMP defines special reserved locators that can be used in certain clauses:

| Locator | Meaning | Use In |
|---------|---------|--------|
| `omp_all_memory` | Represents all memory accessible to the device | `depend`, `doacross` clauses |
| `omp_default_mem_space` | Default memory space | Memory allocators |
| `omp_large_cap_mem_space` | Large capacity memory | Memory allocators |
| `omp_const_mem_space` | Constant memory | Memory allocators |
| `omp_high_bw_mem_space` | High bandwidth memory | Memory allocators |
| `omp_low_lat_mem_space` | Low latency memory | Memory allocators |

**Example:**
```c
#pragma omp task depend(inout: omp_all_memory)
{
  // This task depends on all memory
}
```

---

### 3.2.3 OpenMP Operations

Certain clauses accept **OpenMP operations** as arguments, which are either:

1. **Reduction identifiers**: `+`, `*`, `-`, `min`, `max`, `&`, `|`, `^`, `&&`, `||`, `.and.`, `.or.`, etc.
2. **User-defined operations**: Defined with `declare reduction`

**Examples:**
```fortran
!$omp parallel reduction(+:sum) reduction(max:max_val)
!$omp parallel reduction(my_custom_op:result)
```

---

### 3.2.4 Array Shaping

The **array shaping operator** allows programmers to specify the shape of array data when the shape is not known at compile time.

**Syntax:**
```c
([ type-qualifier-list ] [ * ] )( expression-list )
```

**C/C++ Example:**
```c
int *ptr = get_data();
int n = get_size();

#pragma omp target map(tofrom: ([n][n])ptr[0:n*n])
{
  // ptr is treated as n×n array
}
```

**Fortran:** Fortran does not require explicit array shaping due to its array descriptor support.

---

### 3.2.5 Array Sections

**Array sections** specify contiguous or strided sections of arrays for data mapping and privatization.

**C/C++ Syntax:**
```
array-name [ lower-bound : length ]
array-name [ lower-bound : length : stride ]
```

**Fortran Syntax:**
```
array-name ( lower-bound : upper-bound )
array-name ( lower-bound : upper-bound : stride )
```

**Examples:**

**C/C++:**
```c
// Map first 100 elements
#pragma omp target map(to: array[0:100])

// Map elements 10-19
#pragma omp target map(to: array[10:10])

// Map every other element from 0 to 98
#pragma omp target map(to: array[0:50:2])
```

**Fortran:**
```fortran
! Map elements 1 to 100
!$omp target map(to: array(1:100))

! Map elements 10 to 19
!$omp target map(to: array(10:19))

! Map every other element from 1 to 99
!$omp target map(to: array(1:99:2))
```

**Multi-dimensional Arrays:**

**C/C++:**
```c
// Map 2D section
#pragma omp target map(to: matrix[0:rows][0:cols])
```

**Fortran:**
```fortran
! Map 2D section
!$omp target map(to: matrix(1:rows, 1:cols))
```

---

### 3.2.6 iterator Modifier

The **iterator** modifier allows iteration over a range of values within a clause.

**Syntax:**
```
iterator([ iterators-definition ])
```

**Iterators Definition:**
```
identifier = range-specification [, identifier = range-specification] ...
```

**Range Specification:**
```
begin : end [ : step ]
```

**Example - Using iterator in depend clause:**

**C/C++:**
```c
#pragma omp task depend(iterator(i=0:n), in: a[i])
{
  // Task depends on a[0], a[1], ..., a[n-1]
}
```

**Fortran:**
```fortran
!$omp task depend(iterator(i=1:n), in: a(i))
  ! Task depends on a(1), a(2), ..., a(n)
!$omp end task
```

**Example - Multiple iterators:**
```c
#pragma omp task depend(iterator(i=0:n, j=0:m), in: matrix[i][j])
{
  // Task depends on all elements matrix[i][j] for i in [0,n), j in [0,m)
}
```

**Common Use Cases:**
- Expressing complex task dependencies
- Mapping irregular data patterns
- Specifying non-contiguous array sections in dependencies

---

### 3.3 Conditional Compilation

OpenMP provides mechanisms for conditional compilation that allows code to be compiled differently based on whether OpenMP support is enabled.

**Predefined Macro:**
- `_OPENMP`: Defined when OpenMP compilation is enabled
  - Value: `YYYYMM` where YYYY is year, MM is month of specification
  - OpenMP 5.2 (November 2021): `_OPENMP = 202111`

**C/C++ Example:**
```c
#ifdef _OPENMP
  #include <omp.h>
  int num_threads = omp_get_max_threads();
#else
  int num_threads = 1;
#endif
```

**Fortran - Conditional Compilation Sentinels:**

Fortran uses special sentinels for conditional compilation.

---

### 3.3.1 Fixed Source Form Conditional Compilation Sentinels (Fortran)

In Fortran fixed source form, conditional compilation uses sentinels:

**Sentinels:**
- `!$` - Lines compiled only when OpenMP is enabled
- `c$` - Lines compiled only when OpenMP is enabled  
- `*$` - Lines compiled only when OpenMP is enabled

**Rules:**
- Must start in column 1
- Followed by a space or zero in column 6 for initial line
- Followed by a character other than space/zero in column 6 for continuation

**Example:**
```fortran
      program test
      integer :: nthreads
!$    nthreads = omp_get_max_threads()
      print *, 'Number of threads:', nthreads
      end program
```

Without OpenMP: The line is treated as a comment.
With OpenMP: The line is compiled as `nthreads = omp_get_max_threads()`

---

### 3.3.2 Free Source Form Conditional Compilation Sentinel (Fortran)

In Fortran free source form, conditional compilation uses the `!$` sentinel:

**Sentinel:**
- `!$` - Lines compiled only when OpenMP is enabled

**Rules:**
- Can appear in any column preceded only by white space
- The rest of the line is treated as a Fortran statement when OpenMP is enabled
- Treated as a comment when OpenMP is disabled

**Example:**
```fortran
program test
  integer :: nthreads
!$ nthreads = omp_get_max_threads()
  print *, 'Number of threads:', nthreads
end program
```

**Using with USE statements:**
```fortran
program test
!$ use omp_lib
  implicit none
  integer :: tid
!$ tid = omp_get_thread_num()
  print *, 'Thread ID:', tid
end program
```

---

### 3.4 if Clause

**Name**: `if`  
**Arguments**: `directive-name-modifier : scalar-expression` or `scalar-expression`  
**Properties**: unique for each directive-name-modifier

**Purpose**: Control whether a construct executes in parallel or serialized mode based on a runtime condition.

**Syntax:**
```fortran
if([ directive-name-modifier : ] scalar-logical-expression)
```

**Directive Name Modifiers:**
- `parallel`: For parallel regions
- `task`: For task constructs
- `taskloop`: For taskloop constructs
- `target`: For target regions
- `target data`: For target data regions
- `target enter data`: For target enter data
- `target exit data`: For target exit data
- `target update`: For target update

**Semantics:**
- If the expression evaluates to true, the construct executes normally
- If the expression evaluates to false, the implementation may serialize execution or skip the construct (implementation-defined behavior)

**Examples:**

**Basic if clause:**
```fortran
!$omp parallel if(n > threshold)
  ! Execute in parallel only if n > threshold
  call process_data()
!$omp end parallel
```

**With modifier:**
```fortran
!$omp target if(target: use_device) if(parallel: n > 100)
!$omp teams distribute parallel do
  ! Target executes on device only if use_device is true
  ! Parallel executes in parallel only if n > 100
  do i = 1, n
    a(i) = compute(i)
  end do
!$omp end target
```

**Multiple if clauses for combined constructs:**
```c
#pragma omp target teams distribute parallel for \
            if(target: offload_enabled) \
            if(parallel: n > 1000)
for (int i = 0; i < n; i++) {
  a[i] = compute(i);
}
```

---

### 3.5 destroy Clause

**Name**: `destroy`  
**Arguments**: `destroy-expression` (optional)  
**Properties**: unique

**Purpose**: Specify that an object should be destroyed at a particular point.

**Syntax:**
```fortran
destroy [( destroy-expression )]
```

**Used with:**
- `depobj` directive (to destroy a depend object)

**Example:**
```c
omp_depend_t dep_obj;

// Initialize depend object
#pragma omp depobj(dep_obj) depend(in: a)

// Use the depend object
#pragma omp task depend(depobj: dep_obj)
{
  process(a);
}

// Destroy the depend object
#pragma omp depobj(dep_obj) destroy
```

**Fortran Example:**
```fortran
integer(omp_depend_kind) :: dep_obj

! Initialize depend object
!$omp depobj(dep_obj) depend(in: a)

! Use the depend object
!$omp task depend(depobj: dep_obj)
  call process(a)
!$omp end task

! Destroy the depend object
!$omp depobj(dep_obj) destroy
```

---

## Summary: Key Points for Chapter 3

### Directive Syntax
- **C/C++**: `#pragma omp directive-specification`
- **Fortran**: `!$omp directive-specification` (free form) or `!$omp`, `c$omp`, `*$omp` (fixed form)

### Main Directive Categories
1. **Declarative**: `declare reduction`, `declare simd`, `declare target`, `threadprivate`
2. **Executable**: Most directives that control execution
3. **Stand-alone**: `barrier`, `taskwait`, `taskyield`, `flush`, `cancel`, `cancellation point`
4. **Constructs**: Directives with associated code blocks or loops

### Common Combined Constructs
- `parallel for` / `parallel do`
- `parallel sections`
- `target parallel`
- `target teams distribute parallel for`

### Important Clauses
- **if**: Conditional execution
- **private/firstprivate/lastprivate**: Variable privatization
- **shared**: Variable sharing
- **reduction**: Reduction operations
- **map**: Data mapping for target regions
- **depend**: Task dependencies

### Array Sections
- **C/C++**: `array[start:length:stride]`
- **Fortran**: `array(start:end:stride)`

### Conditional Compilation
- **C/C++**: Use `#ifdef _OPENMP`
- **Fortran**: Use `!$` sentinel

---


---

## CHAPTER 4: BASE LANGUAGE FORMATS AND RESTRICTIONS

This section defines concepts and restrictions on base language code used in OpenMP. The concepts help support base language neutrality for OpenMP directives and their associated semantics.

---

### 4.1 OpenMP Types and Identifiers

OpenMP defines specific types and identifiers for use in API routines and directives.

#### Predefined Types

**C/C++ Types:**
- `omp_lock_t` - Lock variable type
- `omp_nest_lock_t` - Nested lock variable type
- `omp_sync_hint_t` - Synchronization hint type
- `omp_allocator_handle_t` - Memory allocator handle
- `omp_memspace_handle_t` - Memory space handle
- `omp_depend_t` - Depend object type
- `omp_event_handle_t` - Event handle type
- `omp_interop_t` - Interoperability object type
- `omp_proc_bind_t` - Process binding policy type
- `omp_sched_t` - Schedule kind type

**Fortran Types:**
- `integer(kind=omp_lock_kind)` - Lock variable type
- `integer(kind=omp_nest_lock_kind)` - Nested lock variable type
- `integer(kind=omp_sync_hint_kind)` - Synchronization hint type
- `integer(kind=omp_allocator_handle_kind)` - Memory allocator handle
- `integer(kind=omp_memspace_handle_kind)` - Memory space handle
- `integer(kind=omp_depend_kind)` - Depend object type
- `integer(kind=omp_event_handle_kind)` - Event handle type
- `integer(kind=omp_interop_kind)` - Interoperability object type
- `integer(kind=omp_proc_bind_kind)` - Process binding policy type
- `integer(kind=omp_sched_kind)` - Schedule kind type

#### Named Constants

OpenMP defines numerous named constants for various purposes:

**Memory Allocators:**
- `omp_default_mem_alloc` - Default memory allocator
- `omp_large_cap_mem_alloc` - Large capacity memory allocator
- `omp_const_mem_alloc` - Constant memory allocator
- `omp_high_bw_mem_alloc` - High bandwidth memory allocator
- `omp_low_lat_mem_alloc` - Low latency memory allocator
- `omp_cgroup_mem_alloc` - Cgroup memory allocator
- `omp_pteam_mem_alloc` - Parallel team memory allocator
- `omp_thread_mem_alloc` - Thread memory allocator

**Memory Spaces:**
- `omp_default_mem_space` - Default memory space
- `omp_large_cap_mem_space` - Large capacity memory space
- `omp_const_mem_space` - Constant memory space
- `omp_high_bw_mem_space` - High bandwidth memory space
- `omp_low_lat_mem_space` - Low latency memory space

**Synchronization Hints:**
- `omp_sync_hint_none` - No hint
- `omp_sync_hint_uncontended` - Lock is usually uncontended
- `omp_sync_hint_contended` - Lock is usually contended
- `omp_sync_hint_nonspeculative` - Non-speculative
- `omp_sync_hint_speculative` - Speculative

**Schedule Types:**
- `omp_sched_static` - Static schedule
- `omp_sched_dynamic` - Dynamic schedule
- `omp_sched_guided` - Guided schedule
- `omp_sched_auto` - Auto schedule

**Procedure Binding Types:**
- `omp_proc_bind_false` - No binding
- `omp_proc_bind_true` - Binding enabled
- `omp_proc_bind_primary` - Bind to primary thread
- `omp_proc_bind_master` - Bind to master thread (deprecated)
- `omp_proc_bind_close` - Bind close to parent
- `omp_proc_bind_spread` - Spread across places

---

### 4.2 OpenMP Stylized Expressions

**Stylized expressions** are restricted forms of expressions that OpenMP implementations can analyze and optimize. These are primarily used in loop bounds and related constructs.

#### Requirements

A stylized expression:
- Must be loop-invariant (not change during loop execution)
- Must have a well-defined value
- Should not have side effects that affect other loop iterations

**Examples:**

**Valid Stylized Expressions:**
```fortran
! Loop bounds
do i = 1, n          ! n is stylized expression
do i = start, end    ! start and end are stylized
do i = 1, 2*n + 5    ! Arithmetic expression is stylized
```

**Invalid (not stylized):**
```fortran
! Side effects
do i = 1, get_and_increment_n()  ! Function has side effects
```

---

### 4.3 Structured Blocks

A **structured block** is a block of executable statements with:
- A single entry point at the top
- A single exit at the bottom
- No branching into or out of the block (except for certain exceptions)

#### General Structured Block Rules

**Allowed:**
- Normal sequential execution
- Calls to procedures
- OpenMP directives within the block
- Exceptions/error handling (with restrictions)

**Not Allowed:**
- `goto` statements that branch out of the block
- `return` statements that exit the procedure
- Branching to labels outside the block
- C++ `throw` that exits the block (with exceptions)

**C/C++ Example:**
```c
#pragma omp parallel
{
  // Structured block
  int local_var = compute();
  process(local_var);
  // OK: Normal exit at bottom
}
```

**Fortran Example:**
```fortran
!$omp parallel
  ! Structured block
  local_var = compute()
  call process(local_var)
  ! OK: Normal exit at bottom
!$omp end parallel
```

**Invalid Examples:**
```c
#pragma omp parallel
{
  if (condition) {
    goto outside;  // ERROR: Branching out of block
  }
  process();
}
outside:
  // ...
```

---

### 4.3.1 OpenMP Context-Specific Structured Blocks

Some OpenMP constructs require specific types of structured blocks with additional restrictions.

#### 4.3.1.1 OpenMP Allocator Structured Blocks

Used with `allocate` directives and clauses. The block must:
- Contain memory allocation operations
- Follow OpenMP memory allocation rules
- Use specified allocators consistently

**Example:**
```c
#pragma omp allocate(array) allocator(omp_high_bw_mem_alloc)
double *array = malloc(n * sizeof(double));
```

#### 4.3.1.2 OpenMP Function Dispatch Structured Blocks

Used with `dispatch` construct for dynamic function selection.

**Example:**
```c
#pragma omp dispatch device(device_id) 
{
  result = compute_variant(data);
}
```

#### 4.3.1.3 OpenMP Atomic Structured Blocks

Used with `atomic` construct. The block must contain:
- Exactly one atomic operation
- No other OpenMP directives
- Simple memory access pattern

**Examples:**

**Atomic Update:**
```fortran
!$omp atomic
x = x + delta
```

**Atomic Capture:**
```c
#pragma omp atomic capture
{
  v = x;
  x = x + 1;
}
```

**Restrictions:**
- Only one statement allowed (or two for capture)
- Statement must be one of allowed atomic operations
- No function calls (except allowed operations like `+`, `-`, `*`, `/`)

---

### 4.4 Loop Concepts

This section defines how OpenMP interprets and processes loops.

---

### 4.4.1 Canonical Loop Nest Form

For OpenMP to parallelize a loop, it must be in **canonical form**:

**Canonical Loop Requirements:**

**C/C++ Canonical Form:**
```c
for (init-expr; test-expr; incr-expr) {
  // loop body
}
```

Where:
- `init-expr` is one of:
  - `var = lb` (simple assignment)
  - `integer-type var = lb` (declaration with initialization)
- `test-expr` is one of:
  - `var relational-op b` where relational-op is `<`, `<=`, `>`, `>=`
- `incr-expr` is one of:
  - `++var`, `var++`, `--var`, `var--`
  - `var += incr`, `var -= incr`, `var = var + incr`, `var = incr + var`, `var = var - incr`
- `var` must be integer type
- `lb`, `b`, and `incr` must be loop-invariant

**Fortran Canonical Form:**
```fortran
do var = lb, ub [, step]
  ! loop body
end do
```

Where:
- `var` is integer type
- `lb`, `ub`, and `step` are loop-invariant integer expressions
- `step` must not be zero

**Examples:**

**Valid Canonical Loops:**

**C/C++:**
```c
// Simple canonical loop
for (int i = 0; i < n; i++) {
  a[i] = b[i] + c[i];
}

// With stride
for (int i = 0; i < n; i += 2) {
  a[i] = b[i] * 2;
}

// Decreasing loop
for (int i = n-1; i >= 0; i--) {
  process(i);
}
```

**Fortran:**
```fortran
! Simple canonical loop
do i = 1, n
  a(i) = b(i) + c(i)
end do

! With stride
do i = 1, n, 2
  a(i) = b(i) * 2
end do

! Decreasing loop
do i = n, 1, -1
  call process(i)
end do
```

**Non-Canonical (Invalid) Loops:**

```c
// NOT canonical - complex test
for (int i = 0; i < n && i < m; i++) {  // Multiple conditions
  process(i);
}

// NOT canonical - loop-variant bound
for (int i = 0; i < compute_bound(); i++) {  // Bound changes
  process(i);
}

// NOT canonical - non-integer
for (double x = 0.0; x < 1.0; x += 0.1) {  // Floating point
  process(x);
}
```

---

### 4.4.2 OpenMP Loop-Iteration Spaces and Vectors

OpenMP defines the **iteration space** of a loop as the set of all iterations that would execute sequentially.

**Iteration Space Concepts:**

1. **Iteration Count**: Number of iterations in the loop
2. **Iteration Vector**: Multi-dimensional index for nested loops
3. **Logical Iteration Number**: Sequential numbering of iterations (0, 1, 2, ...)

**Single Loop Example:**
```fortran
do i = 5, 14, 2
  ! Iteration space: {5, 7, 9, 11, 13}
  ! Iteration count: 5
  ! Logical iteration 0: i=5
  ! Logical iteration 1: i=7
  ! ...
end do
```

**Nested Loop Example:**
```fortran
do i = 1, 3
  do j = 1, 2
    ! Iteration vector: (i,j)
    ! Iteration space: {(1,1), (1,2), (2,1), (2,2), (3,1), (3,2)}
    ! Iteration count: 6
  end do
end do
```

---

### 4.4.3 collapse Clause

**Name**: `collapse`  
**Arguments**: `n` (positive integer)  
**Purpose**: Treat `n` nested loops as a single loop with a larger iteration space

**Syntax:**
```fortran
collapse(n)
```

**Semantics:**
- Forms a single iteration space from `n` nested loops
- All `n` loops must be in canonical form
- No OpenMP directives allowed between collapsed loops
- Increases parallelism by creating more iterations to distribute

**Examples:**

**Without collapse:**
```fortran
!$omp parallel do
do i = 1, 10
  do j = 1, 20
    a(i,j) = compute(i, j)
  end do
end do
```
- Only 10 iterations available for parallelization

**With collapse:**
```fortran
!$omp parallel do collapse(2)
do i = 1, 10
  do j = 1, 20
    a(i,j) = compute(i, j)
  end do
end do
```
- 200 iterations available for parallelization (10 × 20)
- Better load balancing with more threads

**Three-level collapse:**
```c
#pragma omp parallel for collapse(3)
for (int i = 0; i < n1; i++) {
  for (int j = 0; j < n2; j++) {
    for (int k = 0; k < n3; k++) {
      matrix[i][j][k] = compute(i, j, k);
    }
  }
}
// Total iterations: n1 × n2 × n3
```

**Restrictions:**
- All collapsed loops must be perfectly nested (no code between loop headers)
- All collapsed loops must be in canonical form
- Loop iteration variables must not be modified in loop body

---

### 4.4.4 ordered Clause

**Name**: `ordered`  
**Arguments**: `n` (optional positive integer)  
**Purpose**: Specify that iterations must execute in the order of the sequential loop

**Syntax:**
```fortran
ordered
ordered(n)
```

**Use Cases:**

1. **Sequential Dependencies**: When loop iterations must execute in order
2. **Ordered Sections**: Use with `ordered` directive to specify ordered regions

**Example - Ordered Execution:**
```fortran
!$omp parallel do ordered
do i = 1, n
  ! Parallel work
  temp = compute(a(i))
  
  !$omp ordered
    ! This section executes in sequential order
    call write_result(i, temp)
  !$omp end ordered
end do
```

**Example with doacross Dependencies:**
```c
#pragma omp for ordered(1)
for (int i = 1; i < n; i++) {
  #pragma omp ordered depend(source)
  // Current iteration
  
  #pragma omp ordered depend(sink: i-1)
  // Wait for previous iteration
  a[i] = a[i-1] + compute(i);
}
```

---

### 4.4.5 Consistent Loop Schedules

When the same loop is encountered multiple times (e.g., in different parallel regions), OpenMP may require **consistent loop schedules** to ensure correct behavior.

**Consistency Requirements:**

1. **Same iteration count**: Loop must have same number of iterations each time
2. **Same schedule**: If schedule is specified, it must be the same
3. **Same chunk size**: If specified, must be consistent

**Example:**
```fortran
! First encounter
!$omp parallel do schedule(static, 16)
do i = 1, n
  a(i) = b(i) + c(i)
end do

! Second encounter - must use same schedule
!$omp parallel do schedule(static, 16)
do i = 1, n
  d(i) = a(i) * 2
end do
```

**Consequences of Inconsistency:**
- Undefined behavior
- Race conditions
- Incorrect results

---

## Summary: Key Points for Chapter 4

### Base Language Concepts
1. **OpenMP Types**: Predefined types for locks, allocators, events, etc.
2. **Named Constants**: Memory allocators, memory spaces, hints, schedules
3. **Stylized Expressions**: Loop-invariant expressions for loop bounds

### Structured Blocks
- Single entry at top, single exit at bottom
- No branching in/out (with limited exceptions)
- Required for most OpenMP constructs

### Canonical Loop Form
- **Required** for loop parallelization
- Specific requirements for initialization, test, and increment
- Both single and nested loops must be canonical

**C/C++ Canonical:**
```c
for (init; test; incr) { body }
```

**Fortran Canonical:**
```fortran
do var = lb, ub, step
```

### Loop Clauses
- **collapse(n)**: Collapse n nested loops into single iteration space
- **ordered**: Specify sequential execution order requirements

### Important Restrictions
1. Loop bounds must be loop-invariant
2. Loop variables must be integer type (in most cases)
3. No modifications to loop control variables in body
4. Collapsed loops must be perfectly nested
5. No OpenMP directives between collapsed loop headers

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

---

### 5.8 Data-Mapping Control

Data-mapping clauses control how variables are mapped between different device data environments, primarily used with `target` constructs for offloading to accelerators or GPUs.

#### 5.8.1 Implicit Data-Mapping Attribute Rules

When a variable is referenced in a `target` construct without an explicit data-mapping clause, OpenMP applies implicit mapping rules:

**Scalar Variables:**
- Implicitly mapped as `firstprivate` (copied in, not copied out)

**Array Sections:**
- Must be explicitly mapped (no implicit mapping)

**Pointers:**
- The pointer variable is `firstprivate`
- Pointed-to data is NOT automatically mapped

**Aggregate Types (structs/derived types):**
- Members may be implicitly or explicitly mapped
- Pointer members require special handling

**Example:**
```c
int scalar = 10;
int array[100];
int *ptr = malloc(100 * sizeof(int));

#pragma omp target
{
  // scalar: implicitly firstprivate (value copied in)
  scalar = scalar + 1;  // Modifies device copy, not host
  
  // array: ERROR - must be explicitly mapped
  // array[0] = 1;
  
  // ptr: pointer is firstprivate, but data not mapped
  // ptr[0] = 1;  // ERROR - data not accessible
}
```

---

#### 5.8.3 map Clause

**Name**: `map`  
**Arguments**: `map-type-modifier[, map-type-modifier[, ...]]`, `map-type : locator-list`  
**Properties**: Can appear multiple times  
**Allowed on**: `target`, `target data`, `target enter data`, `target exit data`

**Purpose**: Control data movement between host and device data environments

**Syntax:**
```
map([map-type-modifier[,]] map-type : locator-list)
```

#### Map Types

| Map Type | Direction | Description | Use Case |
|----------|-----------|-------------|----------|
| `to` | Host → Device | Allocate on device, copy from host | Input data |
| `from` | Device → Host | Allocate on device, copy to host at exit | Output data |
| `tofrom` | Bidirectional | Copy to device at entry, back to host at exit | Input/Output data |
| `alloc` | None | Allocate on device, no data transfer | Scratch space |
| `release` | None | Deallocate device memory | Cleanup |
| `delete` | None | Deallocate device memory (force) | Forced cleanup |

#### Map Type Modifiers

| Modifier | Meaning |
|----------|---------|
| `always` | Always perform data transfer (even if already present) |
| `close` | Allocate in high-bandwidth memory close to device |
| `present` | Assume data is already present on device (error if not) |
| `mapper(mapper-identifier)` | Use custom mapper for complex types |

**Basic Examples:**

```c
int n = 1000;
double *a = malloc(n * sizeof(double));
double *b = malloc(n * sizeof(double));
double *c = malloc(n * sizeof(double));

// Map arrays with different types
#pragma omp target map(to: a[0:n], b[0:n]) map(from: c[0:n])
{
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];  // a,b input; c output
  }
}
```

**Fortran Examples:**

```fortran
real(8), dimension(1000) :: a, b, c
integer :: n = 1000

! Map arrays with different types
!$omp target map(to: a(1:n), b(1:n)) map(from: c(1:n))
  do i = 1, n
    c(i) = a(i) + b(i)
  end do
!$omp end target
```

#### Common Mapping Patterns

**1. Read-only input data:**
```c
#pragma omp target map(to: input[0:size])
{
  // Use input data
}
```

**2. Write-only output data:**
```c
#pragma omp target map(from: output[0:size])
{
  // Generate output
}
```

**3. Read-write data:**
```c
#pragma omp target map(tofrom: data[0:size])
{
  // Modify data in place
}
```

**4. Temporary device arrays:**
```c
#pragma omp target map(alloc: temp[0:size])
{
  // Use temp as scratch space, no data transfer
}
```

**5. Persistent data across multiple target regions:**
```c
// Start data region
#pragma omp target data map(tofrom: array[0:n])
{
  // First kernel
  #pragma omp target
  {
    process_step1(array, n);
  }
  
  // Second kernel - array stays on device
  #pragma omp target
  {
    process_step2(array, n);
  }
  
} // Data copied back here
```

#### Advanced: Mapping Structures

**C/C++ Structure Mapping:**
```c
typedef struct {
  int n;
  double *data;
} MyStruct;

MyStruct s;
s.n = 1000;
s.data = malloc(s.n * sizeof(double));

// Map both structure and pointed-to data
#pragma omp target map(to: s) map(tofrom: s.data[0:s.n])
{
  for (int i = 0; i < s.n; i++) {
    s.data[i] = s.data[i] * 2.0;
  }
}
```

**Fortran Derived Type Mapping:**
```fortran
type :: MyType
  integer :: n
  real(8), dimension(:), allocatable :: data
end type

type(MyType) :: obj
allocate(obj%data(1000))
obj%n = 1000

!$omp target map(to: obj%n) map(tofrom: obj%data)
  do i = 1, obj%n
    obj%data(i) = obj%data(i) * 2.0
  end do
!$omp end target
```

#### Map Type Modifier Examples

**Using `always` modifier:**
```c
// Always copy data even if already present
#pragma omp target map(always, tofrom: array[0:n])
{
  process(array, n);
}
```

**Using `present` modifier:**
```c
// Assume array is already on device (error if not)
#pragma omp target data map(to: array[0:n])
{
  #pragma omp target map(present: array[0:n])
  {
    process(array, n);
  }
}
```

#### Array Section Mapping

**Contiguous sections:**
```c
// C/C++: [lower_bound : length]
double array[1000];
#pragma omp target map(to: array[10:100])  // Elements 10-109
```

```fortran
! Fortran: (lower_bound : upper_bound)
real(8), dimension(1000) :: array
!$omp target map(to: array(10:109))  ! Elements 10-109
```

**Multi-dimensional arrays:**
```c
double matrix[100][100];
#pragma omp target map(tofrom: matrix[0:100][0:100])
```

```fortran
real(8), dimension(100,100) :: matrix
!$omp target map(tofrom: matrix(1:100, 1:100))
```

---

#### 5.8.7 defaultmap Clause

**Name**: `defaultmap`  
**Arguments**: `implicit-behavior`, `variable-category`  
**Purpose**: Set default data-mapping behavior for variables of a specific category

**Syntax:**
```
defaultmap(implicit-behavior [: variable-category])
```

**Implicit Behaviors:**
- `alloc`: Allocate on device without data transfer
- `to`: Map to device at entry
- `from`: Map from device at exit
- `tofrom`: Map to device at entry and from device at exit
- `firstprivate`: Private copy initialized from host
- `none`: No implicit mapping (must specify explicitly)
- `default`: Use default implicit mapping rules
- `present`: Assume all variables are present

**Variable Categories:**
- `scalar`: Scalar variables
- `aggregate`: Structures and derived types
- `allocatable`: Allocatable arrays (Fortran)
- `pointer`: Pointer variables

**Examples:**

```c
// All scalars default to tofrom
#pragma omp target defaultmap(tofrom:scalar)
{
  // Scalars automatically mapped as tofrom
}
```

```fortran
! All scalars must be explicitly specified
!$omp target defaultmap(none:scalar)
  ! Must map all scalars explicitly
!$omp end target
```

---

### 5.9 Data-Motion Clauses

Data-motion clauses provide finer control over when data is transferred between host and device.

#### 5.9.1 to Clause

**Name**: `to`  
**Arguments**: `locator-list`  
**Allowed on**: `target update`

**Purpose**: Transfer data from host to device without entering a target region

**Syntax:**
```
to(locator-list)
```

**Example:**
```c
double array[1000];

// Initialize on host
for (int i = 0; i < 1000; i++) {
  array[i] = i * 2.0;
}

// Create device data environment
#pragma omp target data map(alloc: array[0:1000])
{
  // Update device copy with host data
  #pragma omp target update to(array[0:1000])
  
  // Use data on device
  #pragma omp target
  {
    process(array, 1000);
  }
}
```

**Fortran Example:**
```fortran
real(8), dimension(1000) :: array

! Initialize on host
do i = 1, 1000
  array(i) = i * 2.0
end do

! Create device data environment
!$omp target data map(alloc: array)
  ! Update device copy
  !$omp target update to(array)
  
  ! Use on device
  !$omp target
    call process(array, 1000)
  !$omp end target
!$omp end target data
```

---

#### 5.9.2 from Clause

**Name**: `from`  
**Arguments**: `locator-list`  
**Allowed on**: `target update`

**Purpose**: Transfer data from device to host without exiting a target region

**Syntax:**
```
from(locator-list)
```

**Example:**
```c
double result[10];

#pragma omp target data map(alloc: result[0:10])
{
  // Compute on device
  #pragma omp target
  {
    for (int i = 0; i < 10; i++) {
      result[i] = compute(i);
    }
  }
  
  // Copy results back to host (without ending data region)
  #pragma omp target update from(result[0:10])
  
  // Use results on host
  check_results(result, 10);
  
  // Continue with more device computation
  #pragma omp target
  {
    process_more(result, 10);
  }
}
```

**Use Case - Incremental Results:**
```c
#pragma omp target data map(alloc: results[0:n])
{
  for (int iter = 0; iter < max_iter; iter++) {
    // Compute on device
    #pragma omp target
    {
      compute_iteration(results, n, iter);
    }
    
    // Check convergence on host
    #pragma omp target update from(results[0:n])
    if (check_convergence(results, n)) {
      break;
    }
  }
}
```

---

### 5.10 uniform Clause

**Name**: `uniform`  
**Arguments**: `argument-list`  
**Allowed on**: `declare simd`

**Purpose**: Indicate that an argument has the same value for all concurrent executions of a SIMD function

**Syntax:**
```
uniform(argument-list)
```

**Semantics:**
- The specified arguments have uniform values across all SIMD lanes
- Enables optimizations by avoiding redundant loads
- Only valid on function/subroutine declarations

**Example:**

**C/C++:**
```c
// Function where 'n' is uniform across all SIMD lanes
#pragma omp declare simd uniform(n)
double compute(double *a, int i, int n) {
  return a[i] * n;  // n is same for all lanes
}

// Usage
#pragma omp simd
for (int i = 0; i < 1000; i++) {
  result[i] = compute(data, i, scale_factor);
  // scale_factor is uniform across iterations
}
```

**Fortran:**
```fortran
!$omp declare simd(compute) uniform(n)
function compute(a, i, n) result(val)
  real(8), dimension(*) :: a
  integer :: i, n
  real(8) :: val
  val = a(i) * n
end function

! Usage
!$omp simd
do i = 1, 1000
  result(i) = compute(data, i, scale_factor)
end do
!$omp end simd
```

---

### 5.11 aligned Clause

**Name**: `aligned`  
**Arguments**: `list`, `alignment` (optional)  
**Allowed on**: `declare simd`, `simd`

**Purpose**: Specify that list items are aligned to specific byte boundaries for SIMD optimization

**Syntax:**
```
aligned(list [: alignment])
```

**Semantics:**
- Indicates that pointers/arrays are aligned to `alignment`-byte boundaries
- If alignment is omitted, implementation-defined alignment is assumed
- Enables more efficient SIMD code generation
- Incorrect alignment specification leads to undefined behavior

**Examples:**

**C/C++:**
```c
// Declare that pointer is 64-byte aligned
#pragma omp declare simd aligned(a:64)
void process(double *a, int n) {
  for (int i = 0; i < n; i++) {
    a[i] = a[i] * 2.0;
  }
}

// Allocate aligned memory
double *data = aligned_alloc(64, n * sizeof(double));

#pragma omp simd aligned(data:64)
for (int i = 0; i < n; i++) {
  data[i] = data[i] + 1.0;
}
```

**Fortran:**
```fortran
!$omp declare simd(process) aligned(a:64)
subroutine process(a, n)
  real(8), dimension(*) :: a
  integer :: n
  integer :: i
  
  do i = 1, n
    a(i) = a(i) * 2.0
  end do
end subroutine

! Usage with aligned array
!$omp simd aligned(data:64)
do i = 1, n
  data(i) = data(i) + 1.0
end do
!$omp end simd
```

**Combined with Linear:**
```c
#pragma omp declare simd aligned(a,b:32) linear(i:1)
double compute(double *a, double *b, int i) {
  return a[i] + b[i];
}

double *a = aligned_alloc(32, 1000 * sizeof(double));
double *b = aligned_alloc(32, 1000 * sizeof(double));

#pragma omp simd
for (int i = 0; i < 1000; i++) {
  result[i] = compute(a, b, i);
}
```

**Common Alignment Values:**
- 16 bytes: SSE vectors
- 32 bytes: AVX vectors
- 64 bytes: AVX-512 vectors, cache line alignment

---

## Complete Data-Mapping Example

Here's a comprehensive example showing various data-mapping techniques:

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
  int n = 10000;
  double *a = malloc(n * sizeof(double));
  double *b = malloc(n * sizeof(double));
  double *c = malloc(n * sizeof(double));
  double sum = 0.0;
  
  // Initialize input data
  for (int i = 0; i < n; i++) {
    a[i] = i * 1.0;
    b[i] = i * 2.0;
  }
  
  // Start persistent data region
  #pragma omp target data map(to: a[0:n], b[0:n]) \
                           map(alloc: c[0:n])
  {
    // Compute c = a + b on device
    #pragma omp target
    {
      #pragma omp parallel for
      for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
      }
    }
    
    // Copy results back for checking
    #pragma omp target update from(c[0:n])
    
    printf("First result: c[0] = %f\n", c[0]);
    
    // Update coefficient on host
    for (int i = 0; i < n; i++) {
      a[i] *= 2.0;
    }
    
    // Update device copy
    #pragma omp target update to(a[0:n])
    
    // Compute again with updated data
    #pragma omp target map(tofrom: sum)
    {
      #pragma omp parallel for reduction(+:sum)
      for (int i = 0; i < n; i++) {
        c[i] = a[i] * b[i];
        sum += c[i];
      }
    }
    
    // Copy final results back
    #pragma omp target update from(c[0:n])
    
  } // End data region - c is deallocated on device
  
  printf("Sum: %f\n", sum);
  printf("Final result: c[0] = %f\n", c[0]);
  
  free(a);
  free(b);
  free(c);
  
  return 0;
}
```

---

## Data-Mapping Quick Reference

### Map Clause Patterns

| Pattern | Syntax | Use Case |
|---------|--------|----------|
| Input only | `map(to: array[0:n])` | Read-only data |
| Output only | `map(from: array[0:n])` | Write-only results |
| Input/Output | `map(tofrom: array[0:n])` | Read and modify data |
| Scratch space | `map(alloc: temp[0:n])` | Temporary device memory |
| Persistent data | `target data map(...)` + multiple `target` | Keep data on device |

### Data Update Patterns

| Operation | Syntax | When to Use |
|-----------|--------|-------------|
| Copy to device | `target update to(var)` | Update device copy mid-computation |
| Copy from device | `target update from(var)` | Check results without ending region |
| Synchronize | `target update to(...) from(...)` | Sync specific variables |

### Best Practices

1. **Minimize data transfers**: Keep data on device as long as possible
2. **Use `target data` regions**: Avoid redundant map/unmap operations
3. **Coalesce transfers**: Map multiple items together rather than separately
4. **Use appropriate map types**: Don't use `tofrom` when `to` or `from` suffices
5. **Align data**: Use aligned memory for better SIMD performance
6. **Profile data movement**: Data transfer is often the bottleneck

---


**Last Updated**: Based on OpenMP 5.2 Specification (November 2021)  
**Usage**: This file is automatically loaded by GitHub Copilot for all files in the workspace (`applyTo: **/*`)
