# Affine Vectorization Pipeline — Step-by-Step

How Fortran DO loops are lowered through MLIR's affine infrastructure
and vectorized using `affine-super-vectorize`. This document traces every
stage of the pipeline with the **actual IR** produced at each step.

The `SimplifyDoLoop` pass (Fix A) is the key enabler: it removes the
induction-variable spill that Fortran lowering inserts, which otherwise
blocks both affine raising and vectorization.

---

## The Fortran Source

```fortran
program vecadd
  implicit none
  integer, parameter :: N = 1024
  real :: a(N), b(N), c(N)
  integer :: i

  ! Initialise
  do i = 1, N
    a(i) = real(i)
    b(i) = real(2 * i)
  end do

  ! Kernel: simple vector add — should vectorize
  do i = 1, N
    c(i) = a(i) + b(i)
  end do
end program
```

Three arrays of 1024 floats. The **kernel loop** (`c(i) = a(i) + b(i)`)
is the vectorization target. The initialisation loop exists to give the
program meaningful data but is not the focus.

---

## Stage 1: Fortran → FIR

**Command:**

```bash
flang -fc1 -emit-fir vecadd.f90 -o step1_fir.mlir
```

**What it does:** The flang frontend parses Fortran, runs HLFIR lowering
internally, and emits FIR. This is the starting point for the optimisation
pipeline.

**IR dump (kernel loop — `c(i) = a(i) + b(i)`):**

```mlir
    fir.do_loop %arg0 = %c1_i32 to %c1024_i32 step %c1_i32 : i32 {
      fir.store %arg0 to %12 : !fir.ref<i32>               // IV spill: i = arg0
      %42 = fir.load %12 : !fir.ref<i32>                   // reload i
      %43 = fir.convert %42 : (i32) -> i64                  // i → i64
      %44 = fir.array_coor %4(%3) %43                       // &a(i)
          : (!fir.ref<!fir.array<1024xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      %45 = fir.load %44 : !fir.ref<f32>                   // load a(i)
      %46 = fir.load %12 : !fir.ref<i32>                   // reload i AGAIN
      %47 = fir.convert %46 : (i32) -> i64
      %48 = fir.array_coor %7(%6) %47                       // &b(i)
          : (!fir.ref<!fir.array<1024xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      %49 = fir.load %48 : !fir.ref<f32>                   // load b(i)
      %50 = arith.addf %45, %49 fastmath<contract> : f32   // a(i) + b(i)
      %51 = fir.load %12 : !fir.ref<i32>                   // reload i AGAIN
      %52 = fir.convert %51 : (i32) -> i64
      %53 = fir.array_coor %10(%9) %52                      // &c(i)
          : (!fir.ref<!fir.array<1024xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      fir.store %50 to %53 : !fir.ref<f32>                 // c(i) = result
    }
```

**Key observations:**

| Feature | How it appears |
|---|---|
| Loop | `fir.do_loop` with `i32` bounds (1 to 1024, step 1) |
| IV spill | `fir.store %arg0 to %12` — every iteration writes `i` to its memory slot |
| IV reload | `fir.load %12` — the body reloads `i` **three times** (once per array access) |
| Array access | `fir.array_coor %base(%shape) %subscript` — high-level, 1-based |
| The problem | The store-load chain through the memory slot is **opaque** to optimisers |

---

## Stage 2: SimplifyDoLoop (`--simplify-fir-loop`)

**Command:**

```bash
fir-opt --simplify-fir-loop step1_fir.mlir -o step2_simplify.mlir
```

**Source file:** `flang/lib/Optimizer/Transforms/SimplifyDoLoop.cpp`
(new in Fix A / `affine-fix-a-wip-backup`)

**What it does:** Forwards the DO loop's induction variable past its
memory slot. The `fir.store %iv to slot` + `fir.load slot` chain is
replaced by direct use of `%iv`. If the slot has no remaining readers,
the post-loop "final value" store is removed too.

**Why this matters:**

1. The IV spill is a **write to a scalar** inside the loop body, which
   blocks `affine-super-vectorize` from vectorizing.
2. The reload from the slot means array indices depend on a **memory load**,
   not the loop IV directly — so `raise-scf-to-affine` and
   `affine-raise-from-memref` cannot recognise them as affine expressions.

**IR dump (kernel loop after SimplifyDoLoop):**

```mlir
    fir.do_loop %arg0 = %c1_i32 to %c1024_i32 step %c1_i32 : i32 {
      %42 = fir.convert %arg0 : (i32) -> i64               // IV used DIRECTLY (no reload!)
      %43 = fir.array_coor %4(%3) %42                       // &a(i)
          : (!fir.ref<!fir.array<1024xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      %44 = fir.load %43 : !fir.ref<f32>                   // load a(i)
      %45 = fir.convert %arg0 : (i32) -> i64               // IV used DIRECTLY
      %46 = fir.array_coor %7(%6) %45                       // &b(i)
          : (!fir.ref<!fir.array<1024xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      %47 = fir.load %46 : !fir.ref<f32>                   // load b(i)
      %48 = arith.addf %44, %47 fastmath<contract> : f32   // a(i) + b(i)
      %49 = fir.convert %arg0 : (i32) -> i64               // IV used DIRECTLY
      %50 = fir.array_coor %10(%9) %49                      // &c(i)
          : (!fir.ref<!fir.array<1024xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      fir.store %48 to %50 : !fir.ref<f32>                 // c(i) = result
    }
```

**What changed (Stage 1 → Stage 2):**

| Before (Stage 1) | After (Stage 2) | Why |
|---|---|---|
| `fir.store %arg0 to %12` (IV spill) | **Gone** | Pass forwards IV, removes spill |
| `fir.load %12` (3× per iteration) | **Gone** — uses `%arg0` directly | Reloads replaced by IV |
| `fir.store %27 to %12` (post-loop final value) | **Gone** | No readers left for the slot |

**The loop body went from 13 ops to 9 ops.** No memory traffic for the
loop variable at all.

---

## Stage 3: FIR memory ops → memref ops (`--fir-to-memref`)

**Command:**

```bash
fir-opt --simplify-fir-loop --fir-to-memref step1_fir.mlir -o step3_memref.mlir
```

**Source file:** `flang/lib/Optimizer/Transforms/FIRToMemRef.cpp`

**What it does:** Converts FIR's memory operations (`fir.alloca`,
`fir.load`, `fir.store`, `fir.array_coor`) to standard MLIR `memref`
ops. This is the bridge from FIR's type system to MLIR's.

**Key transformation for array access — `fir.array_coor` explodes:**

```mlir
    // Before (1 op):
    fir.array_coor %4(%3) %42 : (!fir.ref<!fir.array<1024xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>

    // After (converted to memref + 0-based indexing):
    %25 = fir.convert %3 : (!fir.ref<!fir.array<1024xf32>>) -> memref<1024xf32>
    %26 = arith.index_cast %22 : i64 to index               // subscript → index
    %27 = arith.subi %26, %c1 : index                       // 1-based → 0-based
    memref.store %24, %25[%27] : memref<1024xf32>
```

Since the arrays are statically sized (`1024xf32`), the memref type is
`memref<1024xf32>` — no dynamic shapes or strides. This is much simpler
than the SAXPY case (which had runtime-sized arrays with `memref<?xf32>`
and `reinterpret_cast`).

---

## Stage 4: FIR loop → SCF loop (`--fir-to-scf`)

**Command:**

```bash
fir-opt --simplify-fir-loop --fir-to-memref --fir-to-scf step1_fir.mlir -o step4_scf.mlir
```

**Source file:** `flang/lib/Optimizer/Transforms/FIRToSCF.cpp`

**What it does:** Converts `fir.do_loop` (i32 bounds) to `scf.for`
(index bounds, 0-based). The Fortran IV is reconstructed from the
0-based index.

**IR dump (kernel loop):**

```mlir
    scf.for %arg0 = %c0_3 to %40 step %c1_4 {

      // ── IV reconstruction (0-based → Fortran 1-based) ──
      %60 = fir.convert %arg0 : (index) -> i64              // scf IV → i64
      %61 = fir.convert %c1_i32 : (i32) -> i64              // lower_bound → i64
      %62 = fir.convert %c1_i32 : (i32) -> i64              // step → i64
      %63 = arith.muli %60, %62 overflow<nsw> : i64         // IV * step
      %64 = arith.addi %61, %63 overflow<nsw> : i64         // lower + IV*step
      %65 = arith.trunci %64 overflow<nsw> : i64 to i32     // → i32

      // ── Array index computation (verbose) ──
      %66 = fir.convert %65 : (i32) -> i64
      %67 = fir.convert %4 : (!fir.ref<!fir.array<1024xf32>>) -> memref<1024xf32>
      %68 = arith.index_cast %66 : i64 to index
      %69 = arith.subi %68, %c1_5 : index                   // 1-based → 0-based
      %70 = arith.muli %69, %c1_5 : index                   // × stride (1)
      %71 = arith.subi %c1_5, %c1_5 : index                 // lower_bound - 1 = 0
      %72 = arith.addi %70, %71 : index                     // final offset
      %73 = memref.load %67[%72] : memref<1024xf32>         // a(i)

      // ... same verbose pattern for b(i) and c(i) ...
    }
```

**What changed (Stage 3 → Stage 4):**

| Before (Stage 3) | After (Stage 4) | Why |
|---|---|---|
| `fir.do_loop %arg0 = 1_i32 to 1024_i32 step 1_i32 : i32` | `scf.for %arg0 = 0 to trip_count step 1` | `scf.for` needs `index` bounds |
| IV was directly the Fortran `i` (i32) | IV is 0-based `index`; Fortran `i` reconstructed via `nsw` arithmetic | Type mismatch forces reconstruction |
| Array index: just `arith.subi %idx, 1` | 4+ redundant arith ops (`muli ×1`, `subi 1-1`, `addi +0`) | `fir.array_coor` expansion is verbose |

**The `nsw` flags** on the reconstruction arithmetic (`overflow<nsw>`) are
critical — they allow the canonicaliser to simplify the round-trip in Stage 6.

---

## Stage 5: FIR casts → arith casts (`--fir-convert-to-arith`)

**Command:**

```bash
fir-opt --simplify-fir-loop --fir-to-memref --fir-to-scf \
        --fir-convert-to-arith step1_fir.mlir -o step5_arith.mlir
```

**Source file:** `flang/lib/Optimizer/Transforms/FIRConvertToArith.cpp`

**What it does:** Replaces `fir.convert` (FIR's generic cast) with
specific `arith` ops (`arith.index_cast`, `arith.extsi`, `arith.trunci`)
when the conversion is between integers/indices. Pointer casts like
`fir.convert %ref : (!fir.ref<...>) -> memref<...>` are left for codegen.

**Why this matters:** `fir.convert` is **opaque** to the arith
canonicaliser. It can't pattern-match through it. Once everything is
`arith.index_cast` and `arith.extsi`, the canonicaliser can analyse
and simplify the cast chains.

---

## Stage 6: Simplify arithmetic (`--canonicalize --cse`)

**Command:**

```bash
fir-opt --simplify-fir-loop --fir-to-memref --fir-to-scf \
        --fir-convert-to-arith --canonicalize --cse \
        step1_fir.mlir -o step6_canon.mlir
```

**What it does:** Fires the `arith` canonicalization patterns from
commit 1, plus standard MLIR canonicalization and common subexpression
elimination.

**IR dump (kernel loop — dramatically simplified):**

```mlir
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      %21 = fir.convert %3 : (!fir.ref<!fir.array<1024xf32>>) -> memref<1024xf32>
      %22 = memref.load %21[%arg0] : memref<1024xf32>       // a(i) — uses %arg0 DIRECTLY!
      %23 = fir.convert %5 : (!fir.ref<!fir.array<1024xf32>>) -> memref<1024xf32>
      %24 = memref.load %23[%arg0] : memref<1024xf32>       // b(i) — uses %arg0 DIRECTLY!
      %25 = arith.addf %22, %24 fastmath<contract> : f32    // a(i) + b(i)
      %26 = fir.convert %7 : (!fir.ref<!fir.array<1024xf32>>) -> memref<1024xf32>
      memref.store %25, %26[%arg0] : memref<1024xf32>       // c(i) = result — %arg0 DIRECTLY!
    }
```

**What changed (Stage 5 → Stage 6) — the big simplification:**

| Before (Stage 5) | After (Stage 6) | What did it |
|---|---|---|
| IV reconstruction: `index→i64→muli×1→addi+1→trunci→i32` | **Gone** | `nsw` patterns: `muli ×1` folds, `addi 1` + `subi 1` cancel |
| Array index: `extsi→index_cast→subi→muli×1→subi(1-1)→addi` | Just `%arg0` | Cast chains collapse; redundant ops fold |
| Trip count: `(1024-1+1)/1` (5 ops) | `%c1024` (constant) | Arithmetic folded at compile time |
| 3 separate index computations | 1 shared `%arg0` | **CSE** eliminated duplicates |
| `fir.convert` to memref (3×) | Still 3× | Pointer casts stay (handled by codegen) |

**Result:** The loop body is now just `memref.load`, `memref.load`,
`arith.addf`, `memref.store` — plus 3 `fir.convert` pointer casts.

---

## Stage 7: SCF → affine loop (`--raise-scf-to-affine`)

**Command:**

```bash
fir-opt --simplify-fir-loop --fir-to-memref --fir-to-scf \
        --fir-convert-to-arith --canonicalize --cse \
        --raise-scf-to-affine step1_fir.mlir -o step7_affine.mlir
```

**Source:** upstream MLIR pass `mlir/lib/Dialect/SCF/Transforms/RaiseToAffine.cpp`

**IR dump (kernel loop — one-line change):**

```mlir
    affine.for %arg0 = 0 to 1024 {                          // CHANGED: was scf.for
      %21 = fir.convert %3 : (!fir.ref<!fir.array<1024xf32>>) -> memref<1024xf32>
      %22 = memref.load %21[%arg0] : memref<1024xf32>       // a(i)
      %23 = fir.convert %5 : (!fir.ref<!fir.array<1024xf32>>) -> memref<1024xf32>
      %24 = memref.load %23[%arg0] : memref<1024xf32>       // b(i)
      %25 = arith.addf %22, %24 fastmath<contract> : f32    // a(i) + b(i)
      %26 = fir.convert %7 : (!fir.ref<!fir.array<1024xf32>>) -> memref<1024xf32>
      memref.store %25, %26[%arg0] : memref<1024xf32>       // c(i) = result
    }
```

**What changed (Stage 6 → Stage 7):**

| Before (Stage 6) | After (Stage 7) | Why |
|---|---|---|
| `scf.for %arg0 = %c0 to %c1024 step %c1` | `affine.for %arg0 = 0 to 1024` | Bounds are constants → valid affine |

**One line changed.** The body is identical. But the loop is now an
`affine.for` — which means affine optimisation passes can see it.

**However:** The body still uses `memref.load`/`memref.store`, not
`affine.load`/`affine.store`. The `fir.convert` pointer casts in the
body are opaque to the affine raiser. That's the next stage.

---

## Stage 8: Raise to fully affine IR

**Command:**

```bash
fir-opt --simplify-fir-loop --fir-to-memref --fir-to-scf \
        --fir-convert-to-arith --canonicalize --cse \
        --raise-scf-to-affine \
        --affine-raise-from-memref \
        --canonicalize \
        --affine-loop-invariant-code-motion \
        step1_fir.mlir -o step8_fullaffine.mlir
```

This is a sequence of 3 sub-passes after the affine raise:

| Sub-pass | What it does |
|---|---|
| `--affine-raise-from-memref` | Converts `memref.load/store` with affine indices → `affine.load/store` |
| `--canonicalize` | Cleans up remaining redundancies |
| `--affine-loop-invariant-code-motion` | Hoists loop-invariant `fir.convert` pointer casts out of the loop |

**IR dump (kernel loop — fully affine):**

```mlir
    // fir.convert hoisted OUTSIDE the loop by LICM:
    %17 = fir.convert %3 : (!fir.ref<!fir.array<1024xf32>>) -> memref<1024xf32>   // a
    %18 = fir.convert %5 : (!fir.ref<!fir.array<1024xf32>>) -> memref<1024xf32>   // b
    %19 = fir.convert %7 : (!fir.ref<!fir.array<1024xf32>>) -> memref<1024xf32>   // c

    affine.for %arg0 = 0 to 1024 {
      %26 = affine.load %17[%arg0] : memref<1024xf32>       // a(i) — AFFINE!
      %27 = affine.load %18[%arg0] : memref<1024xf32>       // b(i) — AFFINE!
      %28 = arith.addf %26, %27 fastmath<contract> : f32    // a(i) + b(i)
      affine.store %28, %19[%arg0] : memref<1024xf32>       // c(i) = result — AFFINE!
    }
```

**What changed (Stage 7 → Stage 8):**

| Before (Stage 7) | After (Stage 8) | Why |
|---|---|---|
| `memref.load %21[%arg0]` | `affine.load %17[%arg0]` | Index `%arg0` is the loop IV → valid affine |
| `memref.store %25, %26[%arg0]` | `affine.store %28, %19[%arg0]` | Same reason |
| `fir.convert` inside loop (3×) | `fir.convert` hoisted outside loop | LICM: loop-invariant |

**Result:** The kernel loop is **perfectly affine** — `affine.for` with
`affine.load`/`affine.store` using the loop IV directly, and nothing
else in the body except the `arith.addf`. This is exactly what
`affine-super-vectorize` needs.

---

## Stage 9: Affine super-vectorize (`--affine-super-vectorize`)

**Command:**

```bash
fir-opt --simplify-fir-loop --fir-to-memref --fir-to-scf \
        --fir-convert-to-arith --canonicalize --cse \
        --raise-scf-to-affine \
        --affine-raise-from-memref \
        --canonicalize \
        --affine-loop-invariant-code-motion \
        --affine-super-vectorize="virtual-vector-size=4" \
        step1_fir.mlir -o step9_vectorized.mlir
```

**Source:** upstream MLIR pass `mlir/lib/Dialect/Affine/Transforms/SuperVectorize.cpp`

**What it does:** Vectorizes `affine.for` loops by replacing scalar
`affine.load`/`affine.store` with `vector.transfer_read`/`vector.transfer_write`
and scalar arithmetic with vector arithmetic.

**IR dump (kernel loop — vectorized!):**

```mlir
    affine.for %arg0 = 0 to 1024 step 4 {                                      // step = 4!
      %26 = ub.poison : f32                                                     // padding value
      %27 = vector.transfer_read %17[%arg0], %26 {in_bounds = [true]}          // READ 4 × a(i)
          : memref<1024xf32>, vector<4xf32>
      %28 = ub.poison : f32
      %29 = vector.transfer_read %18[%arg0], %28 {in_bounds = [true]}          // READ 4 × b(i)
          : memref<1024xf32>, vector<4xf32>
      %30 = arith.addf %27, %29 fastmath<contract> : vector<4xf32>             // ADD 4 at once!
      vector.transfer_write %30, %19[%arg0] {in_bounds = [true]}               // WRITE 4 × c(i)
          : vector<4xf32>, memref<1024xf32>
    }
```

**What changed (Stage 8 → Stage 9):**

| Before (Stage 8) | After (Stage 9) | Why |
|---|---|---|
| `affine.for %arg0 = 0 to 1024` (step 1) | `affine.for %arg0 = 0 to 1024 step 4` | Loop steps by vector width |
| `affine.load %17[%arg0] : memref<1024xf32>` → `f32` | `vector.transfer_read %17[%arg0] ... : vector<4xf32>` | Scalar load → vector load (4 elements) |
| `arith.addf ... : f32` | `arith.addf ... : vector<4xf32>` | Scalar add → vector add (4 lanes) |
| `affine.store %28, %19[%arg0]` → `f32` | `vector.transfer_write %30, %19[%arg0] ... : vector<4xf32>` | Scalar store → vector store (4 elements) |

**🎉 Vectorization succeeded!** Each iteration now processes **4 elements**
instead of 1. The `in_bounds = [true]` attribute tells later passes that
no out-of-bounds masking is needed (1024 is divisible by 4).

---

## Why SimplifyDoLoop (Fix A) is essential

Without `--simplify-fir-loop`, the pipeline fails to vectorize:

```
Stage 1 (FIR):
  fir.do_loop {
    fir.store %iv to %slot          ← IV spill
    %idx = fir.load %slot           ← reload through memory
    ... use %idx for array access ...
  }
```

**Problem 1 — blocks affine raising:** The array index depends on a
`memref.load` from the IV slot, not on the loop IV. Since a memory load
is not an affine expression of the loop IV, `affine-raise-from-memref`
refuses to convert the `memref.load/store` to `affine.load/store`.

**Problem 2 — blocks vectorization:** Even if the loop is `affine.for`,
the `affine.store` to the IV slot in every iteration is a write to a
scalar memory location inside the loop body. `affine-super-vectorize`
requires the loop body to contain only vectorizable operations (loads,
stores, and arithmetic on arrays), so it skips the loop.

**SimplifyDoLoop removes both blockers** by forwarding the IV directly
and eliminating the spill.

---

## The full pipeline (one-liner)

```bash
BIN=/path/to/llvm-project/build/bin

RAISE="--simplify-fir-loop
       --fir-to-memref --fir-to-scf --fir-convert-to-arith
       --canonicalize --cse
       --raise-scf-to-affine
       --affine-raise-from-memref
       --canonicalize
       --affine-loop-invariant-code-motion"

VECTORIZE='--affine-super-vectorize="virtual-vector-size=4"'

# Emit FIR
$BIN/flang -fc1 -emit-fir vecadd.f90 -o vecadd.fir

# Raise to affine + vectorize
$BIN/fir-opt $RAISE $VECTORIZE vecadd.fir -o vecadd_vec.mlir

# Verify vector ops appear
grep 'vector.transfer' vecadd_vec.mlir
```

---

## Summary of changes at each stage

| Stage | Pass | Key IR change |
|---|---|---|
| **1** | `flang -fc1 -emit-fir` | Fortran → `fir.do_loop`, `fir.store`/`fir.load` (IV spill), `fir.array_coor` |
| **2** | `--simplify-fir-loop` | **IV spill removed** — `fir.store`/`fir.load` of IV gone, body uses `%arg0` directly |
| **3** | `--fir-to-memref` | `fir.array_coor` → `memref.load/store` with 0-based index arithmetic |
| **4** | `--fir-to-scf` | `fir.do_loop` (i32) → `scf.for` (0-based index) + IV reconstruction with `nsw` |
| **5** | `--fir-convert-to-arith` | `fir.convert` → `arith.index_cast`/`arith.extsi` (makes casts visible to canonicaliser) |
| **6** | `--canonicalize --cse` | Cast chains collapse, redundant arithmetic folds, index becomes just `%arg0` |
| **7** | `--raise-scf-to-affine` | `scf.for` → `affine.for` (bounds are affine) |
| **8** | `affine-raise + LICM` | `memref.load/store` → `affine.load/store`; `fir.convert` hoisted out of loop |
| **9** | `--affine-super-vectorize` | **Scalar → vector<4xf32>**: `affine.load` → `vector.transfer_read`, step 1 → step 4 |

---

## Pipeline Diagram

```
Fortran source (vecadd.f90)
    │
    ▼  flang -fc1 -emit-fir
┌─────────────────────────────────────────────────────────┐
│ FIR                                                     │
│  fir.do_loop { store IV → load IV → fir.array_coor }   │
└─────────────────────────────────────────────────────────┘
    │
    ▼  fir-opt --simplify-fir-loop           ★ FIX A
┌─────────────────────────────────────────────────────────┐
│ FIR (IV forwarded)                                      │
│  fir.do_loop { use %iv directly → fir.array_coor }     │
└─────────────────────────────────────────────────────────┘
    │
    ▼  --fir-to-memref --fir-to-scf --fir-convert-to-arith
┌─────────────────────────────────────────────────────────┐
│ SCF + memref + arith                                    │
│  scf.for { verbose index arithmetic + memref.load }     │
└─────────────────────────────────────────────────────────┘
    │
    ▼  --canonicalize --cse
┌─────────────────────────────────────────────────────────┐
│ SCF + memref (simplified)                               │
│  scf.for { memref.load array[%arg0] }                   │
└─────────────────────────────────────────────────────────┘
    │
    ▼  --raise-scf-to-affine
┌─────────────────────────────────────────────────────────┐
│ affine.for + memref                                     │
│  affine.for { memref.load array[%arg0] }                │
└─────────────────────────────────────────────────────────┘
    │
    ▼  --affine-raise-from-memref --canonicalize --affine-loop-invariant-code-motion
┌─────────────────────────────────────────────────────────┐
│ ★ Fully Affine IR                                       │
│  affine.for { affine.load array[%arg0] }                │
└─────────────────────────────────────────────────────────┘
    │
    ▼  --affine-super-vectorize="virtual-vector-size=4"
┌─────────────────────────────────────────────────────────┐
│ ★★ Vectorized!                                          │
│  affine.for step 4 {                                    │
│    vector.transfer_read  : vector<4xf32>                │
│    arith.addf            : vector<4xf32>                │
│    vector.transfer_write : vector<4xf32>                │
│  }                                                      │
└─────────────────────────────────────────────────────────┘
```

---

## Commits

| Commit | What it adds |
|---|---|
| `166856b6971e` — *[flang][mlir] Raise FIR loops to affine via memref/scf* | `--fir-to-memref`, `--fir-to-scf`, `--fir-convert-to-arith`, `--raise-scf-to-affine`, `arith.index_cast` canonicalization patterns, memref patterns in `--fir-to-llvm-ir` |
| `984766605b65` — *SimplifyDoLoop (Fix A)* | `--simplify-fir-loop` pass, vector transfer lowering in CodeGen, `ConvertVectorToSCF`/`ConvertVectorToLLVM` registration |
