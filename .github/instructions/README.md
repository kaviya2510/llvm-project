# Flang Bug-Fix Knowledge Base

Comprehensive knowledge base for training AI agents on Flang bug patterns.

## Directory Structure

```
.github/instructions/
├── README.md                          (this file)
├── flang-bug-fixes.instructions.md    (main knowledge base - 48 bugs)
├── pr-review.instructions.md          (PR review guidelines)
├── llvm.instructions.md               (LLVM instructions)
├── lldb.instructions.md               (LLDB instructions)
└── test-cases/
    ├── README.md
    └── openmp-5.2-violations.f90      (OpenMP standard violations)
```

## ⚡ Super Quick Start

### Add a New Bug (Just PR Number!)

```
@workspace Add PR #180000 to the Flang bug knowledge base
```

That's it! Copilot will:
1. Fetch PR details
2. Generate bug entry
3. Add to knowledge base
4. Update indexes

**You just:** Review and commit!

---

## Search for Bugs

```bash
# Search by keyword
grep -i "atomic" flang-bug-fixes.instructions.md

# Search by file
grep "check-omp-structure.cpp" flang-bug-fixes.instructions.md

# Use with Copilot
@workspace What Cray pointer bugs have been fixed?
@workspace Show me atomic bug patterns
```

---

## Test Standards

```bash
cd test-cases
flang-new -fopenmp openmp-5.2-violations.f90  # Should produce errors
```

---

## Main File: flang-bug-fixes.instructions.md

**1,379 lines, 48 documented bugs**

### Structure

- **Part 1**: Bug Categories (Semantic/Compiler/Runtime)
- **Part 2**: Standards & Best Practices
- **Part 3**: Bug Database (48 bugs with full details)
- **Part 4**: Search Indexes (by category, feature, file)
- **Part 5**: Workflow Guide (how to fix bugs using knowledge base)
- **Part 6**: Maintenance Guide (how to add bugs - super simple!)

### Statistics

- **Total**: 48 bugs
- **Runtime**: 2 (4%)
- **Semantic**: 10 (21%)
- **Compiler**: 36 (75%)
- **Last Updated**: January 25, 2026

---

## Usage Examples

### For Fixing Bugs

```
@workspace I have a Cray pointer segfault. Show me similar bugs.
@workspace What's the pattern for atomic type conversions?
@workspace Show me how to fix linear clause bugs
```

### For Adding Bugs

```
@workspace Add PR #133232 to knowledge base
```

### For Learning

```
@workspace What are common OpenMP lowering bugs?
@workspace Show me semantic check patterns
```

---

## Quick Reference

| Need | Command |
|------|---------|
| Add bug | `@workspace Add PR #XXXXX to knowledge base` |
| Search bugs | `grep -i "keyword" flang-bug-fixes.instructions.md` |
| Test violations | `cd test-cases && flang-new -fopenmp openmp-5.2-violations.f90` |
| Learn patterns | `@workspace Show me [feature] bugs` |

---

## Goals

1. **Train AI agents** on Flang bug patterns
2. **Speed up development** - learn from past fixes
3. **Consistency** - reuse proven solutions
4. **Onboarding** - new contributors learn faster
5. **No regressions** - documented patterns prevent repeat bugs

---

## Related Documentation

- [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html)
- [Flang Documentation](https://flang.llvm.org/)
- [OpenMP 5.2 Specification](https://www.openmp.org/specifications/)
- [Fortran 2018 Standard](https://j3-fortran.org/doc/year/18/18-007r1.pdf)

---

**The simpler, the better!** Just provide PR number → Copilot does the rest! 🚀
