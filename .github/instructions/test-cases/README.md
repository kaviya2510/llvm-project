# Test Cases for Flang Bug Knowledge Base

## Files

- **openmp-5.2-violations.f90** - Test cases that violate OpenMP 5.2 standards

## Purpose

These test files validate the bug knowledge base by:
1. Testing if documented semantic checks work correctly
2. Demonstrating OpenMP 5.2 standard violations
3. Providing learning examples of incorrect code
4. Enabling regression testing

## How to Use

### Test with Flang

```bash
# Should produce semantic errors
flang-new -fopenmp openmp-5.2-violations.f90

# Expected errors:
# - Linear clause stride must be integer constant
# - COPYPRIVATE and NOWAIT cannot be used together
# - SAFELEN must be positive
# - Non-PURE procedure in workshare
# - Cray pointee in DSA list
# - Type mismatch in atomic
```

### Link to Knowledge Base

Each test relates to documented bugs:
- Linear clause → Bug #111354, #174916
- COPYPRIVATE/NOWAIT → Bug #73486, #171903
- SAFELEN → Bug #109089
- Workshare → Bug #111358
- Cray pointer → Bug #121028, #133232, #111354
- Atomic → Bug #92346, #108516

## Adding New Tests

When adding a bug to the knowledge base:
1. Add corresponding violation test here
2. Document expected error in comments
3. Link back to bug number

## Related Files

- Knowledge Base: ../flang-bug-fixes.instructions.md
- Templates: ../templates/bug-entry-template.md
