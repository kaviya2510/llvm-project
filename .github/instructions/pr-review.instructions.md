---
applyTo: **/*
---

# LLVM Pull Request Review & Summary Guidelines

When reviewing or summarizing a Pull Request in the LLVM project, follow these comprehensive guidelines:

## 1. PR Summary Structure

Provide a clear, structured summary including:

### Overview
- **What**: Brief description of what the PR changes
- **Why**: Motivation and problem being solved
- **Where**: Which subprojects/components are affected (llvm, clang, mlir, flang, lldb, etc.)
- **Impact**: User-facing changes, API changes, or internal refactoring

### Key Changes
- List major modifications by file/component
- Highlight new features, bug fixes, or optimizations
- Note any breaking changes or deprecations

### Testing
- What tests were added or modified
- Test coverage assessment
- Whether existing tests pass

### Documentation
- Documentation updates (if any)
- API documentation changes
- Release notes impact

## 1.1. Finding Related Context for PRs

Given a PR link (e.g., `https://github.com/llvm/llvm-project/pull/12345`), use these techniques to automatically discover related information:

### A. Find Related PRs (Open and Merged)

```bash
# Search for PRs with similar keywords in title
gh search prs --repo llvm/llvm-project "keyword1 keyword2" --state open --limit 5
gh search prs --repo llvm/llvm-project "keyword1 keyword2" --state closed --limit 5

# Find PRs that modified the same files
gh pr view 12345 --json files --jq '.files[].path' | head -5 | while read file; do
  echo "PRs touching $file:"
  gh search prs --repo llvm/llvm-project "path:$file" --state closed --limit 3 --json number,title,state
done

# Find PRs by the same author on similar topics
AUTHOR=$(gh pr view 12345 --json author --jq '.author.login')
gh search prs --repo llvm/llvm-project "author:$AUTHOR keywords" --state closed --limit 5

# Find linked issues and PRs
gh pr view 12345 --json body --jq '.body' | grep -oP '#\d+' | sort -u
gh pr view 12345 --json closingIssuesReferences --jq '.closingIssuesReferences[].number'
```

### B. Search for RFC Discussions

```bash
# Search GitHub Discussions for RFC
gh search issues --repo llvm/llvm-project "[RFC]" "keyword" --label RFC --limit 5

# Search LLVM Discourse (modern mailing list)
# Format: https://discourse.llvm.org/search?q=keyword%20category%3ARFC
curl -s "https://discourse.llvm.org/search.json?q=keyword%20category%3ARFC" | \
  jq '.topics[0:5] | .[] | {id, title, slug, url: ("https://discourse.llvm.org/t/\(.slug)/\(.id)")}'

# Check PR description for RFC/Discourse links
gh pr view 12345 --json body --jq '.body' | grep -iE "RFC|discourse|discuss|mailing"
gh pr view 12345 --json body --jq '.body' | grep -oP 'https://discourse\.llvm\.org[^\s)]*'
```

### C. Find Mailing List Discussions

```bash
# Search LLVM Discourse (replaced mailing lists)
curl -s "https://discourse.llvm.org/search.json?q=keyword" | \
  jq '.posts[0:10] | .[] | {topic_id, username, created_at, excerpt: .blurb}'

# Search archived mailing lists (pre-2021)
# Use Google with site-specific search:
# site:lists.llvm.org "exact phrase" keyword

# Extract any mailing list links from PR
gh pr view 12345 --json body --jq '.body' | \
  grep -oP 'https?://[^\s)]+lists\.llvm\.org[^\s)]*'
```

### D. Discover Historical Context

```bash
# Get commit history for changed files
gh pr view 12345 --json files --jq '.files[0:5].path' | while read file; do
  echo "=== Recent commits to $file ==="
  git log --oneline --no-merges -n 10 -- "$file"
  echo ""
done

# Search commit messages for related work
KEYWORDS=$(gh pr view 12345 --json title --jq '.title' | grep -oE '\w{4,}' | head -3 | tr '\n' ' ')
git log --all --grep="$KEYWORDS" --oneline -n 15

# Find closed/merged PRs on same topic
gh search prs --repo llvm/llvm-project "$KEYWORDS in:title" --state closed --limit 10

# Check for related issues
gh search issues --repo llvm/llvm-project "$KEYWORDS" --state all --limit 10 --json number,title,state
```

### E. Automated Context Discovery Script

Save as `pr-context-finder.sh`:

```bash
#!/bin/bash
# pr-context-finder.sh - Discover related context for a PR
# Usage: ./pr-context-finder.sh 12345

PR_NUMBER=$1
REPO="llvm/llvm-project"

if [ -z "$PR_NUMBER" ]; then
  echo "Usage: $0 <PR_NUMBER>"
  exit 1
fi

echo "🔍 Finding context for PR #$PR_NUMBER in $REPO"
echo "================================================"
echo ""

# Get PR details
echo "📋 PR Information:"
gh pr view $PR_NUMBER --repo $REPO --json title,author,body,state,createdAt | \
  jq -r '"Title: \(.title)\nAuthor: \(.author.login)\nState: \(.state)\nCreated: \(.createdAt | split("T")[0])"'
echo ""

# Extract keywords from title (filter common words)
TITLE=$(gh pr view $PR_NUMBER --repo $REPO --json title --jq '.title')
KEYWORDS=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | \
  grep -oE '\w{4,}' | grep -vE '^(this|that|with|from|for|and|the)$' | head -5 | tr '\n' '|' | sed 's/|$//')
echo "🔑 Search Keywords: $(echo $KEYWORDS | tr '|' ' ')"
echo ""

# Find related PRs (open and closed separately, gh search doesn't support --state all)
echo "🔗 Related OPEN PRs:"
gh search prs --repo $REPO "$KEYWORDS" --state open --limit 3 --json number,title,state,createdAt | \
  jq -r '.[] | "#\(.number) [\(.state)] \(.title) (\(.createdAt | split("T")[0]))"'
echo ""
echo "🔗 Related CLOSED PRs:"
gh search prs --repo $REPO "$KEYWORDS" --state closed --limit 5 --json number,title,state,createdAt | \
  jq -r '.[] | "#\(.number) [\(.state)] \(.title) (\(.createdAt | split("T")[0]))"'
echo ""

# Find RFCs
echo "💬 Related RFC Discussions:"
gh search issues --repo $REPO "[RFC] $KEYWORDS" --label RFC --limit 3 --json number,title,url | \
  jq -r '.[] | "#\(.number) \(.title)\n  → \(.url)"'
if [ $? -ne 0 ]; then echo "  (No RFCs found)"; fi
echo ""

# Check PR body for references
echo "🔗 References in PR Description:"
BODY=$(gh pr view $PR_NUMBER --repo $REPO --json body --jq '.body')
echo "$BODY" | grep -oP '#\d+' | sort -u | head -5 | xargs -I {} echo "  - Issue/PR: #{}"
echo "$BODY" | grep -oP 'https://discourse\.llvm\.org[^\s)]*' | head -3 | xargs -I {} echo "  - Discourse: {}"
echo "$BODY" | grep -oP 'https://github\.com/llvm/llvm-project/(issues|pull)/\d+' | head -3 | xargs -I {} echo "  - GitHub: {}"
echo ""

# Get changed files
echo "📁 Changed Files (top 10):"
gh pr view $PR_NUMBER --repo $REPO --json files --jq '.files[0:10].path' | sed 's/^/  - /'
echo ""

# File history
echo "📝 Recent Commits to Changed Files:"
gh pr view $PR_NUMBER --repo $REPO --json files --jq '.files[0:3].path' | while read file; do
  if [ -f "$file" ]; then
    echo "  📄 $file"
    git log --oneline --no-merges -n 5 -- "$file" 2>/dev/null | sed 's/^/    /'
    echo ""
  fi
done

echo "✅ Context discovery complete!"
```

### F. Quick Reference Table

| Goal | Command Example |
|------|-----------------|
| View PR | `gh pr view 12345` |
| PR files | `gh pr view 12345 --json files --jq '.files[].path'` |
| Related PRs (open) | `gh search prs --repo llvm/llvm-project "keywords" --state open` |
| Related PRs (closed) | `gh search prs --repo llvm/llvm-project "keywords" --state closed` |
| Find RFCs | `gh search issues --repo llvm/llvm-project "[RFC] keywords" --label RFC` |
| File history | `git log --oneline -n 10 -- path/to/file.cpp` |
| Commit search | `git log --all --grep="keyword" --oneline -n 20` |
| Linked issues | `gh pr view 12345 --json body --jq '.body' \| grep -oP '#\d+'` |
| Discourse search | `curl "https://discourse.llvm.org/search.json?q=keyword"` |

### G. Discovery Workflow by PR Type

**Bug Fix PRs:**
1. Find the original issue: Check PR description for `Fixes #NNN` or `Closes #NNN`
2. Search for similar crashes/bugs: Use error message keywords
3. Check file history: `git log --grep="fix\|bug" -- affected_file.cpp`
4. Look for related fixes: Search closed PRs with "fix" + file path

**Feature Addition PRs:**
1. Search for RFC: `gh search issues "[RFC]" "feature name" --label RFC`
2. Find design discussions: Check Discourse for feature proposal
3. Look for prior attempts: Search closed PRs with feature keywords
4. Check documentation: Look for design docs in `llvm/rfcs` repo

**Refactoring/NFC PRs:**
1. Find code history: `git blame` to understand original design
2. Search for related refactorings: grep commit history for "refactor\|cleanup\|NFC"
3. Check for comments: Look for TODO/FIXME in changed code
4. Find motivation: Search mailing list for discussions about code quality

**Performance PRs:**
1. Find benchmark discussions: Search Discourse for performance topics
2. Look for regression reports: Search issues for "performance\|slow\|regression"
3. Check related optimizations: Search for similar pass names or techniques
4. Find motivation: Look for compile-time or runtime improvement discussions

## 2. Code Review Checklist

### Correctness
- [ ] Logic is sound and handles edge cases
- [ ] No obvious bugs or logic errors
- [ ] Proper error handling
- [ ] Thread safety considerations (if applicable)
- [ ] Memory management is correct (no leaks, use-after-free, etc.)

### Performance
- [ ] No performance regressions
- [ ] Algorithms are efficient
- [ ] Control flow changes don't corrupt performance profile data
- [ ] Debug information remains valid for branches and calls

### Code Quality
- [ ] Follows LLVM coding standards
- [ ] Appropriate use of LLVM data structures and APIs
- [ ] No reinventing existing LLVM utilities
- [ ] Code is readable and maintainable
- [ ] Proper use of assertions and error messages

### Testing
- [ ] Adequate test coverage for new code
- [ ] Tests use appropriate frameworks (lit, FileCheck, gtest)
- [ ] Tests are clear and maintainable
- [ ] Edge cases are tested
- [ ] Negative test cases included

### Documentation
- [ ] Public APIs have Doxygen comments
- [ ] Complex logic has explanatory comments
- [ ] File headers are present
- [ ] Release notes updated (if user-facing)

### Subproject-Specific Checks

#### LLVM Core
- [ ] Pass infrastructure used correctly
- [ ] IR transformations are valid
- [ ] Control flow changes preserve correctness
- [ ] Target-specific code is properly isolated

#### Clang
- [ ] AST modifications maintain consistency
- [ ] Diagnostics are clear and actionable
- [ ] Language standard conformance
- [ ] Driver changes are backward compatible

#### MLIR
- [ ] Dialect conventions followed
- [ ] Operation definitions are complete
- [ ] Verifiers are thorough
- [ ] Pattern rewrites are correct

#### Flang
- [ ] Fortran standard compliance
- [ ] FIR generation is correct
- [ ] Runtime library interactions are safe

#### LLDB
- [ ] Naming follows LLDB conventions (snake_case variables, UpperCamelCase functions)
- [ ] No RTTI or exceptions
- [ ] Python bindings updated (if API changes)
- [ ] Thread safety in debugger operations

#### OpenMP/Offload
- [ ] OpenMP specification compliance
- [ ] Device code generation correctness
- [ ] Runtime library compatibility
- [ ] Host-device data transfer is correct

### Build System
- [ ] CMake changes are correct
- [ ] Dependencies properly specified
- [ ] Cross-platform compatibility
- [ ] No unnecessary dependencies added

### Library Layering
- [ ] No circular dependencies introduced
- [ ] Proper layering maintained
- [ ] Headers include only what they need
- [ ] Forward declarations used appropriately

## 3. Review Communication Guidelines

### Be Constructive
- Explain **why** something should change, not just **what**
- Provide code examples or links to similar code
- Distinguish between blocking issues and suggestions
- Acknowledge good practices and clever solutions

### Be Specific
- Reference specific line numbers
- Quote problematic code
- Suggest concrete alternatives
- Link to relevant documentation or standards

### Be Respectful
- Assume good intent
- Focus on the code, not the person
- Use "we" instead of "you"
- Thank contributors for their work

### Review Priorities
1. **Critical**: Correctness issues, security problems, breaking changes
2. **Important**: Performance issues, API design, testing gaps
3. **Nice-to-have**: Style improvements, refactoring suggestions, documentation enhancements

## 4. Common LLVM PR Patterns

### Good Patterns ✅
- Small, focused changes with clear purpose
- Comprehensive test coverage
- Self-contained commits that build independently
- Clear commit messages following LLVM conventions
- NFC (No Functional Change) commits for refactoring

### Red Flags 🚩
- Mixing refactoring with functional changes
- Missing tests for new functionality
- Large, monolithic changes without explanation
- Breaking API changes without migration path
- Commented-out code or debug printfs
- Use of `using namespace std` in headers
- Platform-specific code without portability layer

## 5. PR Summary Template

Use this template when summarizing a PR:

```markdown
## Summary
[Brief one-paragraph description]

## Changes
- **Component 1**: [What changed and why]
- **Component 2**: [What changed and why]

## Impact
- **Breaking Changes**: [Yes/No - describe if yes]
- **Performance**: [Impact on compile time, runtime, memory]
- **API Changes**: [What APIs were added/modified/removed]

## Testing
- [Test files added/modified]
- [Coverage assessment]

## Review Notes
- **Strengths**: [What's well done]
- **Concerns**: [Issues that need attention]
- **Questions**: [Clarifications needed]

## Recommendation
[Approve / Request Changes / Needs Discussion]
```

## 6. Subproject Priority Areas

When reviewing, pay special attention to these areas per subproject:

### LLVM
- Control flow modifications → profile data corruption
- IR transformations → validity and optimization pipeline
- Target code generation → correctness and ABI compliance

### Clang
- Parser/Sema changes → standard compliance
- Code generation → debug info accuracy
- Diagnostics → clarity and actionability

### MLIR
- Dialect design → consistency and usability
- Pass infrastructure → correctness and composability
- Type system → soundness

### Flang
- Parser → Fortran standard compliance
- Lowering → correctness to FIR/LLVM IR
- Runtime → performance and correctness

### LLDB
- Expression evaluation → language feature support
- Debugger correctness → breakpoints, watchpoints, stepping
- Remote debugging → protocol correctness

## 7. Quick Review Workflow

1. **Read the PR description** - Understand the intent
2. **Check the files changed** - Assess scope and impact
3. **Review tests first** - Understand expected behavior
4. **Review implementation** - Check correctness and quality
5. **Verify documentation** - Ensure changes are documented
6. **Run mental test cases** - Think of edge cases
7. **Check related code** - Look for similar patterns in codebase
8. **Provide feedback** - Clear, actionable, respectful

## 8. Automated Checks to Verify

- [ ] CI/CD pipelines passing
- [ ] clang-format applied
- [ ] No new compiler warnings
- [ ] Sanitizers clean (ASan, UBSan, MSan)
- [ ] Cross-platform builds succeed

## Remember

- **Prioritize correctness** over style
- **Value consistency** with existing code
- **Appreciate contributions** - every PR makes LLVM better
- **Learn from reviews** - both giving and receiving
- **When in doubt**, ask questions rather than making assumptions

Link to official guidelines:
- LLVM Coding Standards: https://llvm.org/docs/CodingStandards.html
- LLVM Developer Policy: https://llvm.org/docs/DeveloperPolicy.html
- Code Review Guidelines: https://llvm.org/docs/CodeReview.html
