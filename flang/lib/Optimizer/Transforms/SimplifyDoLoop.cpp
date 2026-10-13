//===-- SimplifyDoLoop.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace fir {
#define GEN_PASS_DEF_SIMPLIFYDOLOOP
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {
/// Strip fir.convert chains to find the root SSA value.
static mlir::Value stripConverts(mlir::Value val) {
  while (auto conv = val.getDefiningOp<fir::ConvertOp>())
    val = conv.getValue();
  return val;
}

/// The memory reference backing `ref` together with every fir.declare that
/// aliases it, so that accesses spelled through a declare are also found.
static llvm::SmallVector<mlir::Value, 2> collectAliases(mlir::Value ref) {
  llvm::SmallVector<mlir::Value, 2> aliases;
  mlir::Value base = ref;
  if (auto decl = ref.getDefiningOp<fir::DeclareOp>())
    base = decl.getMemref();
  aliases.push_back(base);
  for (mlir::Operation *user : base.getUsers())
    if (auto decl = mlir::dyn_cast<fir::DeclareOp>(user))
      aliases.push_back(decl.getResult());
  return aliases;
}

/// The store that spills `loop`'s induction variable into the Fortran variable
/// named by the DO statement, searched for in the loop's immediate body.
static fir::StoreOp findIVSpill(fir::DoLoopOp loop) {
  mlir::Value iv = loop.getInductionVar();
  for (mlir::Operation &op : loop.getBody()->without_terminator())
    if (auto store = mlir::dyn_cast<fir::StoreOp>(op))
      if (stripConverts(store.getValue()) == iv)
        return store;
  return {};
}

/// True when the slot is only loaded from, stored into, or re-declared. Any
/// other user may observe the slot through an alias this pass cannot see, in
/// which case the spill has to stay.
static bool isSlotContained(llvm::ArrayRef<mlir::Value> aliases) {
  for (mlir::Value alias : aliases)
    for (mlir::Operation *user : alias.getUsers()) {
      if (auto store = mlir::dyn_cast<fir::StoreOp>(user)) {
        if (store.getMemref() != alias)
          return false;
        continue;
      }
      if (!mlir::isa<fir::LoadOp, fir::DeclareOp>(user))
        return false;
    }
  return true;
}

/// Replace reads of the induction variable's slot inside `loop` with the
/// spilled value and drop the spill. Does nothing unless the spill is the only
/// writer the loop can reach and every read sees it. Returns the slot whose
/// reads were forwarded, so the caller can revisit it once every loop has been
/// handled.
static mlir::Value dematerializeIV(fir::DoLoopOp loop,
                                   mlir::DominanceInfo &domInfo) {
  fir::StoreOp spill = findIVSpill(loop);
  if (!spill)
    return {};

  llvm::SmallVector<mlir::Value, 2> aliases =
      collectAliases(spill.getMemref());
  if (!isSlotContained(aliases))
    return {};

  llvm::SmallVector<fir::LoadOp> reads;
  bool unsafe = false;
  loop.walk([&](mlir::Operation *op) {
    if (auto store = mlir::dyn_cast<fir::StoreOp>(op)) {
      // A second writer means a read need not see the induction variable.
      if (store != spill && llvm::is_contained(aliases, store.getMemref()))
        unsafe = true;
      return;
    }
    if (auto load = mlir::dyn_cast<fir::LoadOp>(op)) {
      if (!llvm::is_contained(aliases, load.getMemref()))
        return;
      // A read the spill does not dominate observes an earlier iteration.
      if (domInfo.properlyDominates(spill.getOperation(), load.getOperation()))
        reads.push_back(load);
      else
        unsafe = true;
    }
  });
  if (unsafe)
    return {};

  // fir.store requires the stored value and the referent to have the same
  // type, so the spilled value can replace a read without a conversion.
  mlir::Value spilled = spill.getValue();
  for (fir::LoadOp load : reads) {
    load.replaceAllUsesWith(spilled);
    load.erase();
  }
  spill.erase();
  return aliases.front();
}

/// Drop the writes to a slot that nothing reads any more. Lowering stores the
/// value the Fortran variable is defined to hold once the loop finishes, and
/// for an inner loop that store sits in the enclosing loop's body; removing it
/// when it cannot be observed is what leaves a perfectly nested loop nest.
///
/// Only a local allocation qualifies. A DO statement may name a dummy argument,
/// and the caller reads the variable after the call, so a write to its storage
/// is observable however the procedure itself uses it.
static void removeUnreadSlotStores(mlir::Value slot) {
  llvm::SmallVector<mlir::Value, 2> aliases = collectAliases(slot);
  if (!aliases.front().getDefiningOp<fir::AllocaOp>())
    return;
  if (!isSlotContained(aliases))
    return;

  llvm::SmallVector<fir::StoreOp> writes;
  for (mlir::Value alias : aliases)
    for (mlir::Operation *user : alias.getUsers()) {
      if (mlir::isa<fir::LoadOp>(user))
        return;
      if (auto store = mlir::dyn_cast<fir::StoreOp>(user))
        writes.push_back(store);
    }

  for (fir::StoreOp store : writes)
    store.erase();
}

class SimplifyDoLoopPass
    : public fir::impl::SimplifyDoLoopBase<SimplifyDoLoopPass> {
public:
  void runOnOperation() override {
    llvm::SmallVector<fir::DoLoopOp> loops;
    getOperation().walk([&](fir::DoLoopOp loop) { loops.push_back(loop); });

    mlir::DominanceInfo &domInfo = getAnalysis<mlir::DominanceInfo>();
    llvm::SetVector<mlir::Value> slots;
    for (fir::DoLoopOp loop : loops)
      if (mlir::Value slot = dematerializeIV(loop, domInfo))
        slots.insert(slot);

    // Deferred until every loop has been handled: an outer loop's variable is
    // read by the inner loop's body, so it only becomes unread once the inner
    // loop has had its own reads forwarded.
    for (mlir::Value slot : slots)
      removeUnreadSlotStores(slot);
  }
};
} // namespace
