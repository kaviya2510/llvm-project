//===-- WebAssemblyCFGStackify.cpp - CFG Stackification -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a CFG stacking pass.
///
/// This pass inserts BLOCK, LOOP, TRY, and TRY_TABLE markers to mark the start
/// of scopes, since scope boundaries serve as the labels for WebAssembly's
/// control transfers.
///
/// This is sufficient to convert arbitrary CFGs into a form that works on
/// WebAssembly, provided that all loops are single-entry.
///
/// In case we use exceptions, this pass also fixes mismatches in unwind
/// destinations created during transforming CFG into wasm structured format.
///
//===----------------------------------------------------------------------===//

#include "Utils/WebAssemblyTypeUtilities.h"
#include "WebAssembly.h"
#include "WebAssemblyExceptionInfo.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySortRegion.h"
#include "WebAssemblySubtarget.h"
#include "WebAssemblyTargetMachine.h"
#include "WebAssemblyUtilities.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/WasmEHFuncInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;
using WebAssembly::SortRegionInfo;

#define DEBUG_TYPE "wasm-cfg-stackify"

STATISTIC(NumCallUnwindMismatches, "Number of call unwind mismatches found");
STATISTIC(NumCatchUnwindMismatches, "Number of catch unwind mismatches found");

namespace {
class WebAssemblyCFGStackify final : public MachineFunctionPass {
  MachineDominatorTree *MDT;

  StringRef getPassName() const override { return "WebAssembly CFG Stackify"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<WebAssemblyExceptionInfo>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  // For each block whose label represents the end of a scope, record the block
  // which holds the beginning of the scope. This will allow us to quickly skip
  // over scoped regions when walking blocks.
  SmallVector<MachineBasicBlock *, 8> ScopeTops;
  void updateScopeTops(MachineBasicBlock *Begin, MachineBasicBlock *End) {
    int BeginNo = Begin->getNumber();
    int EndNo = End->getNumber();
    if (!ScopeTops[EndNo] || ScopeTops[EndNo]->getNumber() > BeginNo)
      ScopeTops[EndNo] = Begin;
  }

  // Placing markers.
  void placeMarkers(MachineFunction &MF);
  void placeBlockMarker(MachineBasicBlock &MBB);
  void placeLoopMarker(MachineBasicBlock &MBB);
  void placeTryMarker(MachineBasicBlock &MBB);
  void placeTryTableMarker(MachineBasicBlock &MBB);

  // Unwind mismatch fixing for exception handling
  // - Common functions
  bool fixCallUnwindMismatches(MachineFunction &MF);
  bool fixCatchUnwindMismatches(MachineFunction &MF);
  void recalculateScopeTops(MachineFunction &MF);
  // - Legacy EH
  void addNestedTryDelegate(MachineInstr *RangeBegin, MachineInstr *RangeEnd,
                            MachineBasicBlock *UnwindDest);
  void removeUnnecessaryInstrs(MachineFunction &MF);
  // - Standard EH (exnref)
  void addNestedTryTable(MachineInstr *RangeBegin, MachineInstr *RangeEnd,
                         MachineBasicBlock *UnwindDest);
  MachineBasicBlock *getTrampolineBlock(MachineBasicBlock *UnwindDest);

  // Wrap-up
  using EndMarkerInfo =
      std::pair<const MachineBasicBlock *, const MachineInstr *>;
  unsigned getBranchDepth(const SmallVectorImpl<EndMarkerInfo> &Stack,
                          const MachineBasicBlock *MBB);
  unsigned getDelegateDepth(const SmallVectorImpl<EndMarkerInfo> &Stack,
                            const MachineBasicBlock *MBB);
  unsigned getRethrowDepth(const SmallVectorImpl<EndMarkerInfo> &Stack,
                           const MachineBasicBlock *EHPadToRethrow);
  void rewriteDepthImmediates(MachineFunction &MF);
  void fixEndsAtEndOfFunction(MachineFunction &MF);
  void cleanupFunctionData(MachineFunction &MF);

  // For each BLOCK|LOOP|TRY|TRY_TABLE, the corresponding
  // END_(BLOCK|LOOP|TRY|TRY_TABLE) or DELEGATE (in case of TRY).
  DenseMap<const MachineInstr *, MachineInstr *> BeginToEnd;
  // For each END_(BLOCK|LOOP|TRY|TRY_TABLE) or DELEGATE, the corresponding
  // BLOCK|LOOP|TRY|TRY_TABLE.
  DenseMap<const MachineInstr *, MachineInstr *> EndToBegin;
  // <TRY marker, EH pad> map
  DenseMap<const MachineInstr *, MachineBasicBlock *> TryToEHPad;
  // <EH pad, TRY marker> map
  DenseMap<const MachineBasicBlock *, MachineInstr *> EHPadToTry;

  DenseMap<const MachineBasicBlock *, MachineBasicBlock *>
      UnwindDestToTrampoline;

  // We need an appendix block to place 'end_loop' or 'end_try' marker when the
  // loop / exception bottom block is the last block in a function
  MachineBasicBlock *AppendixBB = nullptr;
  MachineBasicBlock *getAppendixBlock(MachineFunction &MF) {
    if (!AppendixBB) {
      AppendixBB = MF.CreateMachineBasicBlock();
      // Give it a fake predecessor so that AsmPrinter prints its label.
      AppendixBB->addSuccessor(AppendixBB);
      // If the caller trampoline BB exists, insert the appendix BB before it.
      // Otherwise insert it at the end of the function.
      if (CallerTrampolineBB)
        MF.insert(CallerTrampolineBB->getIterator(), AppendixBB);
      else
        MF.push_back(AppendixBB);
    }
    return AppendixBB;
  }

  // Create a caller-dedicated trampoline BB to be used for fixing unwind
  // mismatches where the unwind destination is the caller.
  MachineBasicBlock *CallerTrampolineBB = nullptr;
  MachineBasicBlock *getCallerTrampolineBlock(MachineFunction &MF) {
    if (!CallerTrampolineBB) {
      CallerTrampolineBB = MF.CreateMachineBasicBlock();
      MF.push_back(CallerTrampolineBB);
    }
    return CallerTrampolineBB;
  }

  // Before running rewriteDepthImmediates function, 'delegate' has a BB as its
  // destination operand. getFakeCallerBlock() returns a fake BB that will be
  // used for the operand when 'delegate' needs to rethrow to the caller. This
  // will be rewritten as an immediate value that is the number of block depths
  // + 1 in rewriteDepthImmediates, and this fake BB will be removed at the end
  // of the pass.
  MachineBasicBlock *FakeCallerBB = nullptr;
  MachineBasicBlock *getFakeCallerBlock(MachineFunction &MF) {
    if (!FakeCallerBB)
      FakeCallerBB = MF.CreateMachineBasicBlock();
    return FakeCallerBB;
  }

  // Helper functions to register / unregister scope information created by
  // marker instructions.
  void registerScope(MachineInstr *Begin, MachineInstr *End);
  void registerTryScope(MachineInstr *Begin, MachineInstr *End,
                        MachineBasicBlock *EHPad);
  void unregisterScope(MachineInstr *Begin);

public:
  static char ID; // Pass identification, replacement for typeid
  WebAssemblyCFGStackify() : MachineFunctionPass(ID) {}
  ~WebAssemblyCFGStackify() override { releaseMemory(); }
  void releaseMemory() override;
};
} // end anonymous namespace

char WebAssemblyCFGStackify::ID = 0;
INITIALIZE_PASS(
    WebAssemblyCFGStackify, DEBUG_TYPE,
    "Insert BLOCK/LOOP/TRY/TRY_TABLE markers for WebAssembly scopes", false,
    false)

FunctionPass *llvm::createWebAssemblyCFGStackify() {
  return new WebAssemblyCFGStackify();
}

/// Test whether Pred has any terminators explicitly branching to MBB, as
/// opposed to falling through. Note that it's possible (eg. in unoptimized
/// code) for a branch instruction to both branch to a block and fallthrough
/// to it, so we check the actual branch operands to see if there are any
/// explicit mentions.
static bool explicitlyBranchesTo(MachineBasicBlock *Pred,
                                 MachineBasicBlock *MBB) {
  for (MachineInstr &MI : Pred->terminators())
    for (MachineOperand &MO : MI.explicit_operands())
      if (MO.isMBB() && MO.getMBB() == MBB)
        return true;
  return false;
}

// Returns an iterator to the earliest position possible within the MBB,
// satisfying the restrictions given by BeforeSet and AfterSet. BeforeSet
// contains instructions that should go before the marker, and AfterSet contains
// ones that should go after the marker. In this function, AfterSet is only
// used for validation checking.
template <typename Container>
static MachineBasicBlock::iterator
getEarliestInsertPos(MachineBasicBlock *MBB, const Container &BeforeSet,
                     const Container &AfterSet) {
  auto InsertPos = MBB->end();
  while (InsertPos != MBB->begin()) {
    if (BeforeSet.count(&*std::prev(InsertPos))) {
#ifndef NDEBUG
      // Validation check
      for (auto Pos = InsertPos, E = MBB->begin(); Pos != E; --Pos)
        assert(!AfterSet.count(&*std::prev(Pos)));
#endif
      break;
    }
    --InsertPos;
  }
  return InsertPos;
}

// Returns an iterator to the latest position possible within the MBB,
// satisfying the restrictions given by BeforeSet and AfterSet. BeforeSet
// contains instructions that should go before the marker, and AfterSet contains
// ones that should go after the marker. In this function, BeforeSet is only
// used for validation checking.
template <typename Container>
static MachineBasicBlock::iterator
getLatestInsertPos(MachineBasicBlock *MBB, const Container &BeforeSet,
                   const Container &AfterSet) {
  auto InsertPos = MBB->begin();
  while (InsertPos != MBB->end()) {
    if (AfterSet.count(&*InsertPos)) {
#ifndef NDEBUG
      // Validation check
      for (auto Pos = InsertPos, E = MBB->end(); Pos != E; ++Pos)
        assert(!BeforeSet.count(&*Pos));
#endif
      break;
    }
    ++InsertPos;
  }
  return InsertPos;
}

void WebAssemblyCFGStackify::registerScope(MachineInstr *Begin,
                                           MachineInstr *End) {
  BeginToEnd[Begin] = End;
  EndToBegin[End] = Begin;
}

// When 'End' is not an 'end_try' but a 'delegate', EHPad is nullptr.
void WebAssemblyCFGStackify::registerTryScope(MachineInstr *Begin,
                                              MachineInstr *End,
                                              MachineBasicBlock *EHPad) {
  registerScope(Begin, End);
  TryToEHPad[Begin] = EHPad;
  EHPadToTry[EHPad] = Begin;
}

void WebAssemblyCFGStackify::unregisterScope(MachineInstr *Begin) {
  assert(BeginToEnd.count(Begin));
  MachineInstr *End = BeginToEnd[Begin];
  assert(EndToBegin.count(End));
  BeginToEnd.erase(Begin);
  EndToBegin.erase(End);
  MachineBasicBlock *EHPad = TryToEHPad.lookup(Begin);
  if (EHPad) {
    assert(EHPadToTry.count(EHPad));
    TryToEHPad.erase(Begin);
    EHPadToTry.erase(EHPad);
  }
}

/// Insert a BLOCK marker for branches to MBB (if needed).
// TODO Consider a more generalized way of handling block (and also loop and
// try) signatures when we implement the multi-value proposal later.
void WebAssemblyCFGStackify::placeBlockMarker(MachineBasicBlock &MBB) {
  assert(!MBB.isEHPad());
  MachineFunction &MF = *MBB.getParent();
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  const auto &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();

  // First compute the nearest common dominator of all forward non-fallthrough
  // predecessors so that we minimize the time that the BLOCK is on the stack,
  // which reduces overall stack height.
  MachineBasicBlock *Header = nullptr;
  bool IsBranchedTo = false;
  int MBBNumber = MBB.getNumber();
  for (MachineBasicBlock *Pred : MBB.predecessors()) {
    if (Pred->getNumber() < MBBNumber) {
      Header = Header ? MDT->findNearestCommonDominator(Header, Pred) : Pred;
      if (explicitlyBranchesTo(Pred, &MBB))
        IsBranchedTo = true;
    }
  }
  if (!Header)
    return;
  if (!IsBranchedTo)
    return;

  assert(&MBB != &MF.front() && "Header blocks shouldn't have predecessors");
  MachineBasicBlock *LayoutPred = MBB.getPrevNode();

  // If the nearest common dominator is inside a more deeply nested context,
  // walk out to the nearest scope which isn't more deeply nested.
  for (MachineFunction::iterator I(LayoutPred), E(Header); I != E; --I) {
    if (MachineBasicBlock *ScopeTop = ScopeTops[I->getNumber()]) {
      if (ScopeTop->getNumber() > Header->getNumber()) {
        // Skip over an intervening scope.
        I = std::next(ScopeTop->getIterator());
      } else {
        // We found a scope level at an appropriate depth.
        Header = ScopeTop;
        break;
      }
    }
  }

  // Decide where in MBB to put the BLOCK.

  // Instructions that should go before the BLOCK.
  SmallPtrSet<const MachineInstr *, 4> BeforeSet;
  // Instructions that should go after the BLOCK.
  SmallPtrSet<const MachineInstr *, 4> AfterSet;
  for (const auto &MI : *Header) {
    // If there is a previously placed LOOP marker and the bottom block of the
    // loop is above MBB, it should be after the BLOCK, because the loop is
    // nested in this BLOCK. Otherwise it should be before the BLOCK.
    if (MI.getOpcode() == WebAssembly::LOOP) {
      auto *LoopBottom = BeginToEnd[&MI]->getParent()->getPrevNode();
      if (MBB.getNumber() > LoopBottom->getNumber())
        AfterSet.insert(&MI);
#ifndef NDEBUG
      else
        BeforeSet.insert(&MI);
#endif
    }

    // If there is a previously placed BLOCK/TRY/TRY_TABLE marker and its
    // corresponding END marker is before the current BLOCK's END marker, that
    // should be placed after this BLOCK. Otherwise it should be placed before
    // this BLOCK marker.
    if (MI.getOpcode() == WebAssembly::BLOCK ||
        MI.getOpcode() == WebAssembly::TRY ||
        MI.getOpcode() == WebAssembly::TRY_TABLE) {
      if (BeginToEnd[&MI]->getParent()->getNumber() <= MBB.getNumber())
        AfterSet.insert(&MI);
#ifndef NDEBUG
      else
        BeforeSet.insert(&MI);
#endif
    }

#ifndef NDEBUG
    // All END_(BLOCK|LOOP|TRY|TRY_TABLE) markers should be before the BLOCK.
    if (MI.getOpcode() == WebAssembly::END_BLOCK ||
        MI.getOpcode() == WebAssembly::END_LOOP ||
        MI.getOpcode() == WebAssembly::END_TRY ||
        MI.getOpcode() == WebAssembly::END_TRY_TABLE)
      BeforeSet.insert(&MI);
#endif

    // Terminators should go after the BLOCK.
    if (MI.isTerminator())
      AfterSet.insert(&MI);
  }

  // Local expression tree should go after the BLOCK.
  for (auto I = Header->getFirstTerminator(), E = Header->begin(); I != E;
       --I) {
    if (std::prev(I)->isDebugInstr() || std::prev(I)->isPosition())
      continue;
    if (WebAssembly::isChild(*std::prev(I), MFI))
      AfterSet.insert(&*std::prev(I));
    else
      break;
  }

  // Add the BLOCK.
  WebAssembly::BlockType ReturnType = WebAssembly::BlockType::Void;
  auto InsertPos = getLatestInsertPos(Header, BeforeSet, AfterSet);
  MachineInstr *Begin =
      BuildMI(*Header, InsertPos, Header->findDebugLoc(InsertPos),
              TII.get(WebAssembly::BLOCK))
          .addImm(int64_t(ReturnType));

  // Decide where in MBB to put the END_BLOCK.
  BeforeSet.clear();
  AfterSet.clear();
  for (auto &MI : MBB) {
#ifndef NDEBUG
    // END_BLOCK should precede existing LOOP markers.
    if (MI.getOpcode() == WebAssembly::LOOP)
      AfterSet.insert(&MI);
#endif

    // If there is a previously placed END_LOOP marker and the header of the
    // loop is above this block's header, the END_LOOP should be placed after
    // the END_BLOCK, because the loop contains this block. Otherwise the
    // END_LOOP should be placed before the END_BLOCK. The same for END_TRY.
    //
    // Note that while there can be existing END_TRYs, there can't be
    // END_TRY_TABLEs; END_TRYs are placed when its corresponding EH pad is
    // processed, so they are placed below MBB (EH pad) in placeTryMarker. But
    // END_TRY_TABLE is placed like a END_BLOCK, so they can't be here already.
    if (MI.getOpcode() == WebAssembly::END_LOOP ||
        MI.getOpcode() == WebAssembly::END_TRY) {
      if (EndToBegin[&MI]->getParent()->getNumber() >= Header->getNumber())
        BeforeSet.insert(&MI);
#ifndef NDEBUG
      else
        AfterSet.insert(&MI);
#endif
    }
  }

  // Mark the end of the block.
  InsertPos = getEarliestInsertPos(&MBB, BeforeSet, AfterSet);
  MachineInstr *End = BuildMI(MBB, InsertPos, MBB.findPrevDebugLoc(InsertPos),
                              TII.get(WebAssembly::END_BLOCK));
  registerScope(Begin, End);

  // Track the farthest-spanning scope that ends at this point.
  updateScopeTops(Header, &MBB);
}

/// Insert a LOOP marker for a loop starting at MBB (if it's a loop header).
void WebAssemblyCFGStackify::placeLoopMarker(MachineBasicBlock &MBB) {
  MachineFunction &MF = *MBB.getParent();
  const auto &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  const auto &WEI = getAnalysis<WebAssemblyExceptionInfo>();
  SortRegionInfo SRI(MLI, WEI);
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();

  MachineLoop *Loop = MLI.getLoopFor(&MBB);
  if (!Loop || Loop->getHeader() != &MBB)
    return;

  // The operand of a LOOP is the first block after the loop. If the loop is the
  // bottom of the function, insert a dummy block at the end.
  MachineBasicBlock *Bottom = SRI.getBottom(Loop);
  auto Iter = std::next(Bottom->getIterator());
  if (Iter == MF.end()) {
    getAppendixBlock(MF);
    Iter = std::next(Bottom->getIterator());
  }
  MachineBasicBlock *AfterLoop = &*Iter;

  // Decide where in Header to put the LOOP.
  SmallPtrSet<const MachineInstr *, 4> BeforeSet;
  SmallPtrSet<const MachineInstr *, 4> AfterSet;
  for (const auto &MI : MBB) {
    // LOOP marker should be after any existing loop that ends here. Otherwise
    // we assume the instruction belongs to the loop.
    if (MI.getOpcode() == WebAssembly::END_LOOP)
      BeforeSet.insert(&MI);
#ifndef NDEBUG
    else
      AfterSet.insert(&MI);
#endif
  }

  // Mark the beginning of the loop.
  auto InsertPos = getEarliestInsertPos(&MBB, BeforeSet, AfterSet);
  MachineInstr *Begin = BuildMI(MBB, InsertPos, MBB.findDebugLoc(InsertPos),
                                TII.get(WebAssembly::LOOP))
                            .addImm(int64_t(WebAssembly::BlockType::Void));

  // Decide where in MBB to put the END_LOOP.
  BeforeSet.clear();
  AfterSet.clear();
#ifndef NDEBUG
  for (const auto &MI : MBB)
    // Existing END_LOOP markers belong to parent loops of this loop
    if (MI.getOpcode() == WebAssembly::END_LOOP)
      AfterSet.insert(&MI);
#endif

  // Mark the end of the loop (using arbitrary debug location that branched to
  // the loop end as its location).
  InsertPos = getEarliestInsertPos(AfterLoop, BeforeSet, AfterSet);
  DebugLoc EndDL = AfterLoop->pred_empty()
                       ? DebugLoc()
                       : (*AfterLoop->pred_rbegin())->findBranchDebugLoc();
  MachineInstr *End =
      BuildMI(*AfterLoop, InsertPos, EndDL, TII.get(WebAssembly::END_LOOP));
  registerScope(Begin, End);

  assert((!ScopeTops[AfterLoop->getNumber()] ||
          ScopeTops[AfterLoop->getNumber()]->getNumber() < MBB.getNumber()) &&
         "With block sorting the outermost loop for a block should be first.");
  updateScopeTops(&MBB, AfterLoop);
}

void WebAssemblyCFGStackify::placeTryMarker(MachineBasicBlock &MBB) {
  assert(MBB.isEHPad());
  MachineFunction &MF = *MBB.getParent();
  auto &MDT = getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  const auto &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  const auto &WEI = getAnalysis<WebAssemblyExceptionInfo>();
  SortRegionInfo SRI(MLI, WEI);
  const auto &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();

  // Compute the nearest common dominator of all unwind predecessors
  MachineBasicBlock *Header = nullptr;
  int MBBNumber = MBB.getNumber();
  for (auto *Pred : MBB.predecessors()) {
    if (Pred->getNumber() < MBBNumber) {
      Header = Header ? MDT.findNearestCommonDominator(Header, Pred) : Pred;
      assert(!explicitlyBranchesTo(Pred, &MBB) &&
             "Explicit branch to an EH pad!");
    }
  }
  if (!Header)
    return;

  // If this try is at the bottom of the function, insert a dummy block at the
  // end.
  WebAssemblyException *WE = WEI.getExceptionFor(&MBB);
  assert(WE);
  MachineBasicBlock *Bottom = SRI.getBottom(WE);
  auto Iter = std::next(Bottom->getIterator());
  if (Iter == MF.end()) {
    getAppendixBlock(MF);
    Iter = std::next(Bottom->getIterator());
  }
  MachineBasicBlock *Cont = &*Iter;

  // If the nearest common dominator is inside a more deeply nested context,
  // walk out to the nearest scope which isn't more deeply nested.
  for (MachineFunction::iterator I(Bottom), E(Header); I != E; --I) {
    if (MachineBasicBlock *ScopeTop = ScopeTops[I->getNumber()]) {
      if (ScopeTop->getNumber() > Header->getNumber()) {
        // Skip over an intervening scope.
        I = std::next(ScopeTop->getIterator());
      } else {
        // We found a scope level at an appropriate depth.
        Header = ScopeTop;
        break;
      }
    }
  }

  // Decide where in Header to put the TRY.

  // Instructions that should go before the TRY.
  SmallPtrSet<const MachineInstr *, 4> BeforeSet;
  // Instructions that should go after the TRY.
  SmallPtrSet<const MachineInstr *, 4> AfterSet;
  for (const auto &MI : *Header) {
    // If there is a previously placed LOOP marker and the bottom block of the
    // loop is above MBB, it should be after the TRY, because the loop is nested
    // in this TRY. Otherwise it should be before the TRY.
    if (MI.getOpcode() == WebAssembly::LOOP) {
      auto *LoopBottom = BeginToEnd[&MI]->getParent()->getPrevNode();
      if (MBB.getNumber() > LoopBottom->getNumber())
        AfterSet.insert(&MI);
#ifndef NDEBUG
      else
        BeforeSet.insert(&MI);
#endif
    }

    // All previously inserted BLOCK/TRY markers should be after the TRY because
    // they are all nested blocks/trys.
    if (MI.getOpcode() == WebAssembly::BLOCK ||
        MI.getOpcode() == WebAssembly::TRY)
      AfterSet.insert(&MI);

#ifndef NDEBUG
    // All END_(BLOCK/LOOP/TRY) markers should be before the TRY.
    if (MI.getOpcode() == WebAssembly::END_BLOCK ||
        MI.getOpcode() == WebAssembly::END_LOOP ||
        MI.getOpcode() == WebAssembly::END_TRY)
      BeforeSet.insert(&MI);
#endif

    // Terminators should go after the TRY.
    if (MI.isTerminator())
      AfterSet.insert(&MI);
  }

  // If Header unwinds to MBB (= Header contains 'invoke'), the try block should
  // contain the call within it. So the call should go after the TRY. The
  // exception is when the header's terminator is a rethrow instruction, in
  // which case that instruction, not a call instruction before it, is gonna
  // throw.
  MachineInstr *ThrowingCall = nullptr;
  if (MBB.isPredecessor(Header)) {
    auto TermPos = Header->getFirstTerminator();
    if (TermPos == Header->end() ||
        TermPos->getOpcode() != WebAssembly::RETHROW) {
      for (auto &MI : reverse(*Header)) {
        if (MI.isCall()) {
          AfterSet.insert(&MI);
          ThrowingCall = &MI;
          // Possibly throwing calls are usually wrapped by EH_LABEL
          // instructions. We don't want to split them and the call.
          if (MI.getIterator() != Header->begin() &&
              std::prev(MI.getIterator())->isEHLabel()) {
            AfterSet.insert(&*std::prev(MI.getIterator()));
            ThrowingCall = &*std::prev(MI.getIterator());
          }
          break;
        }
      }
    }
  }

  // Local expression tree should go after the TRY.
  // For BLOCK placement, we start the search from the previous instruction of a
  // BB's terminator, but in TRY's case, we should start from the previous
  // instruction of a call that can throw, or a EH_LABEL that precedes the call,
  // because the return values of the call's previous instructions can be
  // stackified and consumed by the throwing call.
  auto SearchStartPt = ThrowingCall ? MachineBasicBlock::iterator(ThrowingCall)
                                    : Header->getFirstTerminator();
  for (auto I = SearchStartPt, E = Header->begin(); I != E; --I) {
    if (std::prev(I)->isDebugInstr() || std::prev(I)->isPosition())
      continue;
    if (WebAssembly::isChild(*std::prev(I), MFI))
      AfterSet.insert(&*std::prev(I));
    else
      break;
  }

  // Add the TRY.
  auto InsertPos = getLatestInsertPos(Header, BeforeSet, AfterSet);
  MachineInstr *Begin =
      BuildMI(*Header, InsertPos, Header->findDebugLoc(InsertPos),
              TII.get(WebAssembly::TRY))
          .addImm(int64_t(WebAssembly::BlockType::Void));

  // Decide where in Cont to put the END_TRY.
  BeforeSet.clear();
  AfterSet.clear();
  for (const auto &MI : *Cont) {
#ifndef NDEBUG
    // END_TRY should precede existing LOOP markers.
    if (MI.getOpcode() == WebAssembly::LOOP)
      AfterSet.insert(&MI);

    // All END_TRY markers placed earlier belong to exceptions that contains
    // this one.
    if (MI.getOpcode() == WebAssembly::END_TRY)
      AfterSet.insert(&MI);
#endif

    // If there is a previously placed END_LOOP marker and its header is after
    // where TRY marker is, this loop is contained within the 'catch' part, so
    // the END_TRY marker should go after that. Otherwise, the whole try-catch
    // is contained within this loop, so the END_TRY should go before that.
    if (MI.getOpcode() == WebAssembly::END_LOOP) {
      // For a LOOP to be after TRY, LOOP's BB should be after TRY's BB; if they
      // are in the same BB, LOOP is always before TRY.
      if (EndToBegin[&MI]->getParent()->getNumber() > Header->getNumber())
        BeforeSet.insert(&MI);
#ifndef NDEBUG
      else
        AfterSet.insert(&MI);
#endif
    }

    // It is not possible for an END_BLOCK to be already in this block.
  }

  // Mark the end of the TRY.
  InsertPos = getEarliestInsertPos(Cont, BeforeSet, AfterSet);
  MachineInstr *End = BuildMI(*Cont, InsertPos, Bottom->findBranchDebugLoc(),
                              TII.get(WebAssembly::END_TRY));
  registerTryScope(Begin, End, &MBB);

  // Track the farthest-spanning scope that ends at this point. We create two
  // mappings: (BB with 'end_try' -> BB with 'try') and (BB with 'catch' -> BB
  // with 'try'). We need to create 'catch' -> 'try' mapping here too because
  // markers should not span across 'catch'. For example, this should not
  // happen:
  //
  // try
  //   block     --|  (X)
  // catch         |
  //   end_block --|
  // end_try
  for (auto *End : {&MBB, Cont})
    updateScopeTops(Header, End);
}

void WebAssemblyCFGStackify::placeTryTableMarker(MachineBasicBlock &MBB) {
  assert(MBB.isEHPad());
  MachineFunction &MF = *MBB.getParent();
  auto &MDT = getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  const auto &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  const auto &WEI = getAnalysis<WebAssemblyExceptionInfo>();
  SortRegionInfo SRI(MLI, WEI);
  const auto &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();

  // Compute the nearest common dominator of all unwind predecessors
  MachineBasicBlock *Header = nullptr;
  int MBBNumber = MBB.getNumber();
  for (auto *Pred : MBB.predecessors()) {
    if (Pred->getNumber() < MBBNumber) {
      Header = Header ? MDT.findNearestCommonDominator(Header, Pred) : Pred;
      assert(!explicitlyBranchesTo(Pred, &MBB) &&
             "Explicit branch to an EH pad!");
    }
  }
  if (!Header)
    return;

  // Unlike the end_try marker, we don't place an end marker at the end of
  // exception bottom, i.e., at the end of the old 'catch' block. But we still
  // consider the try-catch part as a scope when computing ScopeTops.
  WebAssemblyException *WE = WEI.getExceptionFor(&MBB);
  assert(WE);
  MachineBasicBlock *Bottom = SRI.getBottom(WE);
  auto Iter = std::next(Bottom->getIterator());
  if (Iter == MF.end())
    Iter--;
  MachineBasicBlock *Cont = &*Iter;

  // If the nearest common dominator is inside a more deeply nested context,
  // walk out to the nearest scope which isn't more deeply nested.
  for (MachineFunction::iterator I(Bottom), E(Header); I != E; --I) {
    if (MachineBasicBlock *ScopeTop = ScopeTops[I->getNumber()]) {
      if (ScopeTop->getNumber() > Header->getNumber()) {
        // Skip over an intervening scope.
        I = std::next(ScopeTop->getIterator());
      } else {
        // We found a scope level at an appropriate depth.
        Header = ScopeTop;
        break;
      }
    }
  }

  // Decide where in Header to put the TRY_TABLE.

  // Instructions that should go before the TRY_TABLE.
  SmallPtrSet<const MachineInstr *, 4> BeforeSet;
  // Instructions that should go after the TRY_TABLE.
  SmallPtrSet<const MachineInstr *, 4> AfterSet;
  for (const auto &MI : *Header) {
    // If there is a previously placed LOOP marker and the bottom block of the
    // loop is above MBB, it should be after the TRY_TABLE, because the loop is
    // nested in this TRY_TABLE. Otherwise it should be before the TRY_TABLE.
    if (MI.getOpcode() == WebAssembly::LOOP) {
      auto *LoopBottom = BeginToEnd[&MI]->getParent()->getPrevNode();
      if (MBB.getNumber() > LoopBottom->getNumber())
        AfterSet.insert(&MI);
#ifndef NDEBUG
      else
        BeforeSet.insert(&MI);
#endif
    }

    // All previously inserted BLOCK/TRY_TABLE markers should be after the
    // TRY_TABLE because they are all nested blocks/try_tables.
    if (MI.getOpcode() == WebAssembly::BLOCK ||
        MI.getOpcode() == WebAssembly::TRY_TABLE)
      AfterSet.insert(&MI);

#ifndef NDEBUG
    // All END_(BLOCK/LOOP/TRY_TABLE) markers should be before the TRY_TABLE.
    if (MI.getOpcode() == WebAssembly::END_BLOCK ||
        MI.getOpcode() == WebAssembly::END_LOOP ||
        MI.getOpcode() == WebAssembly::END_TRY_TABLE)
      BeforeSet.insert(&MI);
#endif

    // Terminators should go after the TRY_TABLE.
    if (MI.isTerminator())
      AfterSet.insert(&MI);
  }

  // If Header unwinds to MBB (= Header contains 'invoke'), the try_table block
  // should contain the call within it. So the call should go after the
  // TRY_TABLE. The exception is when the header's terminator is a rethrow
  // instruction, in which case that instruction, not a call instruction before
  // it, is gonna throw.
  MachineInstr *ThrowingCall = nullptr;
  if (MBB.isPredecessor(Header)) {
    auto TermPos = Header->getFirstTerminator();
    if (TermPos == Header->end() ||
        TermPos->getOpcode() != WebAssembly::RETHROW) {
      for (auto &MI : reverse(*Header)) {
        if (MI.isCall()) {
          AfterSet.insert(&MI);
          ThrowingCall = &MI;
          // Possibly throwing calls are usually wrapped by EH_LABEL
          // instructions. We don't want to split them and the call.
          if (MI.getIterator() != Header->begin() &&
              std::prev(MI.getIterator())->isEHLabel()) {
            AfterSet.insert(&*std::prev(MI.getIterator()));
            ThrowingCall = &*std::prev(MI.getIterator());
          }
          break;
        }
      }
    }
  }

  // Local expression tree should go after the TRY_TABLE.
  // For BLOCK placement, we start the search from the previous instruction of a
  // BB's terminator, but in TRY_TABLE's case, we should start from the previous
  // instruction of a call that can throw, or a EH_LABEL that precedes the call,
  // because the return values of the call's previous instructions can be
  // stackified and consumed by the throwing call.
  auto SearchStartPt = ThrowingCall ? MachineBasicBlock::iterator(ThrowingCall)
                                    : Header->getFirstTerminator();
  for (auto I = SearchStartPt, E = Header->begin(); I != E; --I) {
    if (std::prev(I)->isDebugInstr() || std::prev(I)->isPosition())
      continue;
    if (WebAssembly::isChild(*std::prev(I), MFI))
      AfterSet.insert(&*std::prev(I));
    else
      break;
  }

  // Add the TRY_TABLE and a BLOCK for the catch destination. We currently
  // generate only one CATCH clause for a TRY_TABLE, so we need one BLOCK for
  // its destination.
  //
  // Header:
  //   block
  //     try_table (catch ... $MBB)
  //       ...
  //
  // MBB:
  //     end_try_table
  //   end_block                 ;; destination of (catch ...)
  //   ... catch handler body ...
  auto InsertPos = getLatestInsertPos(Header, BeforeSet, AfterSet);
  MachineInstrBuilder BlockMIB =
      BuildMI(*Header, InsertPos, Header->findDebugLoc(InsertPos),
              TII.get(WebAssembly::BLOCK));
  auto *Block = BlockMIB.getInstr();
  MachineInstrBuilder TryTableMIB =
      BuildMI(*Header, InsertPos, Header->findDebugLoc(InsertPos),
              TII.get(WebAssembly::TRY_TABLE))
          .addImm(int64_t(WebAssembly::BlockType::Void))
          .addImm(1); // # of catch clauses
  auto *TryTable = TryTableMIB.getInstr();

  // Add a CATCH_*** clause to the TRY_TABLE. These are pseudo instructions
  // following the destination END_BLOCK to simulate block return values,
  // because we currently don't support them.
  const auto &TLI =
      *MF.getSubtarget<WebAssemblySubtarget>().getTargetLowering();
  WebAssembly::BlockType PtrTy =
      TLI.getPointerTy(MF.getDataLayout()) == MVT::i32
          ? WebAssembly::BlockType::I32
          : WebAssembly::BlockType::I64;
  auto *Catch = WebAssembly::findCatch(&MBB);
  switch (Catch->getOpcode()) {
  case WebAssembly::CATCH:
    // CATCH's destination block's return type is the extracted value type,
    // which is currently the thrown value's pointer type for all supported
    // tags.
    BlockMIB.addImm(int64_t(PtrTy));
    TryTableMIB.addImm(wasm::WASM_OPCODE_CATCH);
    for (const auto &Use : Catch->uses()) {
      // The only use operand a CATCH can have is the tag symbol.
      TryTableMIB.addExternalSymbol(Use.getSymbolName());
      break;
    }
    TryTableMIB.addMBB(&MBB);
    break;
  case WebAssembly::CATCH_REF:
    // CATCH_REF's destination block's return type is the extracted value type
    // followed by an exnref, which is (i32, exnref) in our case. We assign the
    // actual multiavlue signature in MCInstLower. MO_CATCH_BLOCK_SIG signals
    // that this operand is used for catch_ref's multivalue destination.
    BlockMIB.addImm(int64_t(WebAssembly::BlockType::Multivalue));
    Block->getOperand(0).setTargetFlags(WebAssemblyII::MO_CATCH_BLOCK_SIG);
    TryTableMIB.addImm(wasm::WASM_OPCODE_CATCH_REF);
    for (const auto &Use : Catch->uses()) {
      TryTableMIB.addExternalSymbol(Use.getSymbolName());
      break;
    }
    TryTableMIB.addMBB(&MBB);
    break;
  case WebAssembly::CATCH_ALL:
    // CATCH_ALL's destination block's return type is void.
    BlockMIB.addImm(int64_t(WebAssembly::BlockType::Void));
    TryTableMIB.addImm(wasm::WASM_OPCODE_CATCH_ALL);
    TryTableMIB.addMBB(&MBB);
    break;
  case WebAssembly::CATCH_ALL_REF:
    // CATCH_ALL_REF's destination block's return type is exnref.
    BlockMIB.addImm(int64_t(WebAssembly::BlockType::Exnref));
    TryTableMIB.addImm(wasm::WASM_OPCODE_CATCH_ALL_REF);
    TryTableMIB.addMBB(&MBB);
    break;
  }

  // Decide where in MBB to put the END_TRY_TABLE, and the END_BLOCK for the
  // CATCH destination.
  BeforeSet.clear();
  AfterSet.clear();
  for (const auto &MI : MBB) {
#ifndef NDEBUG
    // END_TRY_TABLE should precede existing LOOP markers.
    if (MI.getOpcode() == WebAssembly::LOOP)
      AfterSet.insert(&MI);
#endif

    // If there is a previously placed END_LOOP marker and the header of the
    // loop is above this try_table's header, the END_LOOP should be placed
    // after the END_TRY_TABLE, because the loop contains this block. Otherwise
    // the END_LOOP should be placed before the END_TRY_TABLE.
    if (MI.getOpcode() == WebAssembly::END_LOOP) {
      if (EndToBegin[&MI]->getParent()->getNumber() >= Header->getNumber())
        BeforeSet.insert(&MI);
#ifndef NDEBUG
      else
        AfterSet.insert(&MI);
#endif
    }

#ifndef NDEBUG
    // CATCH, CATCH_REF, CATCH_ALL, and CATCH_ALL_REF are pseudo-instructions
    // that simulate the block return value, so they should be placed after the
    // END_TRY_TABLE.
    if (WebAssembly::isCatch(MI.getOpcode()))
      AfterSet.insert(&MI);
#endif
  }

  // Mark the end of the TRY_TABLE and the BLOCK.
  InsertPos = getEarliestInsertPos(&MBB, BeforeSet, AfterSet);
  MachineInstr *EndTryTable =
      BuildMI(MBB, InsertPos, MBB.findPrevDebugLoc(InsertPos),
              TII.get(WebAssembly::END_TRY_TABLE));
  registerTryScope(TryTable, EndTryTable, &MBB);
  MachineInstr *EndBlock =
      BuildMI(MBB, InsertPos, MBB.findPrevDebugLoc(InsertPos),
              TII.get(WebAssembly::END_BLOCK));
  registerScope(Block, EndBlock);

  // Track the farthest-spanning scope that ends at this point.
  // Unlike the end_try, even if we don't put a end marker at the end of catch
  // block, we still have to create two mappings: (BB with 'end_try_table' -> BB
  // with 'try_table') and (BB after the (conceptual) catch block -> BB with
  // 'try_table').
  //
  // This is what can happen if we don't create the latter mapping:
  //
  // Suppoe in the legacy EH we have this code:
  // try
  //   try
  //     code1
  //   catch (a)
  //   end_try
  //   code2
  // catch (b)
  // end_try
  //
  // If we don't create the latter mapping, try_table markers would be placed
  // like this:
  // try_table
  //   code1
  // end_try_table (a)
  // try_table
  //   code2
  // end_try_table (b)
  //
  // This does not reflect the original structure, and more important problem
  // is, in case 'code1' has an unwind mismatch and should unwind to
  // 'end_try_table (b)' rather than 'end_try_table (a)', we don't have a way to
  // make it jump after 'end_try_table (b)' without creating another block. So
  // even if we don't place 'end_try' marker at the end of 'catch' block
  // anymore, we create ScopeTops mapping the same way as the legacy exception,
  // so the resulting code will look like:
  // try_table
  //   try_table
  //     code1
  //   end_try_table (a)
  //   code2
  // end_try_table (b)
  for (auto *End : {&MBB, Cont})
    updateScopeTops(Header, End);
}

void WebAssemblyCFGStackify::removeUnnecessaryInstrs(MachineFunction &MF) {
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();

  // When there is an unconditional branch right before a catch instruction and
  // it branches to the end of end_try marker, we don't need the branch, because
  // if there is no exception, the control flow transfers to that point anyway.
  // bb0:
  //   try
  //     ...
  //     br bb2      <- Not necessary
  // bb1 (ehpad):
  //   catch
  //     ...
  // bb2:            <- Continuation BB
  //   end
  //
  // A more involved case: When the BB where 'end' is located is an another EH
  // pad, the Cont (= continuation) BB is that EH pad's 'end' BB. For example,
  // bb0:
  //   try
  //     try
  //       ...
  //       br bb3      <- Not necessary
  // bb1 (ehpad):
  //     catch
  // bb2 (ehpad):
  //     end
  //   catch
  //     ...
  // bb3:            <- Continuation BB
  //   end
  //
  // When the EH pad at hand is bb1, its matching end_try is in bb2. But it is
  // another EH pad, so bb0's continuation BB becomes bb3. So 'br bb3' in the
  // code can be deleted. This is why we run 'while' until 'Cont' is not an EH
  // pad.
  for (auto &MBB : MF) {
    if (!MBB.isEHPad())
      continue;

    MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
    SmallVector<MachineOperand, 4> Cond;
    MachineBasicBlock *EHPadLayoutPred = MBB.getPrevNode();

    MachineBasicBlock *Cont = &MBB;
    while (Cont->isEHPad()) {
      MachineInstr *Try = EHPadToTry[Cont];
      MachineInstr *EndTry = BeginToEnd[Try];
      // We started from an EH pad, so the end marker cannot be a delegate
      assert(EndTry->getOpcode() != WebAssembly::DELEGATE);
      Cont = EndTry->getParent();
    }

    bool Analyzable = !TII.analyzeBranch(*EHPadLayoutPred, TBB, FBB, Cond);
    // This condition means either
    // 1. This BB ends with a single unconditional branch whose destinaion is
    //    Cont.
    // 2. This BB ends with a conditional branch followed by an unconditional
    //    branch, and the unconditional branch's destination is Cont.
    // In both cases, we want to remove the last (= unconditional) branch.
    if (Analyzable && ((Cond.empty() && TBB && TBB == Cont) ||
                       (!Cond.empty() && FBB && FBB == Cont))) {
      bool ErasedUncondBr = false;
      (void)ErasedUncondBr;
      for (auto I = EHPadLayoutPred->end(), E = EHPadLayoutPred->begin();
           I != E; --I) {
        auto PrevI = std::prev(I);
        if (PrevI->isTerminator()) {
          assert(PrevI->getOpcode() == WebAssembly::BR);
          PrevI->eraseFromParent();
          ErasedUncondBr = true;
          break;
        }
      }
      assert(ErasedUncondBr && "Unconditional branch not erased!");
    }
  }

  // When there are block / end_block markers that overlap with try / end_try
  // markers, and the block and try markers' return types are the same, the
  // block /end_block markers are not necessary, because try / end_try markers
  // also can serve as boundaries for branches.
  // block         <- Not necessary
  //   try
  //     ...
  //   catch
  //     ...
  //   end
  // end           <- Not necessary
  SmallVector<MachineInstr *, 32> ToDelete;
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      if (MI.getOpcode() != WebAssembly::TRY)
        continue;
      MachineInstr *Try = &MI, *EndTry = BeginToEnd[Try];
      if (EndTry->getOpcode() == WebAssembly::DELEGATE)
        continue;

      MachineBasicBlock *TryBB = Try->getParent();
      MachineBasicBlock *Cont = EndTry->getParent();
      int64_t RetType = Try->getOperand(0).getImm();
      for (auto B = Try->getIterator(), E = std::next(EndTry->getIterator());
           B != TryBB->begin() && E != Cont->end() &&
           std::prev(B)->getOpcode() == WebAssembly::BLOCK &&
           E->getOpcode() == WebAssembly::END_BLOCK &&
           std::prev(B)->getOperand(0).getImm() == RetType;
           --B, ++E) {
        ToDelete.push_back(&*std::prev(B));
        ToDelete.push_back(&*E);
      }
    }
  }
  for (auto *MI : ToDelete) {
    if (MI->getOpcode() == WebAssembly::BLOCK)
      unregisterScope(MI);
    MI->eraseFromParent();
  }
}

// When MBB is split into MBB and Split, we should unstackify defs in MBB that
// have their uses in Split.
static void unstackifyVRegsUsedInSplitBB(MachineBasicBlock &MBB,
                                         MachineBasicBlock &Split) {
  MachineFunction &MF = *MBB.getParent();
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  auto &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();
  auto &MRI = MF.getRegInfo();

  for (auto &MI : Split) {
    for (auto &MO : MI.explicit_uses()) {
      if (!MO.isReg() || MO.getReg().isPhysical())
        continue;
      if (MachineInstr *Def = MRI.getUniqueVRegDef(MO.getReg()))
        if (Def->getParent() == &MBB)
          MFI.unstackifyVReg(MO.getReg());
    }
  }

  // In RegStackify, when a register definition is used multiple times,
  //    Reg = INST ...
  //    INST ..., Reg, ...
  //    INST ..., Reg, ...
  //    INST ..., Reg, ...
  //
  // we introduce a TEE, which has the following form:
  //    DefReg = INST ...
  //    TeeReg, Reg = TEE_... DefReg
  //    INST ..., TeeReg, ...
  //    INST ..., Reg, ...
  //    INST ..., Reg, ...
  // with DefReg and TeeReg stackified but Reg not stackified.
  //
  // But the invariant that TeeReg should be stackified can be violated while we
  // unstackify registers in the split BB above. In this case, we convert TEEs
  // into two COPYs. This COPY will be eventually eliminated in ExplicitLocals.
  //    DefReg = INST ...
  //    TeeReg = COPY DefReg
  //    Reg = COPY DefReg
  //    INST ..., TeeReg, ...
  //    INST ..., Reg, ...
  //    INST ..., Reg, ...
  for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
    if (!WebAssembly::isTee(MI.getOpcode()))
      continue;
    Register TeeReg = MI.getOperand(0).getReg();
    Register Reg = MI.getOperand(1).getReg();
    Register DefReg = MI.getOperand(2).getReg();
    if (!MFI.isVRegStackified(TeeReg)) {
      // Now we are not using TEE anymore, so unstackify DefReg too
      MFI.unstackifyVReg(DefReg);
      unsigned CopyOpc =
          WebAssembly::getCopyOpcodeForRegClass(MRI.getRegClass(DefReg));
      BuildMI(MBB, &MI, MI.getDebugLoc(), TII.get(CopyOpc), TeeReg)
          .addReg(DefReg);
      BuildMI(MBB, &MI, MI.getDebugLoc(), TII.get(CopyOpc), Reg).addReg(DefReg);
      MI.eraseFromParent();
    }
  }
}

// Wrap the given range of instructions with a try-delegate that targets
// 'UnwindDest'. RangeBegin and RangeEnd are inclusive.
void WebAssemblyCFGStackify::addNestedTryDelegate(
    MachineInstr *RangeBegin, MachineInstr *RangeEnd,
    MachineBasicBlock *UnwindDest) {
  auto *BeginBB = RangeBegin->getParent();
  auto *EndBB = RangeEnd->getParent();
  MachineFunction &MF = *BeginBB->getParent();
  const auto &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();

  // Local expression tree before the first call of this range should go
  // after the nested TRY.
  SmallPtrSet<const MachineInstr *, 4> AfterSet;
  AfterSet.insert(RangeBegin);
  for (auto I = MachineBasicBlock::iterator(RangeBegin), E = BeginBB->begin();
       I != E; --I) {
    if (std::prev(I)->isDebugInstr() || std::prev(I)->isPosition())
      continue;
    if (WebAssembly::isChild(*std::prev(I), MFI))
      AfterSet.insert(&*std::prev(I));
    else
      break;
  }

  // Create the nested try instruction.
  auto TryPos = getLatestInsertPos(
      BeginBB, SmallPtrSet<const MachineInstr *, 4>(), AfterSet);
  MachineInstr *Try = BuildMI(*BeginBB, TryPos, RangeBegin->getDebugLoc(),
                              TII.get(WebAssembly::TRY))
                          .addImm(int64_t(WebAssembly::BlockType::Void));

  // Create a BB to insert the 'delegate' instruction.
  MachineBasicBlock *DelegateBB = MF.CreateMachineBasicBlock();
  // If the destination of 'delegate' is not the caller, adds the destination to
  // the BB's successors.
  if (UnwindDest != FakeCallerBB)
    DelegateBB->addSuccessor(UnwindDest);

  auto SplitPos = std::next(RangeEnd->getIterator());
  if (SplitPos == EndBB->end()) {
    // If the range's end instruction is at the end of the BB, insert the new
    // delegate BB after the current BB.
    MF.insert(std::next(EndBB->getIterator()), DelegateBB);
    EndBB->addSuccessor(DelegateBB);

  } else {
    // When the split pos is in the middle of a BB, we split the BB into two and
    // put the 'delegate' BB in between. We normally create a split BB and make
    // it a successor of the original BB (CatchAfterSplit == false), but in case
    // the BB is an EH pad and there is a 'catch' after the split pos
    // (CatchAfterSplit == true), we should preserve the BB's property,
    // including that it is an EH pad, in the later part of the BB, where the
    // 'catch' is.
    bool CatchAfterSplit = false;
    if (EndBB->isEHPad()) {
      for (auto I = MachineBasicBlock::iterator(SplitPos), E = EndBB->end();
           I != E; ++I) {
        if (WebAssembly::isCatch(I->getOpcode())) {
          CatchAfterSplit = true;
          break;
        }
      }
    }

    MachineBasicBlock *PreBB = nullptr, *PostBB = nullptr;
    if (!CatchAfterSplit) {
      // If the range's end instruction is in the middle of the BB, we split the
      // BB into two and insert the delegate BB in between.
      // - Before:
      // bb:
      //   range_end
      //   other_insts
      //
      // - After:
      // pre_bb: (previous 'bb')
      //   range_end
      // delegate_bb: (new)
      //   delegate
      // post_bb: (new)
      //   other_insts
      PreBB = EndBB;
      PostBB = MF.CreateMachineBasicBlock();
      MF.insert(std::next(PreBB->getIterator()), PostBB);
      MF.insert(std::next(PreBB->getIterator()), DelegateBB);
      PostBB->splice(PostBB->end(), PreBB, SplitPos, PreBB->end());
      PostBB->transferSuccessors(PreBB);
    } else {
      // - Before:
      // ehpad:
      //   range_end
      //   catch
      //   ...
      //
      // - After:
      // pre_bb: (new)
      //   range_end
      // delegate_bb: (new)
      //   delegate
      // post_bb: (previous 'ehpad')
      //   catch
      //   ...
      assert(EndBB->isEHPad());
      PreBB = MF.CreateMachineBasicBlock();
      PostBB = EndBB;
      MF.insert(PostBB->getIterator(), PreBB);
      MF.insert(PostBB->getIterator(), DelegateBB);
      PreBB->splice(PreBB->end(), PostBB, PostBB->begin(), SplitPos);
      // We don't need to transfer predecessors of the EH pad to 'PreBB',
      // because an EH pad's predecessors are all through unwind edges and they
      // should still unwind to the EH pad, not PreBB.
    }
    unstackifyVRegsUsedInSplitBB(*PreBB, *PostBB);
    PreBB->addSuccessor(DelegateBB);
    PreBB->addSuccessor(PostBB);
  }

  // Add a 'delegate' instruction in the delegate BB created above.
  MachineInstr *Delegate = BuildMI(DelegateBB, RangeEnd->getDebugLoc(),
                                   TII.get(WebAssembly::DELEGATE))
                               .addMBB(UnwindDest);
  registerTryScope(Try, Delegate, nullptr);
}

// Given an unwind destination, return a trampoline BB. A trampoline BB is a
// destination of a nested try_table inserted to fix an unwind mismatch. It
// contains an end_block, which is the target of the try_table, and a throw_ref,
// to rethrow the exception to the right try_table.
// try_table (catch ... )
//   block exnref
//     ...
//     try_table (catch_all_ref N)
//       some code
//     end_try_table
//     ...
//     unreachable
//   end_block                      ;; Trampoline BB
//   throw_ref
// end_try_table
MachineBasicBlock *
WebAssemblyCFGStackify::getTrampolineBlock(MachineBasicBlock *UnwindDest) {
  // We need one trampoline BB per unwind destination, even though there are
  // multiple try_tables target the same unwind destination. If we have already
  // created one for the given UnwindDest, return it.
  auto It = UnwindDestToTrampoline.find(UnwindDest);
  if (It != UnwindDestToTrampoline.end())
    return It->second;

  auto &MF = *UnwindDest->getParent();
  auto &MRI = MF.getRegInfo();
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();

  MachineInstr *Block = nullptr;
  MachineBasicBlock *TrampolineBB = nullptr;
  DebugLoc EndDebugLoc;

  if (UnwindDest == getFakeCallerBlock(MF)) {
    // If the unwind destination is the caller, create a caller-dedicated
    // trampoline BB at the end of the function and wrap the whole function with
    // a block.
    auto BeginPos = MF.begin()->begin();
    while (WebAssembly::isArgument(BeginPos->getOpcode()))
      BeginPos++;
    Block = BuildMI(*MF.begin(), BeginPos, MF.begin()->begin()->getDebugLoc(),
                    TII.get(WebAssembly::BLOCK))
                .addImm(int64_t(WebAssembly::BlockType::Exnref));
    TrampolineBB = getCallerTrampolineBlock(MF);
    MachineBasicBlock *PrevBB = &*std::prev(CallerTrampolineBB->getIterator());
    EndDebugLoc = PrevBB->findPrevDebugLoc(PrevBB->end());
  } else {
    // If the unwind destination is another EH pad, create a trampoline BB for
    // the unwind dest and insert a block instruction right after the target
    // try_table.
    auto *TargetBeginTry = EHPadToTry[UnwindDest];
    auto *TargetEndTry = BeginToEnd[TargetBeginTry];
    auto *TargetBeginBB = TargetBeginTry->getParent();
    auto *TargetEndBB = TargetEndTry->getParent();

    Block = BuildMI(*TargetBeginBB, std::next(TargetBeginTry->getIterator()),
                    TargetBeginTry->getDebugLoc(), TII.get(WebAssembly::BLOCK))
                .addImm(int64_t(WebAssembly::BlockType::Exnref));
    TrampolineBB = MF.CreateMachineBasicBlock();
    EndDebugLoc = TargetEndTry->getDebugLoc();
    MF.insert(TargetEndBB->getIterator(), TrampolineBB);
    TrampolineBB->addSuccessor(UnwindDest);
  }

  // Insert an end_block, catch_all_ref (pseudo instruction), and throw_ref
  // instructions in the trampoline BB.
  MachineInstr *EndBlock =
      BuildMI(TrampolineBB, EndDebugLoc, TII.get(WebAssembly::END_BLOCK));
  auto ExnReg = MRI.createVirtualRegister(&WebAssembly::EXNREFRegClass);
  BuildMI(TrampolineBB, EndDebugLoc, TII.get(WebAssembly::CATCH_ALL_REF))
      .addDef(ExnReg);
  BuildMI(TrampolineBB, EndDebugLoc, TII.get(WebAssembly::THROW_REF))
      .addReg(ExnReg);

  // The trampoline BB's return type is exnref because it is a target of
  // catch_all_ref. But the body type of the block we just created is not. We
  // add an 'unreachable' right before the 'end_block' to make the code valid.
  MachineBasicBlock *TrampolineLayoutPred = TrampolineBB->getPrevNode();
  BuildMI(TrampolineLayoutPred, TrampolineLayoutPred->findBranchDebugLoc(),
          TII.get(WebAssembly::UNREACHABLE));

  registerScope(Block, EndBlock);
  UnwindDestToTrampoline[UnwindDest] = TrampolineBB;
  return TrampolineBB;
}

// Wrap the given range of instructions with a try_table-end_try_table that
// targets 'UnwindDest'. RangeBegin and RangeEnd are inclusive.
void WebAssemblyCFGStackify::addNestedTryTable(MachineInstr *RangeBegin,
                                               MachineInstr *RangeEnd,
                                               MachineBasicBlock *UnwindDest) {
  auto *BeginBB = RangeBegin->getParent();
  auto *EndBB = RangeEnd->getParent();

  MachineFunction &MF = *BeginBB->getParent();
  const auto &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();

  // Get the trampoline BB that the new try_table will unwind to.
  auto *TrampolineBB = getTrampolineBlock(UnwindDest);

  // Local expression tree before the first call of this range should go
  // after the nested TRY_TABLE.
  SmallPtrSet<const MachineInstr *, 4> AfterSet;
  AfterSet.insert(RangeBegin);
  for (auto I = MachineBasicBlock::iterator(RangeBegin), E = BeginBB->begin();
       I != E; --I) {
    if (std::prev(I)->isDebugInstr() || std::prev(I)->isPosition())
      continue;
    if (WebAssembly::isChild(*std::prev(I), MFI))
      AfterSet.insert(&*std::prev(I));
    else
      break;
  }

  // Create the nested try_table instruction.
  auto TryTablePos = getLatestInsertPos(
      BeginBB, SmallPtrSet<const MachineInstr *, 4>(), AfterSet);
  MachineInstr *TryTable =
      BuildMI(*BeginBB, TryTablePos, RangeBegin->getDebugLoc(),
              TII.get(WebAssembly::TRY_TABLE))
          .addImm(int64_t(WebAssembly::BlockType::Void))
          .addImm(1) // # of catch clauses
          .addImm(wasm::WASM_OPCODE_CATCH_ALL_REF)
          .addMBB(TrampolineBB);

  // Create a BB to insert the 'end_try_table' instruction.
  MachineBasicBlock *EndTryTableBB = MF.CreateMachineBasicBlock();
  EndTryTableBB->addSuccessor(TrampolineBB);

  auto SplitPos = std::next(RangeEnd->getIterator());
  if (SplitPos == EndBB->end()) {
    // If the range's end instruction is at the end of the BB, insert the new
    // end_try_table BB after the current BB.
    MF.insert(std::next(EndBB->getIterator()), EndTryTableBB);
    EndBB->addSuccessor(EndTryTableBB);

  } else {
    // When the split pos is in the middle of a BB, we split the BB into two and
    // put the 'end_try_table' BB in between. We normally create a split BB and
    // make it a successor of the original BB (CatchAfterSplit == false), but in
    // case the BB is an EH pad and there is a 'catch' after split pos
    // (CatchAfterSplit == true), we should preserve the BB's property,
    // including that it is an EH pad, in the later part of the BB, where the
    // 'catch' is.
    bool CatchAfterSplit = false;
    if (EndBB->isEHPad()) {
      for (auto I = MachineBasicBlock::iterator(SplitPos), E = EndBB->end();
           I != E; ++I) {
        if (WebAssembly::isCatch(I->getOpcode())) {
          CatchAfterSplit = true;
          break;
        }
      }
    }

    MachineBasicBlock *PreBB = nullptr, *PostBB = nullptr;
    if (!CatchAfterSplit) {
      // If the range's end instruction is in the middle of the BB, we split the
      // BB into two and insert the end_try_table BB in between.
      // - Before:
      // bb:
      //   range_end
      //   other_insts
      //
      // - After:
      // pre_bb: (previous 'bb')
      //   range_end
      // end_try_table_bb: (new)
      //   end_try_table
      // post_bb: (new)
      //   other_insts
      PreBB = EndBB;
      PostBB = MF.CreateMachineBasicBlock();
      MF.insert(std::next(PreBB->getIterator()), PostBB);
      MF.insert(std::next(PreBB->getIterator()), EndTryTableBB);
      PostBB->splice(PostBB->end(), PreBB, SplitPos, PreBB->end());
      PostBB->transferSuccessors(PreBB);
    } else {
      // - Before:
      // ehpad:
      //   range_end
      //   catch
      //   ...
      //
      // - After:
      // pre_bb: (new)
      //   range_end
      // end_try_table_bb: (new)
      //   end_try_table
      // post_bb: (previous 'ehpad')
      //   catch
      //   ...
      assert(EndBB->isEHPad());
      PreBB = MF.CreateMachineBasicBlock();
      PostBB = EndBB;
      MF.insert(PostBB->getIterator(), PreBB);
      MF.insert(PostBB->getIterator(), EndTryTableBB);
      PreBB->splice(PreBB->end(), PostBB, PostBB->begin(), SplitPos);
      // We don't need to transfer predecessors of the EH pad to 'PreBB',
      // because an EH pad's predecessors are all through unwind edges and they
      // should still unwind to the EH pad, not PreBB.
    }
    unstackifyVRegsUsedInSplitBB(*PreBB, *PostBB);
    PreBB->addSuccessor(EndTryTableBB);
    PreBB->addSuccessor(PostBB);
  }

  // Add a 'end_try_table' instruction in the EndTryTable BB created above.
  MachineInstr *EndTryTable = BuildMI(EndTryTableBB, RangeEnd->getDebugLoc(),
                                      TII.get(WebAssembly::END_TRY_TABLE));
  registerTryScope(TryTable, EndTryTable, nullptr);
}

// In the standard (exnref) EH, we fix unwind mismatches by adding a new
// block~end_block inside of the unwind destination try_table~end_try_table:
// try_table ...
//   block exnref                   ;; (new)
//     ...
//     try_table (catch_all_ref N)  ;; (new) to trampoline BB
//       code
//     end_try_table                ;; (new)
//     ...
//   end_block                      ;; (new) trampoline BB
//   throw_ref                      ;; (new)
// end_try_table
//
// To do this, we will create a new BB that will contain the new 'end_block' and
// 'throw_ref' and insert it before the 'end_try_table' BB.
//
// But there are cases when there are 'end_loop'(s) before the 'end_try_table'
// in the same BB. (There can't be 'end_block' before 'end_try_table' in the
// same BB because EH pads can't be directly branched to.) Then after fixing
// unwind mismatches this will create the mismatching markers like below:
// bb0:
//   try_table
//   block exnref
//   ...
//   loop
//   ...
// new_bb:
//   end_block
// end_try_table_bb:
//   end_loop
//   end_try_table
//
// So if an end_try_table BB has an end_loop before the end_try_table, we split
// the BB with the end_loop as a separate BB before the end_try_table BB, so
// that after we fix the unwind mismatch, the code will be like:
// bb0:
//   try_table
//   block exnref
//   ...
//   loop
//   ...
// end_loop_bb:
//   end_loop
// new_bb:
//   end_block
// end_try_table_bb:
//   end_try_table
static void splitEndLoopBB(MachineBasicBlock *EndTryTableBB) {
  auto &MF = *EndTryTableBB->getParent();
  MachineInstr *EndTryTable = nullptr, *EndLoop = nullptr;
  for (auto &MI : reverse(*EndTryTableBB)) {
    if (MI.getOpcode() == WebAssembly::END_TRY_TABLE) {
      EndTryTable = &MI;
      continue;
    }
    if (EndTryTable && MI.getOpcode() == WebAssembly::END_LOOP) {
      EndLoop = &MI;
      break;
    }
  }
  if (!EndLoop)
    return;

  auto *EndLoopBB = MF.CreateMachineBasicBlock();
  MF.insert(EndTryTableBB->getIterator(), EndLoopBB);
  auto SplitPos = std::next(EndLoop->getIterator());
  EndLoopBB->splice(EndLoopBB->end(), EndTryTableBB, EndTryTableBB->begin(),
                    SplitPos);
  EndLoopBB->addSuccessor(EndTryTableBB);
}

bool WebAssemblyCFGStackify::fixCallUnwindMismatches(MachineFunction &MF) {
  // This function is used for both the legacy EH and the standard (exnref) EH,
  // and the reason we have unwind mismatches is the same for the both of them,
  // but the code examples in the comments are going to be different. To make
  // the description less confusing, we write the basically same comments twice,
  // once for the legacy EH and the standard EH.
  //
  // -- Legacy EH --------------------------------------------------------------
  //
  // Linearizing the control flow by placing TRY / END_TRY markers can create
  // mismatches in unwind destinations for throwing instructions, such as calls.
  //
  // We use the 'delegate' instruction to fix the unwind mismatches. 'delegate'
  // instruction delegates an exception to an outer 'catch'. It can target not
  // only 'catch' but all block-like structures including another 'delegate',
  // but with slightly different semantics than branches. When it targets a
  // 'catch', it will delegate the exception to that catch. It is being
  // discussed how to define the semantics when 'delegate''s target is a non-try
  // block: it will either be a validation failure or it will target the next
  // outer try-catch. But anyway our LLVM backend currently does not generate
  // such code. The example below illustrates where the 'delegate' instruction
  // in the middle will delegate the exception to, depending on the value of N.
  // try
  //   try
  //     block
  //       try
  //         try
  //           call @foo
  //         delegate N    ;; Where will this delegate to?
  //       catch           ;; N == 0
  //       end
  //     end               ;; N == 1 (invalid; will not be generated)
  //   delegate            ;; N == 2
  // catch                 ;; N == 3
  // end
  //                       ;; N == 4 (to caller)
  //
  // 1. When an instruction may throw, but the EH pad it will unwind to can be
  //    different from the original CFG.
  //
  // Example: we have the following CFG:
  // bb0:
  //   call @foo    ; if it throws, unwind to bb2
  // bb1:
  //   call @bar    ; if it throws, unwind to bb3
  // bb2 (ehpad):
  //   catch
  //   ...
  // bb3 (ehpad)
  //   catch
  //   ...
  //
  // And the CFG is sorted in this order. Then after placing TRY markers, it
  // will look like: (BB markers are omitted)
  // try
  //   try
  //     call @foo
  //     call @bar   ;; if it throws, unwind to bb3
  //   catch         ;; ehpad (bb2)
  //     ...
  //   end_try
  // catch           ;; ehpad (bb3)
  //   ...
  // end_try
  //
  // Now if bar() throws, it is going to end up in bb2, not bb3, where it is
  // supposed to end up. We solve this problem by wrapping the mismatching call
  // with an inner try-delegate that rethrows the exception to the right
  // 'catch'.
  //
  // try
  //   try
  //     call @foo
  //     try               ;; (new)
  //       call @bar
  //     delegate 1 (bb3)  ;; (new)
  //   catch               ;; ehpad (bb2)
  //     ...
  //   end_try
  // catch                 ;; ehpad (bb3)
  //   ...
  // end_try
  //
  // ---
  // 2. The same as 1, but in this case an instruction unwinds to a caller
  //    function and not another EH pad.
  //
  // Example: we have the following CFG:
  // bb0:
  //   call @foo       ; if it throws, unwind to bb2
  // bb1:
  //   call @bar       ; if it throws, unwind to caller
  // bb2 (ehpad):
  //   catch
  //   ...
  //
  // And the CFG is sorted in this order. Then after placing TRY markers, it
  // will look like:
  // try
  //   call @foo
  //   call @bar     ;; if it throws, unwind to caller
  // catch           ;; ehpad (bb2)
  //   ...
  // end_try
  //
  // Now if bar() throws, it is going to end up in bb2, when it is supposed
  // throw up to the caller. We solve this problem in the same way, but in this
  // case 'delegate's immediate argument is the number of block depths + 1,
  // which means it rethrows to the caller.
  // try
  //   call @foo
  //   try                  ;; (new)
  //     call @bar
  //   delegate 1 (caller)  ;; (new)
  // catch                  ;; ehpad (bb2)
  //   ...
  // end_try
  //
  // Before rewriteDepthImmediates, delegate's argument is a BB. In case of the
  // caller, it will take a fake BB generated by getFakeCallerBlock(), which
  // will be converted to a correct immediate argument later.
  //
  // In case there are multiple calls in a BB that may throw to the caller, they
  // can be wrapped together in one nested try-delegate scope. (In 1, this
  // couldn't happen, because may-throwing instruction there had an unwind
  // destination, i.e., it was an invoke before, and there could be only one
  // invoke within a BB.)
  //
  // -- Standard EH ------------------------------------------------------------
  //
  // Linearizing the control flow by placing TRY / END_TRY_TABLE markers can
  // create mismatches in unwind destinations for throwing instructions, such as
  // calls.
  //
  // We use the a nested 'try_table'~'end_try_table' instruction to fix the
  // unwind mismatches. try_table's catch clauses take an immediate argument
  // that specifics which block we should branch to.
  //
  // 1. When an instruction may throw, but the EH pad it will unwind to can be
  //    different from the original CFG.
  //
  // Example: we have the following CFG:
  // bb0:
  //   call @foo    ; if it throws, unwind to bb2
  // bb1:
  //   call @bar    ; if it throws, unwind to bb3
  // bb2 (ehpad):
  //   catch
  //   ...
  // bb3 (ehpad)
  //   catch
  //   ...
  //
  // And the CFG is sorted in this order. Then after placing TRY_TABLE markers
  // (and BLOCK markers for the TRY_TABLE's destinations), it will look like:
  // (BB markers are omitted)
  // block
  //   try_table (catch ... 0)
  //     block
  //       try_table (catch ... 0)
  //         call @foo
  //         call @bar              ;; if it throws, unwind to bb3
  //       end_try_table
  //     end_block                  ;; ehpad (bb2)
  //     ...
  //   end_try_table
  // end_block                      ;; ehpad (bb3)
  // ...
  //
  // Now if bar() throws, it is going to end up in bb2, not bb3, where it is
  // supposed to end up. We solve this problem by wrapping the mismatching call
  // with an inner try_table~end_try_table that sends the exception to the the
  // 'trampoline' block, which rethrows, or 'bounces' it to the right
  // end_try_table:
  // block
  //   try_table (catch ... 0)
  //     block exnref                       ;; (new)
  //       block
  //         try_table (catch ... 0)
  //           call @foo
  //           try_table (catch_all_ref 2)  ;; (new) to trampoline BB
  //             call @bar
  //           end_try_table                ;; (new)
  //         end_try_table
  //       end_block                        ;; ehpad (bb2)
  //       ...
  //     end_block                          ;; (new) trampoline BB
  //     throw_ref                          ;; (new)
  //   end_try_table
  // end_block                              ;; ehpad (bb3)
  //
  // ---
  // 2. The same as 1, but in this case an instruction unwinds to a caller
  //    function and not another EH pad.
  //
  // Example: we have the following CFG:
  // bb0:
  //   call @foo       ; if it throws, unwind to bb2
  // bb1:
  //   call @bar       ; if it throws, unwind to caller
  // bb2 (ehpad):
  //   catch
  //   ...
  //
  // And the CFG is sorted in this order. Then after placing TRY_TABLE markers
  // (and BLOCK markers for the TRY_TABLE's destinations), it will look like:
  // block
  //   try_table (catch ... 0)
  //     call @foo
  //     call @bar              ;; if it throws, unwind to caller
  //   end_try_table
  // end_block                  ;; ehpad (bb2)
  // ...
  //
  // Now if bar() throws, it is going to end up in bb2, when it is supposed
  // throw up to the caller. We solve this problem in the same way, but in this
  // case 'delegate's immediate argument is the number of block depths + 1,
  // which means it rethrows to the caller.
  // block exnref                       ;; (new)
  //   block
  //     try_table (catch ... 0)
  //       call @foo
  //       try_table (catch_all_ref 2)  ;; (new) to trampoline BB
  //         call @bar
  //       end_try_table                ;; (new)
  //     end_try_table
  //   end_block                        ;; ehpad (bb2)
  //   ...
  // end_block                          ;; (new) caller trampoline BB
  // throw_ref                          ;; (new) throw to the caller
  //
  // Before rewriteDepthImmediates, try_table's catch clauses' argument is a
  // trampoline BB from which we throw_ref the exception to the right
  // end_try_table. In case of the caller, it will take a new caller-dedicated
  // trampoline BB generated by getCallerTrampolineBlock(), which throws the
  // exception to the caller.
  //
  // In case there are multiple calls in a BB that may throw to the caller, they
  // can be wrapped together in one nested try_table-end_try_table scope. (In 1,
  // this couldn't happen, because may-throwing instruction there had an unwind
  // destination, i.e., it was an invoke before, and there could be only one
  // invoke within a BB.)

  SmallVector<const MachineBasicBlock *, 8> EHPadStack;
  // Range of intructions to be wrapped in a new nested try~delegate or
  // try_table~end_try_table. A range exists in a single BB and does not span
  // multiple BBs.
  using TryRange = std::pair<MachineInstr *, MachineInstr *>;
  // In original CFG, <unwind destination BB, a vector of try/try_table ranges>
  DenseMap<MachineBasicBlock *, SmallVector<TryRange, 4>> UnwindDestToTryRanges;

  // Gather possibly throwing calls (i.e., previously invokes) whose current
  // unwind destination is not the same as the original CFG. (Case 1)

  for (auto &MBB : reverse(MF)) {
    bool SeenThrowableInstInBB = false;
    for (auto &MI : reverse(MBB)) {
      if (WebAssembly::isTry(MI.getOpcode()))
        EHPadStack.pop_back();
      else if (WebAssembly::isCatch(MI.getOpcode()))
        EHPadStack.push_back(MI.getParent());

      // In this loop we only gather calls that have an EH pad to unwind. So
      // there will be at most 1 such call (= invoke) in a BB, so after we've
      // seen one, we can skip the rest of BB. Also if MBB has no EH pad
      // successor or MI does not throw, this is not an invoke.
      if (SeenThrowableInstInBB || !MBB.hasEHPadSuccessor() ||
          !WebAssembly::mayThrow(MI))
        continue;
      SeenThrowableInstInBB = true;

      // If the EH pad on the stack top is where this instruction should unwind
      // next, we're good.
      MachineBasicBlock *UnwindDest = nullptr;
      for (auto *Succ : MBB.successors()) {
        // Even though semantically a BB can have multiple successors in case an
        // exception is not caught by a catchpad, the first unwind destination
        // should appear first in the successor list, based on the calculation
        // in findUnwindDestinations() in SelectionDAGBuilder.cpp.
        if (Succ->isEHPad()) {
          UnwindDest = Succ;
          break;
        }
      }
      if (EHPadStack.back() == UnwindDest)
        continue;

      // Include EH_LABELs in the range before and after the invoke
      MachineInstr *RangeBegin = &MI, *RangeEnd = &MI;
      if (RangeBegin->getIterator() != MBB.begin() &&
          std::prev(RangeBegin->getIterator())->isEHLabel())
        RangeBegin = &*std::prev(RangeBegin->getIterator());
      if (std::next(RangeEnd->getIterator()) != MBB.end() &&
          std::next(RangeEnd->getIterator())->isEHLabel())
        RangeEnd = &*std::next(RangeEnd->getIterator());

      // If not, record the range.
      UnwindDestToTryRanges[UnwindDest].push_back(
          TryRange(RangeBegin, RangeEnd));
      LLVM_DEBUG(dbgs() << "- Call unwind mismatch: MBB = " << MBB.getName()
                        << "\nCall = " << MI
                        << "\nOriginal dest = " << UnwindDest->getName()
                        << "  Current dest = " << EHPadStack.back()->getName()
                        << "\n\n");
    }
  }

  assert(EHPadStack.empty());

  // Gather possibly throwing calls that are supposed to unwind up to the caller
  // if they throw, but currently unwind to an incorrect destination. Unlike the
  // loop above, there can be multiple calls within a BB that unwind to the
  // caller, which we should group together in a range. (Case 2)

  MachineInstr *RangeBegin = nullptr, *RangeEnd = nullptr; // inclusive

  // Record the range.
  auto RecordCallerMismatchRange = [&](const MachineBasicBlock *CurrentDest) {
    UnwindDestToTryRanges[getFakeCallerBlock(MF)].push_back(
        TryRange(RangeBegin, RangeEnd));
    LLVM_DEBUG(dbgs() << "- Call unwind mismatch: MBB = "
                      << RangeBegin->getParent()->getName()
                      << "\nRange begin = " << *RangeBegin
                      << "Range end = " << *RangeEnd
                      << "\nOriginal dest = caller  Current dest = "
                      << CurrentDest->getName() << "\n\n");
    RangeBegin = RangeEnd = nullptr; // Reset range pointers
  };

  for (auto &MBB : reverse(MF)) {
    bool SeenThrowableInstInBB = false;
    for (auto &MI : reverse(MBB)) {
      bool MayThrow = WebAssembly::mayThrow(MI);

      // If MBB has an EH pad successor and this is the last instruction that
      // may throw, this instruction unwinds to the EH pad and not to the
      // caller.
      if (MBB.hasEHPadSuccessor() && MayThrow && !SeenThrowableInstInBB)
        SeenThrowableInstInBB = true;

      // We wrap up the current range when we see a marker even if we haven't
      // finished a BB.
      else if (RangeEnd && WebAssembly::isMarker(MI.getOpcode()))
        RecordCallerMismatchRange(EHPadStack.back());

      // If EHPadStack is empty, that means it correctly unwinds to the caller
      // if it throws, so we're good. If MI does not throw, we're good too.
      else if (EHPadStack.empty() || !MayThrow) {
      }

      // We found an instruction that unwinds to the caller but currently has an
      // incorrect unwind destination. Create a new range or increment the
      // currently existing range.
      else {
        if (!RangeEnd)
          RangeBegin = RangeEnd = &MI;
        else
          RangeBegin = &MI;
      }

      // Update EHPadStack.
      if (WebAssembly::isTry(MI.getOpcode()))
        EHPadStack.pop_back();
      else if (WebAssembly::isCatch(MI.getOpcode()))
        EHPadStack.push_back(MI.getParent());
    }

    if (RangeEnd)
      RecordCallerMismatchRange(EHPadStack.back());
  }

  assert(EHPadStack.empty());

  // We don't have any unwind destination mismatches to resolve.
  if (UnwindDestToTryRanges.empty())
    return false;

  // When end_loop is before end_try_table within the same BB in unwind
  // destinations, we should split the end_loop into another BB.
  if (!WebAssembly::WasmUseLegacyEH)
    for (auto &[UnwindDest, _] : UnwindDestToTryRanges) {
      auto It = EHPadToTry.find(UnwindDest);
      // If UnwindDest is the fake caller block, it will not be in EHPadToTry
      // map
      if (It != EHPadToTry.end()) {
        auto *TryTable = It->second;
        auto *EndTryTable = BeginToEnd[TryTable];
        splitEndLoopBB(EndTryTable->getParent());
      }
    }

  // Now we fix the mismatches by wrapping calls with inner try-delegates.
  for (auto &P : UnwindDestToTryRanges) {
    NumCallUnwindMismatches += P.second.size();
    MachineBasicBlock *UnwindDest = P.first;
    auto &TryRanges = P.second;

    for (auto Range : TryRanges) {
      MachineInstr *RangeBegin = nullptr, *RangeEnd = nullptr;
      std::tie(RangeBegin, RangeEnd) = Range;
      auto *MBB = RangeBegin->getParent();

      // If this BB has an EH pad successor, i.e., ends with an 'invoke', and if
      // the current range contains the invoke, now we are going to wrap the
      // invoke with try-delegate or try_table-end_try_table, making the
      // 'delegate' or 'end_try_table' BB the new successor instead, so remove
      // the EH pad succesor here. The BB may not have an EH pad successor if
      // calls in this BB throw to the caller.
      if (UnwindDest != getFakeCallerBlock(MF)) {
        MachineBasicBlock *EHPad = nullptr;
        for (auto *Succ : MBB->successors()) {
          if (Succ->isEHPad()) {
            EHPad = Succ;
            break;
          }
        }
        if (EHPad)
          MBB->removeSuccessor(EHPad);
      }

      if (WebAssembly::WasmUseLegacyEH)
        addNestedTryDelegate(RangeBegin, RangeEnd, UnwindDest);
      else
        addNestedTryTable(RangeBegin, RangeEnd, UnwindDest);
    }
  }

  return true;
}

// Returns the single destination of try_table, if there is one. All try_table
// we generate in this pass has a single destination, i.e., a single catch
// clause.
static MachineBasicBlock *getSingleUnwindDest(const MachineInstr *TryTable) {
  if (TryTable->getOperand(1).getImm() != 1)
    return nullptr;
  switch (TryTable->getOperand(2).getImm()) {
  case wasm::WASM_OPCODE_CATCH:
  case wasm::WASM_OPCODE_CATCH_REF:
    return TryTable->getOperand(4).getMBB();
  case wasm::WASM_OPCODE_CATCH_ALL:
  case wasm::WASM_OPCODE_CATCH_ALL_REF:
    return TryTable->getOperand(3).getMBB();
  default:
    llvm_unreachable("try_table: Invalid catch clause\n");
  }
}

bool WebAssemblyCFGStackify::fixCatchUnwindMismatches(MachineFunction &MF) {
  // This function is used for both the legacy EH and the standard (exnref) EH,
  // and the reason we have unwind mismatches is the same for the both of them,
  // but the code examples in the comments are going to be different. To make
  // the description less confusing, we write the basically same comments twice,
  // once for the legacy EH and the standard EH.
  //
  // -- Legacy EH --------------------------------------------------------------
  //
  // There is another kind of unwind destination mismatches besides call unwind
  // mismatches, which we will call "catch unwind mismatches". See this example
  // after the marker placement:
  // try
  //   try
  //     call @foo
  //   catch __cpp_exception  ;; ehpad A (next unwind dest: caller)
  //     ...
  //   end_try
  // catch_all                ;; ehpad B
  //   ...
  // end_try
  //
  // 'call @foo's unwind destination is the ehpad A. But suppose 'call @foo'
  // throws a foreign exception that is not caught by ehpad A, and its next
  // destination should be the caller. But after control flow linearization,
  // another EH pad can be placed in between (e.g. ehpad B here), making the
  // next unwind destination incorrect. In this case, the foreign exception will
  // instead go to ehpad B and will be caught there instead. In this example the
  // correct next unwind destination is the caller, but it can be another outer
  // catch in other cases.
  //
  // There is no specific 'call' or 'throw' instruction to wrap with a
  // try-delegate, so we wrap the whole try-catch-end with a try-delegate and
  // make it rethrow to the right destination, which is the caller in the
  // example below:
  // try
  //   try                     ;; (new)
  //     try
  //       call @foo
  //     catch __cpp_exception ;; ehpad A (next unwind dest: caller)
  //       ...
  //     end_try
  //   delegate 1 (caller)     ;; (new)
  // catch_all                 ;; ehpad B
  //   ...
  // end_try
  //
  // The right destination may be another EH pad or the caller. (The example
  // here shows the case it is the caller.)
  //
  // -- Standard EH ------------------------------------------------------------
  //
  // There is another kind of unwind destination mismatches besides call unwind
  // mismatches, which we will call "catch unwind mismatches". See this example
  // after the marker placement:
  // block
  //   try_table (catch_all_ref 0)
  //     block
  //       try_table (catch ... 0)
  //         call @foo
  //       end_try_table
  //     end_block                  ;; ehpad A (next unwind dest: caller)
  //     ...
  //   end_try_table
  // end_block                      ;; ehpad B
  // ...
  //
  // 'call @foo's unwind destination is the ehpad A. But suppose 'call @foo'
  // throws a foreign exception that is not caught by ehpad A, and its next
  // destination should be the caller. But after control flow linearization,
  // another EH pad can be placed in between (e.g. ehpad B here), making the
  // next unwind destination incorrect. In this case, the foreign exception will
  // instead go to ehpad B and will be caught there instead. In this example the
  // correct next unwind destination is the caller, but it can be another outer
  // catch in other cases.
  //
  // There is no specific 'call' or 'throw' instruction to wrap with an inner
  // try_table-end_try_table, so we wrap the whole try_table-end_try_table with
  // an inner try_table-end_try_table that sends the exception to a trampoline
  // BB. We rethrow the sent exception using a throw_ref to the right
  // destination, which is the caller in the example below:
  // block exnref
  //   block
  //     try_table (catch_all_ref 0)
  //       try_table (catch_all_ref 2)  ;; (new) to trampoline
  //         block
  //           try_table (catch ... 0)
  //             call @foo
  //           end_try_table
  //         end_block                  ;; ehpad A (next unwind dest: caller)
  //       end_try_table                ;; (new)
  //       ...
  //     end_try_table
  //   end_block                        ;; ehpad B
  //   ...
  // end_block                          ;; (new) caller trampoline BB
  // throw_ref                          ;; (new) throw to the caller
  //
  // The right destination may be another EH pad or the caller. (The example
  // here shows the case it is the caller.)

  const auto *EHInfo = MF.getWasmEHFuncInfo();
  assert(EHInfo);
  SmallVector<const MachineBasicBlock *, 8> EHPadStack;
  // For EH pads that have catch unwind mismatches, a map of <EH pad, its
  // correct unwind destination>.
  DenseMap<MachineBasicBlock *, MachineBasicBlock *> EHPadToUnwindDest;

  for (auto &MBB : reverse(MF)) {
    for (auto &MI : reverse(MBB)) {
      if (MI.getOpcode() == WebAssembly::TRY)
        EHPadStack.pop_back();
      else if (MI.getOpcode() == WebAssembly::TRY_TABLE) {
        // We want to exclude try_tables created in fixCallUnwindMismatches.
        // Check if the try_table's unwind destination matches the EH pad stack
        // top. If it is created in fixCallUnwindMismatches, it wouldn't.
        if (getSingleUnwindDest(&MI) == EHPadStack.back())
          EHPadStack.pop_back();
      } else if (MI.getOpcode() == WebAssembly::DELEGATE)
        EHPadStack.push_back(&MBB);
      else if (WebAssembly::isCatch(MI.getOpcode())) {
        auto *EHPad = &MBB;

        // If the BB has a catch pseudo instruction but is not marked as an EH
        // pad, it's a trampoline BB we created in fixCallUnwindMismatches. Skip
        // it.
        if (!EHPad->isEHPad())
          continue;

        // catch_all always catches an exception, so we don't need to do
        // anything
        if (WebAssembly::isCatchAll(MI.getOpcode())) {
        }

        // This can happen when the unwind dest was removed during the
        // optimization, e.g. because it was unreachable.
        else if (EHPadStack.empty() && EHInfo->hasUnwindDest(EHPad)) {
          LLVM_DEBUG(dbgs() << "EHPad (" << EHPad->getName()
                            << "'s unwind destination does not exist anymore"
                            << "\n\n");
        }

        // The EHPad's next unwind destination is the caller, but we incorrectly
        // unwind to another EH pad.
        else if (!EHPadStack.empty() && !EHInfo->hasUnwindDest(EHPad)) {
          EHPadToUnwindDest[EHPad] = getFakeCallerBlock(MF);
          LLVM_DEBUG(dbgs()
                     << "- Catch unwind mismatch:\nEHPad = " << EHPad->getName()
                     << "  Original dest = caller  Current dest = "
                     << EHPadStack.back()->getName() << "\n\n");
        }

        // The EHPad's next unwind destination is an EH pad, whereas we
        // incorrectly unwind to another EH pad.
        else if (!EHPadStack.empty() && EHInfo->hasUnwindDest(EHPad)) {
          auto *UnwindDest = EHInfo->getUnwindDest(EHPad);
          if (EHPadStack.back() != UnwindDest) {
            EHPadToUnwindDest[EHPad] = UnwindDest;
            LLVM_DEBUG(dbgs() << "- Catch unwind mismatch:\nEHPad = "
                              << EHPad->getName() << "  Original dest = "
                              << UnwindDest->getName() << "  Current dest = "
                              << EHPadStack.back()->getName() << "\n\n");
          }
        }

        EHPadStack.push_back(EHPad);
      }
    }
  }

  assert(EHPadStack.empty());
  if (EHPadToUnwindDest.empty())
    return false;

  // When end_loop is before end_try_table within the same BB in unwind
  // destinations, we should split the end_loop into another BB.
  for (auto &[_, UnwindDest] : EHPadToUnwindDest) {
    auto It = EHPadToTry.find(UnwindDest);
    // If UnwindDest is the fake caller block, it will not be in EHPadToTry map
    if (It != EHPadToTry.end()) {
      auto *TryTable = It->second;
      auto *EndTryTable = BeginToEnd[TryTable];
      splitEndLoopBB(EndTryTable->getParent());
    }
  }

  NumCatchUnwindMismatches += EHPadToUnwindDest.size();
  SmallPtrSet<MachineBasicBlock *, 4> NewEndTryBBs;

  for (auto &[EHPad, UnwindDest] : EHPadToUnwindDest) {
    MachineInstr *Try = EHPadToTry[EHPad];
    MachineInstr *EndTry = BeginToEnd[Try];
    if (WebAssembly::WasmUseLegacyEH) {
      addNestedTryDelegate(Try, EndTry, UnwindDest);
      NewEndTryBBs.insert(EndTry->getParent());
    } else {
      addNestedTryTable(Try, EndTry, UnwindDest);
    }
  }

  if (!WebAssembly::WasmUseLegacyEH)
    return true;

  // Adding a try-delegate wrapping an existing try-catch-end can make existing
  // branch destination BBs invalid. For example,
  //
  // - Before:
  // bb0:
  //   block
  //     br bb3
  // bb1:
  //     try
  //       ...
  // bb2: (ehpad)
  //     catch
  // bb3:
  //     end_try
  //   end_block   ;; 'br bb3' targets here
  //
  // Suppose this try-catch-end has a catch unwind mismatch, so we need to wrap
  // this with a try-delegate. Then this becomes:
  //
  // - After:
  // bb0:
  //   block
  //     br bb3    ;; invalid destination!
  // bb1:
  //     try       ;; (new instruction)
  //       try
  //         ...
  // bb2: (ehpad)
  //       catch
  // bb3:
  //       end_try ;; 'br bb3' still incorrectly targets here!
  // delegate_bb:  ;; (new BB)
  //     delegate  ;; (new instruction)
  // split_bb:     ;; (new BB)
  //   end_block
  //
  // Now 'br bb3' incorrectly branches to an inner scope.
  //
  // As we can see in this case, when branches target a BB that has both
  // 'end_try' and 'end_block' and the BB is split to insert a 'delegate', we
  // have to remap existing branch destinations so that they target not the
  // 'end_try' BB but the new 'end_block' BB. There can be multiple 'delegate's
  // in between, so we try to find the next BB with 'end_block' instruction. In
  // this example, the 'br bb3' instruction should be remapped to 'br split_bb'.
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      if (MI.isTerminator()) {
        for (auto &MO : MI.operands()) {
          if (MO.isMBB() && NewEndTryBBs.count(MO.getMBB())) {
            auto *BrDest = MO.getMBB();
            bool FoundEndBlock = false;
            for (; std::next(BrDest->getIterator()) != MF.end();
                 BrDest = BrDest->getNextNode()) {
              for (const auto &MI : *BrDest) {
                if (MI.getOpcode() == WebAssembly::END_BLOCK) {
                  FoundEndBlock = true;
                  break;
                }
              }
              if (FoundEndBlock)
                break;
            }
            assert(FoundEndBlock);
            MO.setMBB(BrDest);
          }
        }
      }
    }
  }

  return true;
}

void WebAssemblyCFGStackify::recalculateScopeTops(MachineFunction &MF) {
  // Renumber BBs and recalculate ScopeTop info because new BBs might have been
  // created and inserted during fixing unwind mismatches.
  MF.RenumberBlocks();
  MDT->updateBlockNumbers();
  ScopeTops.clear();
  ScopeTops.resize(MF.getNumBlockIDs());
  for (auto &MBB : reverse(MF)) {
    for (auto &MI : reverse(MBB)) {
      if (ScopeTops[MBB.getNumber()])
        break;
      switch (MI.getOpcode()) {
      case WebAssembly::END_BLOCK:
      case WebAssembly::END_LOOP:
      case WebAssembly::END_TRY:
      case WebAssembly::END_TRY_TABLE:
      case WebAssembly::DELEGATE:
        updateScopeTops(EndToBegin[&MI]->getParent(), &MBB);
        break;
      case WebAssembly::CATCH_LEGACY:
      case WebAssembly::CATCH_ALL_LEGACY:
        updateScopeTops(EHPadToTry[&MBB]->getParent(), &MBB);
        break;
      }
    }
  }
}

/// In normal assembly languages, when the end of a function is unreachable,
/// because the function ends in an infinite loop or a noreturn call or similar,
/// it isn't necessary to worry about the function return type at the end of
/// the function, because it's never reached. However, in WebAssembly, blocks
/// that end at the function end need to have a return type signature that
/// matches the function signature, even though it's unreachable. This function
/// checks for such cases and fixes up the signatures.
void WebAssemblyCFGStackify::fixEndsAtEndOfFunction(MachineFunction &MF) {
  const auto &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();

  if (MFI.getResults().empty())
    return;

  // MCInstLower will add the proper types to multivalue signatures based on the
  // function return type
  WebAssembly::BlockType RetType =
      MFI.getResults().size() > 1
          ? WebAssembly::BlockType::Multivalue
          : WebAssembly::BlockType(
                WebAssembly::toValType(MFI.getResults().front()));

  SmallVector<MachineBasicBlock::reverse_iterator, 4> Worklist;
  Worklist.push_back(MF.rbegin()->rbegin());

  auto Process = [&](MachineBasicBlock::reverse_iterator It) {
    auto *MBB = It->getParent();
    while (It != MBB->rend()) {
      MachineInstr &MI = *It++;
      if (MI.isPosition() || MI.isDebugInstr())
        continue;
      switch (MI.getOpcode()) {
      case WebAssembly::END_TRY: {
        // If a 'try''s return type is fixed, both its try body and catch body
        // should satisfy the return type, so we need to search 'end'
        // instructions before its corresponding 'catch' too.
        auto *EHPad = TryToEHPad.lookup(EndToBegin[&MI]);
        assert(EHPad);
        auto NextIt =
            std::next(WebAssembly::findCatch(EHPad)->getReverseIterator());
        if (NextIt != EHPad->rend())
          Worklist.push_back(NextIt);
        [[fallthrough]];
      }
      case WebAssembly::END_BLOCK:
      case WebAssembly::END_LOOP:
      case WebAssembly::END_TRY_TABLE:
      case WebAssembly::DELEGATE:
        EndToBegin[&MI]->getOperand(0).setImm(int32_t(RetType));
        continue;
      default:
        // Something other than an `end`. We're done for this BB.
        return;
      }
    }
    // We've reached the beginning of a BB. Continue the search in the previous
    // BB.
    Worklist.push_back(MBB->getPrevNode()->rbegin());
  };

  while (!Worklist.empty())
    Process(Worklist.pop_back_val());
}

// WebAssembly functions end with an end instruction, as if the function body
// were a block.
static void appendEndToFunction(MachineFunction &MF,
                                const WebAssemblyInstrInfo &TII) {
  BuildMI(MF.back(), MF.back().end(),
          MF.back().findPrevDebugLoc(MF.back().end()),
          TII.get(WebAssembly::END_FUNCTION));
}

// We added block~end_block and try_table~end_try_table markers in
// placeTryTableMarker. But When catch clause's destination has a return type,
// as in the case of catch with a concrete tag, catch_ref, and catch_all_ref.
// For example:
// block exnref
//   try_table (catch_all_ref 0)
//     ...
//   end_try_table
// end_block
// ... use exnref ...
//
// This code is not valid because the block's body type is not exnref. So we add
// an unreachable after the 'end_try_table' to make the code valid here:
// block exnref
//   try_table (catch_all_ref 0)
//     ...
//   end_try_table
//   unreachable      (new)
// end_block
//
// Because 'unreachable' is a terminator we also need to split the BB.
static void addUnreachableAfterTryTables(MachineFunction &MF,
                                         const WebAssemblyInstrInfo &TII) {
  std::vector<MachineInstr *> EndTryTables;
  for (auto &MBB : MF)
    for (auto &MI : MBB)
      if (MI.getOpcode() == WebAssembly::END_TRY_TABLE)
        EndTryTables.push_back(&MI);

  for (auto *EndTryTable : EndTryTables) {
    auto *MBB = EndTryTable->getParent();
    auto *NewEndTryTableBB = MF.CreateMachineBasicBlock();
    MF.insert(MBB->getIterator(), NewEndTryTableBB);
    auto SplitPos = std::next(EndTryTable->getIterator());
    NewEndTryTableBB->splice(NewEndTryTableBB->end(), MBB, MBB->begin(),
                             SplitPos);
    NewEndTryTableBB->addSuccessor(MBB);
    BuildMI(NewEndTryTableBB, EndTryTable->getDebugLoc(),
            TII.get(WebAssembly::UNREACHABLE));
  }
}

/// Insert BLOCK/LOOP/TRY/TRY_TABLE markers at appropriate places.
void WebAssemblyCFGStackify::placeMarkers(MachineFunction &MF) {
  // We allocate one more than the number of blocks in the function to
  // accommodate for the possible fake block we may insert at the end.
  ScopeTops.resize(MF.getNumBlockIDs() + 1);
  // Place the LOOP for MBB if MBB is the header of a loop.
  for (auto &MBB : MF)
    placeLoopMarker(MBB);

  const MCAsmInfo *MCAI = MF.getTarget().getMCAsmInfo();
  for (auto &MBB : MF) {
    if (MBB.isEHPad()) {
      // Place the TRY/TRY_TABLE for MBB if MBB is the EH pad of an exception.
      if (MCAI->getExceptionHandlingType() == ExceptionHandling::Wasm &&
          MF.getFunction().hasPersonalityFn()) {
        if (WebAssembly::WasmUseLegacyEH)
          placeTryMarker(MBB);
        else
          placeTryTableMarker(MBB);
      }
    } else {
      // Place the BLOCK for MBB if MBB is branched to from above.
      placeBlockMarker(MBB);
    }
  }

  if (MCAI->getExceptionHandlingType() == ExceptionHandling::Wasm &&
      MF.getFunction().hasPersonalityFn()) {
    const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
    // Add an 'unreachable' after 'end_try_table's.
    addUnreachableAfterTryTables(MF, TII);
    // Fix mismatches in unwind destinations induced by linearizing the code.
    fixCallUnwindMismatches(MF);
    fixCatchUnwindMismatches(MF);
    // addUnreachableAfterTryTables and fixUnwindMismatches create new BBs, so
    // we need to recalculate ScopeTops.
    recalculateScopeTops(MF);
  }
}

unsigned WebAssemblyCFGStackify::getBranchDepth(
    const SmallVectorImpl<EndMarkerInfo> &Stack, const MachineBasicBlock *MBB) {
  unsigned Depth = 0;
  for (auto X : reverse(Stack)) {
    if (X.first == MBB)
      break;
    ++Depth;
  }
  assert(Depth < Stack.size() && "Branch destination should be in scope");
  return Depth;
}

unsigned WebAssemblyCFGStackify::getDelegateDepth(
    const SmallVectorImpl<EndMarkerInfo> &Stack, const MachineBasicBlock *MBB) {
  if (MBB == FakeCallerBB)
    return Stack.size();
  // Delegate's destination is either a catch or a another delegate BB. When the
  // destination is another delegate, we can compute the argument in the same
  // way as branches, because the target delegate BB only contains the single
  // delegate instruction.
  if (!MBB->isEHPad()) // Target is a delegate BB
    return getBranchDepth(Stack, MBB);

  // When the delegate's destination is a catch BB, we need to use its
  // corresponding try's end_try BB because Stack contains each marker's end BB.
  // Also we need to check if the end marker instruction matches, because a
  // single BB can contain multiple end markers, like this:
  // bb:
  //   END_BLOCK
  //   END_TRY
  //   END_BLOCK
  //   END_TRY
  //   ...
  //
  // In case of branches getting the immediate that targets any of these is
  // fine, but delegate has to exactly target the correct try.
  unsigned Depth = 0;
  const MachineInstr *EndTry = BeginToEnd[EHPadToTry[MBB]];
  for (auto X : reverse(Stack)) {
    if (X.first == EndTry->getParent() && X.second == EndTry)
      break;
    ++Depth;
  }
  assert(Depth < Stack.size() && "Delegate destination should be in scope");
  return Depth;
}

unsigned WebAssemblyCFGStackify::getRethrowDepth(
    const SmallVectorImpl<EndMarkerInfo> &Stack,
    const MachineBasicBlock *EHPadToRethrow) {
  unsigned Depth = 0;
  for (auto X : reverse(Stack)) {
    const MachineInstr *End = X.second;
    if (End->getOpcode() == WebAssembly::END_TRY) {
      auto *EHPad = TryToEHPad[EndToBegin[End]];
      if (EHPadToRethrow == EHPad)
        break;
    }
    ++Depth;
  }
  assert(Depth < Stack.size() && "Rethrow destination should be in scope");
  return Depth;
}

void WebAssemblyCFGStackify::rewriteDepthImmediates(MachineFunction &MF) {
  // Now rewrite references to basic blocks to be depth immediates.
  SmallVector<EndMarkerInfo, 8> Stack;

  auto RewriteOperands = [&](MachineInstr &MI) {
    // Rewrite MBB operands to be depth immediates.
    SmallVector<MachineOperand, 4> Ops(MI.operands());
    while (MI.getNumOperands() > 0)
      MI.removeOperand(MI.getNumOperands() - 1);
    for (auto MO : Ops) {
      if (MO.isMBB()) {
        if (MI.getOpcode() == WebAssembly::DELEGATE)
          MO = MachineOperand::CreateImm(getDelegateDepth(Stack, MO.getMBB()));
        else if (MI.getOpcode() == WebAssembly::RETHROW)
          MO = MachineOperand::CreateImm(getRethrowDepth(Stack, MO.getMBB()));
        else
          MO = MachineOperand::CreateImm(getBranchDepth(Stack, MO.getMBB()));
      }
      MI.addOperand(MF, MO);
    }
  };

  for (auto &MBB : reverse(MF)) {
    for (MachineInstr &MI : llvm::reverse(MBB)) {
      switch (MI.getOpcode()) {
      case WebAssembly::BLOCK:
      case WebAssembly::TRY:
        assert(ScopeTops[Stack.back().first->getNumber()]->getNumber() <=
                   MBB.getNumber() &&
               "Block/try/try_table marker should be balanced");
        Stack.pop_back();
        break;

      case WebAssembly::TRY_TABLE:
        assert(ScopeTops[Stack.back().first->getNumber()]->getNumber() <=
                   MBB.getNumber() &&
               "Block/try/try_table marker should be balanced");
        Stack.pop_back();
        RewriteOperands(MI);
        break;

      case WebAssembly::LOOP:
        assert(Stack.back().first == &MBB && "Loop top should be balanced");
        Stack.pop_back();
        break;

      case WebAssembly::END_BLOCK:
      case WebAssembly::END_TRY:
      case WebAssembly::END_TRY_TABLE:
        Stack.push_back(std::make_pair(&MBB, &MI));
        break;

      case WebAssembly::END_LOOP:
        Stack.push_back(std::make_pair(EndToBegin[&MI]->getParent(), &MI));
        break;

      case WebAssembly::DELEGATE:
        RewriteOperands(MI);
        Stack.push_back(std::make_pair(&MBB, &MI));
        break;

      default:
        if (MI.isTerminator())
          RewriteOperands(MI);
        break;
      }
    }
  }
  assert(Stack.empty() && "Control flow should be balanced");
}

void WebAssemblyCFGStackify::cleanupFunctionData(MachineFunction &MF) {
  if (FakeCallerBB)
    MF.deleteMachineBasicBlock(FakeCallerBB);
  AppendixBB = FakeCallerBB = CallerTrampolineBB = nullptr;
}

void WebAssemblyCFGStackify::releaseMemory() {
  ScopeTops.clear();
  BeginToEnd.clear();
  EndToBegin.clear();
  TryToEHPad.clear();
  EHPadToTry.clear();
  UnwindDestToTrampoline.clear();
}

bool WebAssemblyCFGStackify::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** CFG Stackifying **********\n"
                       "********** Function: "
                    << MF.getName() << '\n');
  const MCAsmInfo *MCAI = MF.getTarget().getMCAsmInfo();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();

  releaseMemory();

  // Liveness is not tracked for VALUE_STACK physreg.
  MF.getRegInfo().invalidateLiveness();

  // Place the BLOCK/LOOP/TRY/TRY_TABLE markers to indicate the beginnings of
  // scopes.
  placeMarkers(MF);

  // Remove unnecessary instructions possibly introduced by try/end_trys.
  if (MCAI->getExceptionHandlingType() == ExceptionHandling::Wasm &&
      MF.getFunction().hasPersonalityFn() && WebAssembly::WasmUseLegacyEH)
    removeUnnecessaryInstrs(MF);

  // Convert MBB operands in terminators to relative depth immediates.
  rewriteDepthImmediates(MF);

  // Fix up block/loop/try/try_table signatures at the end of the function to
  // conform to WebAssembly's rules.
  fixEndsAtEndOfFunction(MF);

  // Add an end instruction at the end of the function body.
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  appendEndToFunction(MF, TII);

  cleanupFunctionData(MF);

  MF.getInfo<WebAssemblyFunctionInfo>()->setCFGStackified();
  return true;
}
