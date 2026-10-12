//===-- FIRConvertToArith.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace fir {
#define GEN_PASS_DEF_FIRCONVERTTOARITH
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {
/// A type this pass is willing to rewrite: signless integers wider than i1, and
/// index. i1 and unsigned integers are excluded because fir.convert
/// zero-extends them, which arith.extsi would get wrong.
static bool isRewritableType(mlir::Type ty) {
  if (mlir::isa<mlir::IndexType>(ty))
    return true;
  auto intTy = mlir::dyn_cast<mlir::IntegerType>(ty);
  return intTy && intTy.isSignless() && intTy.getWidth() > 1;
}

struct ConvertToArithCast : mlir::OpRewritePattern<fir::ConvertOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(fir::ConvertOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value in = op.getValue();
    mlir::Type fromTy = in.getType();
    mlir::Type toTy = op.getType();
    if (!isRewritableType(fromTy) || !isRewritableType(toTy))
      return mlir::failure();

    if (fromTy == toTy) {
      rewriter.replaceOp(op, in);
      return mlir::success();
    }

    if (mlir::isa<mlir::IndexType>(fromTy) ||
        mlir::isa<mlir::IndexType>(toTy)) {
      rewriter.replaceOpWithNewOp<mlir::arith::IndexCastOp>(op, toTy, in);
      return mlir::success();
    }

    unsigned fromBits = mlir::cast<mlir::IntegerType>(fromTy).getWidth();
    unsigned toBits = mlir::cast<mlir::IntegerType>(toTy).getWidth();
    if (fromBits > toBits)
      rewriter.replaceOpWithNewOp<mlir::arith::TruncIOp>(op, toTy, in);
    else
      rewriter.replaceOpWithNewOp<mlir::arith::ExtSIOp>(op, toTy, in);
    return mlir::success();
  }
};

class FIRConvertToArithPass
    : public fir::impl::FIRConvertToArithBase<FIRConvertToArithPass> {
public:
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<ConvertToArithCast>(&getContext());
    mlir::walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};
} // namespace
