// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <iostream>

using namespace mlir;

namespace mlir {
namespace iree_compiler {

namespace {

//===----------------------------------------------------------------------===//
// Op Conversion Patterns
//===----------------------------------------------------------------------===//

/// Converts vector transfer ops to SPIR-V joint matrix load/store ops.
struct ConvertVectorTransferOp final
    : public OpInterfaceConversionPattern<VectorTransferOpInterface> {
  using OpInterfaceConversionPattern::OpInterfaceConversionPattern;

  LogicalResult matchAndRewrite(
      VectorTransferOpInterface op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Don't support masked load/store.
    std::cout<<"here A\n";
    if (op.getMaskType()) return failure();

    // Expect inbound access.
    if (op.in_bounds()) {
      auto inBounds = op.in_bounds()->getAsValueRange<BoolAttr>();
      if (!llvm::all_of(inBounds, [](bool v) { return v; })) return failure();
    }

    // Expect transfers over memrefs.
    auto memrefType = op.getShapedType().dyn_cast<MemRefType>();
    if (!memrefType) return failure();

    // Expect 2-D vectors.
    std::cout<<"here B\n";
    auto vectorType = op.getVectorType();
    if (vectorType.getRank() != 2) {
    std::cout<<"Failing with\n";
    op.dump();  
      return failure();
    }
   std::cout<<"here C\n";
    // TODO: Use coloumn major with transposed transfer ops.
    if (!op.permutation_map().isMinorIdentity()) return failure();

    int64_t offset = 0;
    SmallVector<int64_t, 2> strides;
    if (failed(getStridesAndOffset(memrefType, strides, offset))){
      std::cout<<"failing due to strides\n";
      return failure();
    }
    auto stride = strides[0];
    if (ShapedType::isDynamicStrideOrOffset(stride)) return failure();
   std::cout<<"here 1\n";
    auto loc = op.getLoc();

    auto i32Type = rewriter.getI32Type();
    auto strideValue = rewriter.create<spirv::ConstantOp>(
        loc, i32Type, IntegerAttr::get(i32Type, stride));
    /*auto coloumnMajor = rewriter.create<spirv::ConstantOp>(
        loc, rewriter.getI1Type(), rewriter.getBoolAttr(false));*/
    //Type matType = typeConverter->convertType(vectorType);
    //std::cout<<"type convertor\n";
    //matType.dump();
    Type matType1 = spirv::JointMatrixINTELType::get(
                vectorType.getElementType(), spirv::Scope::Subgroup, vectorType.getDimSize(0),
                vectorType.getDimSize(1), spirv::MatrixLayout::ColumnMajor);
    //std::cout<<"manual convertor\n";
   // matType1.dump();
    /*Type matTypePackedB = spirv::JointMatrixINTELType::get(
                vectorType, spirv::Scope::Subgroup, vectorType.getDimSize(0),
                vectorType.getDimSize(1), spirv::MatrixLayout::PackedB);*/
    std::cout<<"here 2\n";
    if (auto readOp = dyn_cast<vector::TransferReadOp>(*op)) {
      vector::TransferReadOp::Adaptor adaptor(operands,
                                              op->getAttrDictionary());
      Value bufferPtr = spirv::getElementPtrDirect(
          *getTypeConverter<SPIRVTypeConverter>(), memrefType,
          adaptor.getSource(), adaptor.getIndices(), loc, rewriter);
  for (Operation *user : op->getUsers()) {
    std::cout<<"checking uses\n";
    auto contract = dyn_cast<vector::ContractionOp>(user);
    if (!contract)
      return failure();
    if (contract.getLhs() == op->getResult(0)) {
      std::cout<<"Use 1\n";
      rewriter.replaceOpWithNewOp<spirv::JointMatrixLoadINTELOp>(
          op, matType1, bufferPtr, strideValue,
          spirv::MatrixLayout::ColumnMajor, spirv::Scope::Subgroup,
          spirv::MemoryAccessAttr(),IntegerAttr());
      return success();
    }
    if (contract.getRhs() == op->getResult(0)) {
      std::cout<<"Use 2\n";
      rewriter.replaceOpWithNewOp<spirv::JointMatrixLoadINTELOp>(
          op, matType1, bufferPtr, strideValue,
          spirv::MatrixLayout::PackedB, spirv::Scope::Subgroup,
          spirv::MemoryAccessAttr(),IntegerAttr());
      return success();
    }
      if (contract.getAcc() == op->getResult(0)) {
      std::cout<<"Use 3\n";
      rewriter.replaceOpWithNewOp<spirv::JointMatrixLoadINTELOp>(
          op, matType1, bufferPtr, strideValue,
          spirv::MatrixLayout::ColumnMajor, spirv::Scope::Subgroup,
          spirv::MemoryAccessAttr(),IntegerAttr());
      return success();
    }
  }
}

    if (auto writeOp = dyn_cast<vector::TransferWriteOp>(*op)) {
      vector::TransferWriteOp::Adaptor adaptor(operands,
                                               op->getAttrDictionary());
      Value bufferPtr = spirv::getElementPtrDirect(
          *getTypeConverter<SPIRVTypeConverter>(), memrefType,
          adaptor.getSource(), adaptor.getIndices(), loc, rewriter);
      rewriter.create<spirv::JointMatrixStoreINTELOp>(
          loc, bufferPtr, adaptor.getVector(), strideValue,
          spirv::MatrixLayout::ColumnMajor, spirv::Scope::Subgroup, 
          spirv::MemoryAccessAttr(),IntegerAttr());
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

/// Converts vector.contract ops to SPIR-V joint matrix multiple-add ops.
struct ConvertVectorContractOp final
    : public OpConversionPattern<vector::ContractionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      vector::ContractionOp contractOp, OpAdaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    if (!llvm::empty(contractOp.getMasks())) return failure();

    // Check that this is a matmul operation.
    auto iterators = contractOp.getIteratorTypes().getValue();
    if (iterators.size() != 3 || !isParallelIterator(iterators[0]) ||
        !isParallelIterator(iterators[1]) ||
        !isReductionIterator(iterators[2])) {
      return failure();
    }
    if (contractOp.getKind() != vector::CombiningKind::ADD) return failure();

    // Column major matmuls should have been lowered to transpose + contract
    // by this point. Transpose can be handled by load/store operations.
    if (!isRowMajorMatmul(contractOp.getIndexingMapsAttr())) return failure();

    rewriter.replaceOpWithNewOp<spirv::JointMatrixMadINTELOp>(
        contractOp, operands.getAcc().getType(), operands.getLhs(),
        operands.getRhs(), operands.getAcc(), spirv::Scope::Subgroup);
    return success();
  }
};

/// Converts splat vector constants to constant SPIR-V joint matrix ops.
struct ConvertConstantMatrix final
    : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      arith::ConstantOp op, OpAdaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    // Only convert 2-D vector constants.
    auto vectorType = op.getType().dyn_cast<VectorType>();
    if (!vectorType || vectorType.getRank() != 2) return failure();

    // Only convert splat integer/float vectors.
    auto values = op.getValue().dyn_cast<DenseIntOrFPElementsAttr>();
    if (!values || !values.isSplat()) return failure();
    Attribute value = values.getSplatValue<Attribute>();

    auto elementType = values.getType().getElementType();
    Value splatValue = rewriter.create<spirv::ConstantOp>(
        op.getLoc(), typeConverter->convertType(elementType), value);

    auto matType = typeConverter->convertType(vectorType);
    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(op, matType,
                                                             splatValue);
    return success();
  }
};

/// Converts elementwise ops to SPIR-V joint matrix elementwise ops.
template <typename SrcOpType, typename DstOpType>
struct ConvertElementwiseOp final : public OpConversionPattern<SrcOpType> {
  using OpConversionPattern<SrcOpType>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SrcOpType op, typename SrcOpType::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // All operands should be of joint matrix types.
    for (Value operand : adaptor.getOperands()) {
      if (!operand.getType().isa<spirv::JointMatrixINTELType>())
        return failure();
    }

    // Only support ops with one result.
    if (op->getNumResults() != 1) return failure();

    auto matType = this->typeConverter->convertType(op.getType());
    rewriter.replaceOpWithNewOp<DstOpType>(op, matType, adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Main Pass
//===----------------------------------------------------------------------===//

struct SPIRVVectorToJointOpsPass final
    : public SPIRVVectorToJointOpsBase<SPIRVVectorToJointOpsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    func::FuncOp funcOp = getOperation();

    spirv::TargetEnvAttr targetAttr = getSPIRVTargetEnvAttr(funcOp);
    SPIRVTypeConverter typeConverter(targetAttr);

    // Inject conversion rules for 2-D vector types to joint matrix types.
    //
    // Note that we don't perform legality check here; we just directly convert.
    // Legality check is expected to be done when deciding the whole pass
    // pipeline is feasible and also in SPIR-V ConversionTarget.
    typeConverter.addConversion(
        [&typeConverter](VectorType type) -> Optional<Type> {
          
          if (type.getRank() != 2) return llvm::None;

          Type elementType = typeConverter.convertType(type.getElementType());
          return spirv::JointMatrixINTELType::get(
              elementType, spirv::Scope::Subgroup, type.getDimSize(0),
                type.getDimSize(1), spirv::MatrixLayout::ColumnMajor);
        });


    // Inject another conversion rule for MemRef types.
    //
    // This is for consistency purpose: we will run FlattenMemRefSubspanPass
    // later. That pass flattens all MemRefs into 1-D unknown-sized ones before
    // invoking upstream SPIR-V type converter. So in the end all MemRefs will
    // be converted into SPIR-V runtime arrays. But here if we don't inject the
    // following rule, we'll convert MemRefs into constant-sized arrays. That
    // would cause consistency issues. It's a bit unfortunate to have this; it's
    // a result of performing joint matrix conversions earlier (it needs
    // to be done before FlattenMemRefSubspanPass because we need 2-D MemRefs)
    // and conversions spreading across upstream and IREE repos..
    typeConverter.addConversion(
        [&typeConverter](MemRefType type) -> Optional<Type> {
          if (!type.hasStaticShape()) return llvm::None;
          // In IREE all MemRefs are originated from subspan ops, which should
          // have identity layout.
          if (!type.getLayout().isIdentity()) return llvm::None;
          auto storage = spirv::mapMemorySpaceToOpenCLStorageClass(
              type.getMemorySpaceAsInt());
          auto flattenedType = MemRefType::get(
              ShapedType::kDynamicSize, type.getElementType(), AffineMap(),
              spirv::StorageClassAttr::get(type.getContext(), *storage));
          return typeConverter.convertType(flattenedType);
        });

    // Add unrealized conversion cast ops to bridge type conversions: we are
    // only converting the joint matrix subset; the rest needs to be done
    // at a later stage.
    auto addUnrealizedCast = [](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) {
      auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
      return Optional<Value>(cast.getResult(0));
    };
    typeConverter.addSourceMaterialization(addUnrealizedCast);
    typeConverter.addTargetMaterialization(addUnrealizedCast);

    RewritePatternSet patterns(context);
    patterns.add<
        ConvertConstantMatrix, ConvertVectorContractOp, ConvertVectorTransferOp>(typeConverter,
                                                               context);

    std::unique_ptr<ConversionTarget> target =
        SPIRVConversionTarget::get(targetAttr);
    target->addLegalOp<UnrealizedConversionCastOp>();
    //target->addIllegalDialect<vector::VectorDialect>();

    if (failed(applyPartialConversion(funcOp, *target, std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVVectorToJointOpsPass() {
  return std::make_unique<SPIRVVectorToJointOpsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
