// Copyright 2022 Nod Labs

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

static int constexpr outputTileSize = 6;
static int operatorId = 0;

static bool hasAllOneValues(DenseIntElementsAttr attr) {
  return llvm::all_of(
      attr, [](APInt element) { return element.getSExtValue() == 1; });
}

static void transpose(SmallVectorImpl<float> &inputTensor, SmallVectorImpl<float> &outputTensor, int dim0, int dim1) {
  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim0; j++) {
      outputTensor.push_back(inputTensor[j * dim1 + i]);
    }
  }
}

namespace {

class ConvertConv2DNhwcHwcf final
    : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  static Value createMatmulBody(ValueRange args, Type elementTy, Location loc, PatternRewriter &rewriter) {
    if (elementTy.isa<FloatType>()) {
      auto res = rewriter.create<arith::MulFOp>(loc, args[0], args[1]);
      return rewriter.create<arith::AddFOp>(loc, args[2], res);
    }
    if (elementTy.isa<IntegerType>()) {
      auto res = rewriter.create<arith::MulIOp>(loc, args[0], args[1]);
      return rewriter.create<arith::AddIOp>(loc, args[2], res);
    }
    return Value();
  }

  static Value createDoubleMatmulBody(ValueRange args, Type elementTy, Location loc, PatternRewriter &rewriter) {
    if (elementTy.isa<FloatType>()) {
      auto res = rewriter.create<arith::MulFOp>(loc, args[0], args[1]);
      res = rewriter.create<arith::MulFOp>(loc, res, args[2]);
      return rewriter.create<arith::AddFOp>(loc, args[3], res);
    }
    if (elementTy.isa<IntegerType>()) {
      auto res = rewriter.create<arith::MulIOp>(loc, args[0], args[1]);
      res = rewriter.create<arith::MulIOp>(loc, res, args[2]);
      return rewriter.create<arith::AddIOp>(loc, args[3], res);
    }
    return Value();
  }

  // Creates:
  // matmul(input, transform)
  // where input: NxHxWxC and transform:TxT to produce output: TxTxCxNxH'xW'
  // or
  // matmul(transform, input)
  // where transform: TxT and input: TxTxCxNxH'xW' to produce output: TxTxCxNxH'xW'
  static Value createInputMatmul(Value tensor, Value transform, SmallVectorImpl<int64_t> &outputShape,
                                 Location loc, PatternRewriter &rewriter) {

    auto tensorType = tensor.getType().cast<ShapedType>();
    auto elementTy = tensorType.getElementType();
    auto transformedType = RankedTensorType::get(outputShape, elementTy);
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, outputShape, elementTy);
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elementTy));
    Value accumulator = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).result();

    SmallVector<AffineExpr> idExprs;
    auto tensorRank = tensorType.getRank();
    int64_t iterationSpaceDim = 7;
    for (auto i = 0; i < iterationSpaceDim; i++)
      idExprs.push_back(getAffineDimExpr(i, rewriter.getContext()));

    SmallVector<AffineExpr> inputExprs, transformExprs, outputExprs;

    SmallVector<AffineExpr> first, second;
    std::string prefix;
    if (tensorRank == 4) {
      // Here we are doing matmul(input, transform)
      // ------------------------------------------------------------------------------
      //              N   H   W   C   T   T   T
      // ------------------------------------------------------------------------------
      // Input Map  (d0, d1, d2, d3, d4, d5, d6) -> (d0, outputTile * d1 + d4, outputTile * d2 + d6, d3)
      // Output Map (d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d3, d0, d1, d2)
      // Filter Map (d0, d1, d2, d3, d4, d5, d6) -> (d6, d5)

      inputExprs = {idExprs[0], 
                    outputTileSize * idExprs[1] + idExprs[4],
                    outputTileSize * idExprs[2] + idExprs[6],
                    idExprs[3]};
      outputExprs = {idExprs[4], idExprs[5], idExprs[3], idExprs[0], idExprs[1], idExprs[2]};
      transformExprs = {idExprs[6], idExprs[5]};
      first = inputExprs;
      second = transformExprs;
      prefix = "IB";

    } else {
      // Here we are doing matmul(transform, input)
      // ------------------------------------------------------------------------------
      //              T   T   C   N   H   W   T
      // ------------------------------------------------------------------------------
      // Input Map  (d0, d1, d2, d3, d4, d5, d6) -> (d6, d1, d2, d3, d4, d5)
      // Output Map (d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5)
      // Filter Map (d0, d1, d2, d3, d4, d5, d6) -> (d0, d6)

      inputExprs.push_back(idExprs[6]);
      for (int i = 0; i < 6; i++) {
        outputExprs.push_back(idExprs[i]);
        if (i > 0) inputExprs.push_back(idExprs[i]);
      }
      transformExprs = {idExprs[0], idExprs[6]};
      first = transformExprs;
      second = inputExprs;
      prefix = "BTIB";

    }
     
    SmallVector<AffineMap> indexingMaps = {
      AffineMap::get(iterationSpaceDim, 0, first, rewriter.getContext()),
      AffineMap::get(iterationSpaceDim, 0, second, rewriter.getContext()),
      AffineMap::get(iterationSpaceDim, 0, outputExprs, rewriter.getContext()),
    };

    SmallVector<StringRef> iteratorTypes;
    for (auto i = 0; i < iterationSpaceDim; i++) {
      iteratorTypes.push_back(i < iterationSpaceDim - 1 ?
        getParallelIteratorTypeName() : getReductionIteratorTypeName());
    }

    auto linalgOp = rewriter.create<linalg::GenericOp>(loc, transformedType, 
      ValueRange({tensor, transform}), accumulator,
      indexingMaps, iteratorTypes, 
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = createMatmulBody(args, elementTy, loc, rewriter);
        b.create<linalg::YieldOp>(loc, result);
      });

    const std::string identifier = "winograd_" + prefix + "_" + std::to_string(operatorId);
    linalgOp->setAttr("id", rewriter.getStringAttr(identifier));

    return linalgOp.getResult(0);
  }

  // Creates:
  // matmul(transform.T, matmul(filter, transform))
  // where filter: HxHxCxF and transform:HxT to produce output: TxTxFxC
  static Value createFilterMatmul(Value tensor, Value transform, SmallVectorImpl<int64_t> &outputShape,
                                  Location loc, PatternRewriter &rewriter) {

    auto tensorType = tensor.getType().cast<ShapedType>();
    auto elementTy = tensorType.getElementType();
    auto transformedType = RankedTensorType::get(outputShape, elementTy);
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, outputShape, elementTy);
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elementTy));
    Value accumulator = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).result();

    SmallVector<AffineExpr> idExprs;
    int64_t iterationSpaceDim = 6;
    for (auto i = 0; i < iterationSpaceDim; i++)
      idExprs.push_back(getAffineDimExpr(i, rewriter.getContext()));

    std::string prefix{"GFGT"};

    // Here we are doing matmul(transform, matmul(filter, transform.T))
    // ------------------------------------------------------------------------------
    //                  T   T   C   F   H   H
    // ------------------------------------------------------------------------------
    // Filter Map     (d0, d1, d2, d3, d4, d5) -> (d4, d5, d2, d3)
    // Output Map     (d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)
    // Transform0 Map (d0, d1, d2, d3, d4, d5) -> (d1, d5)
    // Transform1 Map (d0, d1, d2, d3, d4, d5) -> (d0, d4)

    SmallVector<AffineExpr> filterExprs = {idExprs[4], idExprs[5], idExprs[2], idExprs[3]};
    SmallVector<AffineExpr> outputExprs = {idExprs[0], idExprs[1], idExprs[2], idExprs[3]};
    SmallVector<AffineExpr> transform0Exprs = {idExprs[1], idExprs[5]};
    SmallVector<AffineExpr> transform1Exprs = {idExprs[0], idExprs[4]};

    SmallVector<AffineMap> indexingMaps = {
      AffineMap::get(iterationSpaceDim, 0, filterExprs, rewriter.getContext()),
      AffineMap::get(iterationSpaceDim, 0, transform0Exprs, rewriter.getContext()),
      AffineMap::get(iterationSpaceDim, 0, transform1Exprs, rewriter.getContext()),
      AffineMap::get(iterationSpaceDim, 0, outputExprs, rewriter.getContext()),
    };

    SmallVector<StringRef> iteratorTypes;
    for (auto i = 0; i < iterationSpaceDim; i++) {
      iteratorTypes.push_back(i < iterationSpaceDim - 2 ?
        getParallelIteratorTypeName() : getReductionIteratorTypeName());
    }

    auto linalgOp = rewriter.create<linalg::GenericOp>(loc, transformedType, 
      ValueRange({tensor, transform, transform}), accumulator,
      indexingMaps, iteratorTypes, 
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = createDoubleMatmulBody(args, elementTy, loc, rewriter);
        b.create<linalg::YieldOp>(loc, result);
      });

    const std::string identifier = "winograd_" + prefix + "_" + std::to_string(operatorId);
    linalgOp->setAttr("id", rewriter.getStringAttr(identifier));

    return linalgOp.getResult(0);
  }

  // Creates:
  // matmul(output, transform)
  // where input: TxTxCxNxH'xW' and transform:TxP to produce output: NxH'xW'xCxTxP
  // or
  // matmul(transform, output)
  // where transform: PxT and input: NxH'xW'xCxTxP to produce output: NxH'xW'xCxPxP
  static Value createOutputMatmul(Value tensor, Value transform, SmallVectorImpl<int64_t> &outputShape,
                                 Location loc, PatternRewriter &rewriter) {

    auto tensorType = tensor.getType().cast<ShapedType>();
    auto tensorRank = tensorType.getRank();
    auto elementTy = tensorType.getElementType();
    auto transformedType = RankedTensorType::get(outputShape, elementTy);
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, outputShape, elementTy);
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elementTy));
    Value accumulator = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).result();

    SmallVector<AffineExpr> idExprs;
    int64_t iterationSpaceDim = 7;
    for (auto i = 0; i < iterationSpaceDim; i++)
      idExprs.push_back(getAffineDimExpr(i, rewriter.getContext()));

    SmallVector<AffineExpr> inputExprs, transformExprs, outputExprs;
    SmallVector<AffineExpr> first, second;
    std::string prefix;
    if (tensorRank == 6) {
      // Here we are doing matmul(input, transform)
      // ------------------------------------------------------------------------------
      //              T   P   C   N   H   W   T
      // ------------------------------------------------------------------------------
      // Input Map  (d0, d1, d2, d3, d4, d5, d6) -> (d0, d6, d2, d3, d4, d5)
      // Output Map (d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d2, d0, d1)
      // Filter Map (d0, d1, d2, d3, d4, d5, d6) -> (d6, d1)

      inputExprs = {idExprs[0], idExprs[6], idExprs[2], idExprs[3], idExprs[4], idExprs[5]};
      outputExprs = {idExprs[3], idExprs[4], idExprs[5], idExprs[2], idExprs[0], idExprs[1]};
      transformExprs = {idExprs[6], idExprs[1]};
      first = inputExprs;
      second = transformExprs;
      prefix = "OA";

    } else {
      // Here we are doing matmul(transform, input)
      // ------------------------------------------------------------------------------
      //              N   H   W   C   P   P   T
      // ------------------------------------------------------------------------------
      // Input Map  (d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d6, d5)
      // Output Map (d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5)
      // Filter Map (d0, d1, d2, d3, d4, d5, d6) -> (d4, d6)

      for (int i = 0; i < 6; i++) {
        if (i < 4) inputExprs.push_back(idExprs[i]);
        outputExprs.push_back(idExprs[i]);
      }
      inputExprs.push_back(idExprs[6]);
      inputExprs.push_back(idExprs[5]);
      transformExprs = {idExprs[4], idExprs[6]};
      first = transformExprs;
      second = inputExprs;
      prefix = "ATOA";

    }
     
    SmallVector<AffineMap> indexingMaps = {
      AffineMap::get(iterationSpaceDim, 0, first, rewriter.getContext()),
      AffineMap::get(iterationSpaceDim, 0, second, rewriter.getContext()),
      AffineMap::get(iterationSpaceDim, 0, outputExprs, rewriter.getContext()),
    };

    SmallVector<StringRef> iteratorTypes;
    for (auto i = 0; i < iterationSpaceDim; i++) {
      iteratorTypes.push_back(i < iterationSpaceDim - 1 ?
        getParallelIteratorTypeName() : getReductionIteratorTypeName());
    }

    auto linalgOp = rewriter.create<linalg::GenericOp>(loc, transformedType, 
      ValueRange({tensor, transform}), accumulator,
      indexingMaps, iteratorTypes, 
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = createMatmulBody(args, elementTy, loc, rewriter);
        b.create<linalg::YieldOp>(loc, result);
      });

    const std::string identifier = "winograd_" + prefix + "_" + std::to_string(operatorId);
    linalgOp->setAttr("id", rewriter.getStringAttr(identifier));
    return linalgOp.getResult(0);
  }

  static Value createPermutation(Value tensor, Location loc, PatternRewriter &rewriter,
                                std::vector<int> permutation) {
    
    auto tensorType = tensor.getType().cast<ShapedType>();
    auto elementTy = tensorType.getElementType();
    auto tensorShape = tensorType.getShape();
    auto tensorRank = tensorType.getRank();

    SmallVector<int64_t> outputShape;
    for (unsigned i = 0; i < tensorRank; i++)
      outputShape.push_back(tensorShape[permutation[i]]);

    Value output = rewriter.create<tensor::EmptyOp>(loc, outputShape, elementTy);

    SmallVector<AffineExpr> idExprs;
    SmallVector<AffineExpr> swapExprs;
    for (unsigned i = 0; i < tensorRank; i++)
      idExprs.push_back(getAffineDimExpr(i, rewriter.getContext()));
    for (unsigned i = 0; i < tensorRank; i++)
      swapExprs.push_back(idExprs[permutation[i]]);

    AffineMap inputMap = AffineMap::get(tensorRank, 0, idExprs, rewriter.getContext());
    AffineMap outputMap = AffineMap::get(tensorRank, 0, swapExprs, rewriter.getContext());
    SmallVector<AffineMap> indexingMaps{inputMap, outputMap};
    SmallVector<StringRef> iteratorTypes(tensorRank, getParallelIteratorTypeName());
    return rewriter.create<linalg::GenericOp>(
                             loc, output.getType(), tensor, output,
                             indexingMaps, iteratorTypes,
                             [](OpBuilder &b, Location loc, ValueRange args) {
                               b.create<linalg::YieldOp>(loc, args[0]);
                             })
                         .getResult(0);
  }

  static Value constructOutput(Value tensor, Location loc, PatternRewriter &rewriter,
                               ArrayRef<int64_t> outputShape) {
    auto tensorType = tensor.getType().cast<ShapedType>();
    auto elementTy = tensorType.getElementType();
    auto tensorRank = tensorType.getRank();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, outputShape, elementTy);
    SmallVector<StringRef> iteratorTypes(tensorRank, getParallelIteratorTypeName());
    SmallVector<AffineExpr> idExprs;
    SmallVector<AffineExpr> inputExprs, outputExprs;
    for (unsigned i = 0; i < tensorRank; i++) {
      idExprs.push_back(getAffineDimExpr(i, rewriter.getContext()));
      inputExprs.push_back(idExprs[i]);
    }

    // ------------------------------------------------------------------------------
    //              N   H   W   C   T   T 
    // ------------------------------------------------------------------------------
    // Input Map   (d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)
    // Output Map  (d0, d1, d2, d3, d4, d5) -> (d0, outputTile * d1 + d4, outputTile * d2 + d5, d3)
    outputExprs.push_back(idExprs[0]);
    outputExprs.push_back(idExprs[1] * outputTileSize + idExprs[4]);
    outputExprs.push_back(idExprs[2] * outputTileSize + idExprs[5]);
    outputExprs.push_back(idExprs[3]);

    SmallVector<AffineMap> indexingMaps = {
      AffineMap::get(tensorRank, 0, inputExprs, rewriter.getContext()),
      AffineMap::get(tensorRank, 0, outputExprs, rewriter.getContext()),
    };

    auto linalgOp = rewriter.create<linalg::GenericOp>(
                             loc, empty.getType(), tensor, empty,
                             indexingMaps, iteratorTypes,
                             [&](OpBuilder &b, Location loc, ValueRange args) {
                               b.create<linalg::YieldOp>(loc, args[0]);
                             });
    linalgOp->setAttr("id", rewriter.getStringAttr("winograd_construct_output_" + std::to_string(operatorId)));
                         
    return linalgOp.getResult(0);

  }

  static Value createCollapse(Value tensor, Location loc, PatternRewriter &rewriter,
                              SmallVectorImpl<int64_t> &outputShape,
                              SmallVectorImpl<ReassociationIndices> &reassociations) {
    auto tensorType = tensor.getType().cast<ShapedType>();
    auto elementTy = tensorType.getElementType();
    auto resultType = RankedTensorType::get(outputShape, elementTy);
    return rewriter.create<tensor::CollapseShapeOp>(loc, resultType,
            tensor, reassociations);
  }

  static Value createExpand(Value tensor, Location loc, PatternRewriter &rewriter,
                            SmallVectorImpl<int64_t> &outputShape,
                            SmallVectorImpl<ReassociationIndices> &reassociations) {
    auto tensorType = tensor.getType().cast<ShapedType>();
    auto elementTy = tensorType.getElementType();
    auto resultType = RankedTensorType::get(outputShape, elementTy);
    return rewriter.create<tensor::ExpandShapeOp>(loc, resultType,
            tensor, reassociations);
  }

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {

    // Check that kernel size = 3x3
    auto kernelType = convOp.getInputs()[1].getType().cast<ShapedType>();
    auto kernelShape = kernelType.getShape();
    const int kh = kernelShape[0];
    const int kw = kernelShape[1];
    if ((kh != 3) || (kw != 3)) return failure();

    // Check that strides = 1
    if (!hasAllOneValues(convOp.getStrides())) return failure();

    // Check that dilations = 1
    if (!hasAllOneValues(convOp.getDilations())) return failure();

    auto loc = convOp.getLoc();
    auto elementTy = kernelType.getElementType();
    // Create transformation constants
    // These are tile size specific
    SmallVector<float> BT{
      1,     0, -21./4.,        0,  21./4.,       0, -1, 0,
      0,     1,       1,  -17./4., -17./4.,       1,  1, 0,
      0,    -1,       1,   17./4., -17./4.,      -1,  1, 0,
      0,  1./2,   1./4.,   -5./2.,  -5./4.,       2,  1, 0,
      0,  -1./2,  1./4.,    5./2.,  -5./4.,      -2,  1, 0,
      0,      2,      4,   -5./2.,      -5,   1./2.,  1, 0,
      0,     -2,      4,    5./2.,      -5,  -1./2.,  1, 0,
      0,     -1,      0,   21./4.,       0, -21./4.,  0, 1
    };
    SmallVector<float> G{
      1, 0, 0,
      -2./9., -2./9., -2./9.,
      -2./9., 2./9., -2./9.,
      1./90, 1./45, 2./45,
      1./90, -1./45, 2./45,
      32./45, 16./45, 8./45,
      32./45, -16./45, 8./45,
      0, 0, 1
    };
    SmallVector<float> AT{
      1,1, 1, 1,  1,     1,      1,  0,
      0,1,-1, 2, -2,  1./2,  -1./2,  0,
      0,1, 1, 4,  4,  1./4,   1./4,  0,
      0,1,-1, 8, -8,  1./8,  -1./8,  0,
      0,1, 1,16, 16, 1./16,  1./16,  0,
      0,1,-1,32,-32, 1./32, -1./32,  1
    };
  
    // Transpose the values above
    int inputTileSize = outputTileSize + kh - 1;
    SmallVector<float> B, A;
    transpose(BT, B, inputTileSize, inputTileSize);
    transpose(AT, A, outputTileSize, inputTileSize);

    auto BTValue = rewriter.create<arith::ConstantOp>(loc, DenseFPElementsAttr::get(
      RankedTensorType::get({inputTileSize, inputTileSize}, rewriter.getF32Type()), BT));
    auto BValue = rewriter.create<arith::ConstantOp>(loc, DenseFPElementsAttr::get(
      RankedTensorType::get({inputTileSize, inputTileSize}, rewriter.getF32Type()), B));
    auto GValue = rewriter.create<arith::ConstantOp>(loc, DenseFPElementsAttr::get(
      RankedTensorType::get({inputTileSize, kh}, rewriter.getF32Type()), G));
    auto ATValue = rewriter.create<arith::ConstantOp>(loc, DenseFPElementsAttr::get(
      RankedTensorType::get({outputTileSize, inputTileSize}, rewriter.getF32Type()), AT));
    auto AValue = rewriter.create<arith::ConstantOp>(loc, DenseFPElementsAttr::get(
      RankedTensorType::get({inputTileSize, outputTileSize}, rewriter.getF32Type()), A));
    
    // Construct the transformed input
    Value input = convOp.getInputs()[0];
    auto inputType = input.getType().cast<ShapedType>();
    auto inputShape = inputType.getShape();
    const int in = inputShape[0];
    const int ih = inputShape[1];
    const int iw = inputShape[2];
    const int ic = inputShape[3];

    // First, pad the input (if required)
    int padH = outputTileSize * std::ceil((float) (ih - inputTileSize) / outputTileSize) 
             + inputTileSize - ih;
    int padW = outputTileSize * std::ceil((float) (iw - inputTileSize) / outputTileSize) 
             + inputTileSize - iw;

    Value paddedInput;
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elementTy));
    if ((padW > 0) || (padH > 0)) {
      SmallVector<int64_t> lowPad(4, 0);
      SmallVector<int64_t> highPad{0, padH, padW, 0};
      auto padTensorOp = rewriter.create<tensor::PadOp>(loc, input, lowPad, highPad, ValueRange(), ValueRange());
      auto &region = padTensorOp.getRegion();
      int rank = padTensorOp.getResultType().getRank();
      SmallVector<Type> blockArgTypes(rank, rewriter.getIndexType());
      SmallVector<Location> blockArgLocs(rank, loc);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.createBlock(&region, region.end(), blockArgTypes, blockArgLocs);
      rewriter.create<tensor::YieldOp>(loc, zero);
      paddedInput = padTensorOp.getResult();
    } else {
      paddedInput = input;
    }

    const int ihm = std::ceil((ih + padH - kh + 1) / outputTileSize);
    const int iwm = std::ceil((iw + padW - kw + 1) / outputTileSize);

    // Next, compute the transformed input
    SmallVector<int64_t, 4> outputShape({inputTileSize, inputTileSize, ic, in, ihm, iwm});
    auto IB = createInputMatmul(paddedInput, BValue, outputShape, loc, rewriter);
    auto transformedInput = createInputMatmul(BTValue, IB, outputShape, loc, rewriter);

    // Construct the transformed filter
    Value filter = convOp.getInputs()[1];
    auto filterShape = filter.getType().cast<ShapedType>().getShape();

    const int oc = filterShape[3];
    SmallVector<int64_t> newFilterShape{inputTileSize, inputTileSize, oc, ic};
    auto transformedFilter = createFilterMatmul(filter, GValue, newFilterShape, loc, rewriter);

    // Construct the batch matrix multiplication (element-wise multiply)
    // Input shape: (T, T, Cin, N, H', W') -> (T*T, Cin, N*H'*W')
    // Filter shape: (T, T, Cout, Cin) -> (T*T, Cout, Cin)
    SmallVector<int64_t> collapsedShape{inputTileSize * inputTileSize, ic, in * ihm * iwm};
    SmallVector<ReassociationIndices> reassociations = {{0, 1}, {2}, {3, 4, 5}};
    auto batchInput = createCollapse(transformedInput, loc, rewriter, collapsedShape, reassociations);

    SmallVector<int64_t> collapsedFilterShape{inputTileSize * inputTileSize, oc, ic};
    SmallVector<ReassociationIndices> filterReassociations = {{0, 1}, {2}, {3}};
    auto batchFilter = createCollapse(transformedFilter, loc, rewriter, collapsedFilterShape, filterReassociations);

    SmallVector<int64_t> bmmShape{inputTileSize * inputTileSize, oc, in * ihm * iwm};
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, bmmShape, elementTy);
    Value accumulator = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).result();
    auto bmmType = RankedTensorType::get(bmmShape, elementTy);
    auto bmmOp = rewriter.create<linalg::BatchMatmulOp>(loc, bmmType,
      ValueRange({batchFilter, batchInput}), ValueRange({accumulator}));
    bmmOp->setAttr("id", rewriter.getStringAttr("winograd_bmm_" + std::to_string(operatorId)));
    auto result = bmmOp.getResult(0);

    // Transform the output
    // Output shape: (T*T, Cout, N*H'*W') -> (T, T, Cout, N, H', W')
    SmallVector<int64_t> expandedShape{inputTileSize, inputTileSize, oc, in, ihm, iwm};
    SmallVector<ReassociationIndices> resultReassociations = {{0, 1}, {2}, {3, 4, 5}};
    auto expandedResult = createExpand(result, loc, rewriter, expandedShape, resultReassociations);

    SmallVector<int64_t, 4> transformedOutputShape({in, ihm, iwm, oc, inputTileSize, outputTileSize});
    auto OA = createOutputMatmul(expandedResult, AValue, transformedOutputShape, loc, rewriter);
    transformedOutputShape[4] = outputTileSize;
    auto transformedOutput = createOutputMatmul(ATValue, OA, transformedOutputShape, loc, rewriter);

    // Construct the output
    SmallVector<int64_t> finalOutputShape{in, ih + padH, iw + padW, oc};
    Value finalOutput = constructOutput(transformedOutput, loc, rewriter, finalOutputShape);

    // Extract the relevant slice (only required if padding was applied)
    if ((padW > 0) || (padH > 0)) {
      Value output = convOp.getOutputs()[0];
      auto desiredOutputShape = output.getType().cast<ShapedType>().getShape();
      finalOutput = rewriter.create<tensor::ExtractSliceOp>(loc, output.getType(), finalOutput, 
        ValueRange({}), ValueRange({}), ValueRange({}), 
        rewriter.getI64ArrayAttr({0, 0, 0, 0}),
        rewriter.getI64ArrayAttr(desiredOutputShape),
        rewriter.getI64ArrayAttr({1, 1, 1, 1}));
    }

    rewriter.replaceOp(convOp, ArrayRef<Value>{finalOutput});
    operatorId++;
    return success();
  }
};

struct ConvertConv2DToWinogradPass
    : ConvertConv2DToWinogradBase<ConvertConv2DToWinogradPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<ConvertConv2DNhwcHwcf>(
        context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createConvertConv2DToWinogradPass() {
  return std::make_unique<ConvertConv2DToWinogradPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
