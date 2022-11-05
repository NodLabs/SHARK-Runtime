// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//=== TileAndDistributeToWorkgroupsPass.cpp - Tile to workgroups pass ----===//
//
// This pass distributes the operations within the module to workgroups. This
// pass is created to move tile and distribution out of flow level and into
// the backends. For now this is mostly a bridge pass to connect things during
// the transition, and eventually might just be deprecated in favor of a
// utility method.
//
//===---------------------------------------------------------------------===//
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Transforms.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-conv2d-to-winograd"

namespace mlir {
namespace iree_compiler {

static int constexpr outputTileSize = 6;

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

class ConvertWinogradOutputTransform final
    : public OpRewritePattern<IREE::Flow::WinogradOutputTransformOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  // State that changes during tiling
  struct TensorState {
    SmallVector<OpFoldResult, 4> sizes;  
    SmallVector<OpFoldResult, 4> offsets;  
    SmallVector<OpFoldResult, 4> strides;  
    SmallVector<int64_t, 4> currentSize;  
  };

  // Per loop information
  struct TilingInfo {
    char dim;
    int64_t lo;
    int64_t hi;
    int64_t step;
    int64_t tile;
    int isWorkgroup;
  };

  // The tiling schedule that is implemented
  struct TilingSchedule {
    // Map from tensor -> format
    // "input" : "nhwc"
    std::unordered_map<std::string, std::string> tensorFormat;
    std::vector<TilingInfo> tilingInfo;
  };

  static void initState(TensorState &state, ArrayRef<int64_t> shape,
                        PatternRewriter &rewriter) {
    for (int i = 0; i < shape.size(); i++) {
      state.offsets.push_back(rewriter.getIndexAttr(0));
      state.sizes.push_back(rewriter.getIndexAttr(shape[i]));
      state.strides.push_back(rewriter.getIndexAttr(1));
      state.currentSize.push_back(shape[i]);
    }
  }

  static void generateWorkgroupIdsAndCounts(SmallVectorImpl<Value> &ids,
                                            SmallVectorImpl<Value> &counts,
                                            size_t &numWorkgroups,
                                            Location loc,
                                            PatternRewriter &rewriter) {
    for (int i = 0; i < numWorkgroups; i++) {
      ids.push_back(rewriter.create<IREE::HAL::InterfaceWorkgroupIDOp>(loc, numWorkgroups - i - 1));
      counts.push_back(rewriter.create<IREE::HAL::InterfaceWorkgroupCountOp>(loc, numWorkgroups - i - 1));
    }
  }

  static void computeWorkgroupLoopParams(SmallVectorImpl<Value> &lbs,
                                         SmallVectorImpl<Value> &ubs,
                                         SmallVectorImpl<Value> &steps,
                                         size_t &numWorkgroups,
                                         std::string tensor,
                                         ArrayRef<int64_t> tensorShape,
                                         TilingSchedule &schedule,
                                         Location loc,
                                         PatternRewriter &rewriter) {
    numWorkgroups = 0;
    for (auto info : schedule.tilingInfo) {
      if (info.isWorkgroup) {
        lbs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, info.lo));
        if (info.hi < 0)  {
          size_t pos = schedule.tensorFormat[tensor].find(info.dim);
          info.hi = tensorShape[pos];
        }
        ubs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, info.hi));
        steps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, info.step));
        numWorkgroups++;
      }
    }
  }

  static void computeLoopParams(SmallVectorImpl<Value> &lbs,
                                SmallVectorImpl<Value> &ubs,
                                SmallVectorImpl<Value> &steps,
                                SmallVectorImpl<Value> &tiles,
                                size_t numWorkgroups,
                                std::string tensor,
                                ArrayRef<int64_t> tensorShape,
                                TilingSchedule &schedule,
                                Location loc,
                                PatternRewriter &rewriter) {
    for (auto info : schedule.tilingInfo) {
      if (!info.isWorkgroup) {
        lbs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, info.lo));
        if (info.hi < 0)  {
          size_t pos = schedule.tensorFormat[tensor].find(info.dim);
          info.hi = tensorShape[pos];
        }
        ubs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, info.hi));
        steps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, info.step));
        tiles.push_back(rewriter.create<arith::ConstantIndexOp>(loc, info.tile));
      }
    }
  }

  static void updateWorkgroupLoops(scf::LoopNest &workgroupLoopNest, 
                                   size_t numWorkgroups,
                                   SmallVectorImpl<Value> &ids,
                                   SmallVectorImpl<Value> &counts,
                                   TilingSchedule &schedule,
                                   Location loc,
                                   PatternRewriter &rewriter) {
    auto workgroupLoops = workgroupLoopNest.loops;
    Value lb, step;
    rewriter.setInsertionPoint(workgroupLoops[0]);
    for (int i = 0; i < numWorkgroups; i++) {
      if (i > 0) {
        rewriter.setInsertionPoint(&workgroupLoops[i - 1].getBody()->front());
      }
      AffineExpr s0;
      bindSymbols(rewriter.getContext(), s0);
      auto tileSize = schedule.tilingInfo[i].tile;
      AffineMap map = AffineMap::get(0, 1, {s0 * rewriter.getAffineConstantExpr(tileSize)}, rewriter.getContext());
      lb = rewriter.createOrFold<AffineApplyOp>(loc, map, ValueRange{ids[i]});
      step = rewriter.createOrFold<AffineApplyOp>(loc, map, ValueRange{counts[i]});
      workgroupLoops[i].setLowerBound(lb);
      workgroupLoops[i].setStep(step);
    }
  }

  static void updateLoopsAndState(scf::LoopNest &loopNest, 
                                  ArrayRef<int64_t> outputShape,
                                  SmallVectorImpl<Value> &tileSizes,
                                  size_t numWorkgroups,
                                  TilingSchedule &schedule,
                                  TensorState &inputState,
                                  TensorState &outputState,
                                  Location loc,
                                  PatternRewriter &rewriter) {
    auto loops = loopNest.loops;
    // First reset all input and output offsets to 0
    for (auto &offset : inputState.offsets) {
      offset = rewriter.getIndexAttr(0);
    }
    for (auto &offset : outputState.offsets) {
      offset = rewriter.getIndexAttr(0);
    }
    for (int i = 0; i < loops.size(); i++) {
      auto info = schedule.tilingInfo[i + numWorkgroups];
      size_t pos = schedule.tensorFormat["input"].find(info.dim);
      size_t opos = schedule.tensorFormat["output"].find(info.dim);
      auto iv = loops[i].getInductionVar();
      if ((info.dim == 'h') || (info.dim == 'w')) {
        rewriter.setInsertionPoint(&loops[i].getBody()->front());
        AffineExpr dim0;
        //auto t = rewriter.getAffineConstantExpr(info.tile);
        //auto delta = rewriter.getAffineConstantExpr(inputShape[pos]);
        bindDims(rewriter.getContext(), dim0);
        //AffineMap minMap = AffineMap::get(1, 0, {-dim0 + delta, t}, rewriter.getContext());
        inputState.sizes[pos] = rewriter.getIndexAttr(info.tile);
        //inputState.sizes[pos] = rewriter.createOrFold<AffineMinOp>(loc, minMap, ValueRange{iv});
        //cmpVals.push_back(rewriter.create<arith::CmpIOp>(
        //      loc, arith::CmpIPredicate::eq, inputState.sizes[pos].dyn_cast<Value>(),
        //      tileSizes[i]));
        auto s = rewriter.getAffineConstantExpr(outputTileSize);
        AffineMap outputMap = AffineMap::get(1, 0, {dim0 * s}, rewriter.getContext());
        size_t offpos = info.dim == 'h' ? schedule.tensorFormat["output"].find('H') 
                                        : schedule.tensorFormat["output"].find('W');
        outputState.offsets[offpos] = rewriter.createOrFold<AffineApplyOp>(loc, outputMap, ValueRange{iv});
        //auto delta = rewriter.getAffineConstantExpr(outputShape[offpos]);
        //AffineMap minMap = AffineMap::get(1, 0, {-dim0 + delta, s}, rewriter.getContext());
        //outputState.sizes[offpos] = rewriter.createOrFold<AffineMinOp>(loc, minMap, ValueRange{iv});
        outputState.sizes[offpos] = rewriter.getIndexAttr(outputTileSize);
      }
      if ((info.dim == 'c') || (info.dim == 'n')) {
        // Assumes input and output have c dimension
        inputState.sizes[pos] = rewriter.getIndexAttr(info.tile);
        outputState.sizes[opos] = rewriter.getIndexAttr(info.tile);
        outputState.offsets[opos] = iv;
      }
      inputState.offsets[pos] = iv;
      inputState.currentSize[pos] = info.tile;
    }

    rewriter.setInsertionPoint(&loops.back().getBody()->front());
  }

  static IREE::Flow::DispatchTensorLoadOp getTensorLoadOp(IREE::Flow::WinogradOutputTransformOp op) {
     return op.getOutput().getDefiningOp<IREE::Flow::DispatchTensorLoadOp>();
  }

  static IREE::Flow::DispatchTensorStoreOp getTensorStoreOp(IREE::Flow::WinogradOutputTransformOp op) {
    auto users = llvm::to_vector(op.getResult().getUsers());
    assert(!users.empty());
    return dyn_cast<IREE::Flow::DispatchTensorStoreOp>(users[0]);
  }

  static void updateStateAfterWorkgroupTiling(TensorState &state, std::string tensor,
                                              TilingSchedule &schedule, 
                                              scf::LoopNest &workgroupLoopNest,
                                              PatternRewriter &rewriter) {
    auto workgroupLoops = workgroupLoopNest.loops;
    int i{0};
    for (auto info : schedule.tilingInfo) {
      if (info.isWorkgroup) {
        size_t pos = schedule.tensorFormat[tensor].find(info.dim);
        state.offsets[pos] = workgroupLoops[i].getInductionVar(); 
        state.sizes[pos] = rewriter.getIndexAttr(info.tile);
        state.currentSize[pos] = info.tile;
        i++;
      }
    }
  }

  static Value generateFlowLoads(IREE::Flow::WinogradOutputTransformOp op,
                                 std::string tensor,
                                 TensorState &state,
                                 Type elementType,
                                 Location loc,
                                 PatternRewriter &rewriter) {
    if (tensor == "input") {
      auto tensorLoadOp = getTensorLoadOp(op);
      auto tensorType = RankedTensorType::get(state.currentSize, elementType);
      return rewriter.create<IREE::Flow::DispatchTensorLoadOp>(loc, tensorType,
        tensorLoadOp.getSource(), ValueRange({}), state.offsets, state.sizes, state.strides).getResult();
    } else {
      auto tensorStoreOp = getTensorStoreOp(op);
      auto tensorType = RankedTensorType::get(state.currentSize, elementType);
      return rewriter.create<IREE::Flow::DispatchTensorLoadOp>(loc, tensorType,
        tensorStoreOp.getTarget(), ValueRange({}), state.offsets, state.sizes, state.strides).getResult();
    }
  }

  static void generateFlowStores(IREE::Flow::WinogradOutputTransformOp op,
                                  Value result,
                                  TensorState &state,
                                  Location loc,
                                  PatternRewriter &rewriter) {
      auto tensorStoreOp = getTensorStoreOp(op);
      rewriter.create<IREE::Flow::DispatchTensorStoreOp>(loc, result,
         tensorStoreOp.getTarget(), ValueRange({}), state.offsets, state.sizes, state.strides);
  }

  static Value extractInputSlice(SmallVectorImpl<int64_t> &rankReducedSize,
                                 TensorState &state,
                                 Value inputSlice,
                                 Type elementType,
                                 TilingSchedule &schedule,
                                 SmallVectorImpl<Value> &tileSizes,
                                 Value zero,
                                 Location loc,
                                 PatternRewriter &rewriter) {

    auto tensorType = RankedTensorType::get(rankReducedSize, elementType);
    return rewriter.create<tensor::ExtractSliceOp>(loc, tensorType,
      inputSlice, state.offsets, state.sizes, state.strides).getResult();
    //auto thenBuilder = [&](OpBuilder &builder, Location loc) {
    //  auto tensorType = RankedTensorType::get(rankReducedSize, elementType);
    //  SmallVector<OpFoldResult, 4> sizes = state.sizes;
    //  // Force shapes to be static
    //  for (auto info : schedule.tilingInfo) {
    //    if (!info.isWorkgroup) {
    //      if ((info.dim == 'h') || (info.dim == 'w')) {
    //        size_t pos = schedule.tensorFormat["input"].find(info.dim);
    //        sizes[pos] = rewriter.getIndexAttr(info.tile);
    //      }
    //    }
    //  }
    //  auto res = builder.create<tensor::ExtractSliceOp>(loc, tensorType,
    //    inputSlice, state.offsets, sizes, state.strides).getResult();
    //  builder.create<scf::YieldOp>(loc, res);
    //};

    //auto elseBuilder = [&](OpBuilder &builder, Location loc) {
    //  SmallVector<int64_t> rankReducedDynamicSize;
    //  for (auto size : llvm::enumerate(state.currentSize)) {
    //    auto tensorFormat = schedule.tensorFormat["input"];
    //    if ((tensorFormat[size.index()] == 'h') || (tensorFormat[size.index()] == 'w')) {
    //      rankReducedDynamicSize.push_back(ShapedType::kDynamicSize);
    //      continue;
    //    }
    //    if (size.value() == 1) continue;
    //    rankReducedDynamicSize.push_back(size.value());
    //  }
    //  auto tensorType = RankedTensorType::get(rankReducedDynamicSize, elementType);
    //  auto slice = builder.create<tensor::ExtractSliceOp>(loc, tensorType,
    //    inputSlice, state.offsets, state.sizes, state.strides).getResult();
    //  SmallVector<OpFoldResult> lowPad{rankReducedDynamicSize.size(), builder.getIndexAttr(0)};
    //  SmallVector<OpFoldResult> hiPad;
    //  int k{0};
    //  for (auto infoPair : llvm::enumerate(schedule.tilingInfo)) {
    //    auto info = infoPair.value();
    //    if (info.isWorkgroup) continue;
    //    if ((info.dim == 'h') || (info.dim == 'w')) {
    //      size_t pos = schedule.tensorFormat["input"].find(info.dim);
    //      hiPad.push_back(builder.create<arith::SubIOp>(loc, tileSizes[k++], state.sizes[pos].dyn_cast<Value>()).getResult());
    //    }
    //  }
    //  auto padTensorOp = builder.create<tensor::PadOp>(loc, 
    //     RankedTensorType::get(rankReducedSize, elementType), slice, lowPad, hiPad);
    //  auto &region = padTensorOp.getRegion();
    //  int rank = padTensorOp.getResultType().getRank();
    //  SmallVector<Type> blockArgTypes(rank, rewriter.getIndexType());
    //  SmallVector<Location> blockArgLocs(rank, loc);
    //  builder.createBlock(&region, region.end(), blockArgTypes, blockArgLocs);
    //  builder.create<tensor::YieldOp>(loc, zero);
    //  builder.setInsertionPointAfter(padTensorOp);
    //  builder.create<scf::YieldOp>(loc, padTensorOp.getResult());
    //};

    //return rewriter.create<scf::IfOp>(loc, 
    //   RankedTensorType::get(rankReducedSize, elementType), 
    //   ifCond, thenBuilder, elseBuilder).getResult(0);
  }

  static Value extractOutputSlice(SmallVectorImpl<int64_t> &rankReducedSize,
                                  TensorState &state,
                                  Value iterArg,
                                  Type elementType,
                                  Location loc,
                                  PatternRewriter &rewriter) {
    return rewriter.create<tensor::ExtractSliceOp>(loc, 
          RankedTensorType::get(rankReducedSize, elementType), iterArg,
          state.offsets, state.sizes, state.strides);
  }

  static Value computeTransform(Value input,
                                Value output,
                                Value scratch,
                                Value zero,
                                int ATrows,
                                int ATcols,
                                Value A,
                                Value AT,
                                Type elementType,
                                Location loc,
                                PatternRewriter &rewriter) {
    Value interim, accumulator;
    auto matmulType = RankedTensorType::get({ATcols, ATrows}, elementType);
    accumulator = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{scratch}).result();
    interim = rewriter.create<linalg::MatmulOp>(loc, matmulType, ValueRange{input, A}, accumulator).getResult(0);
    matmulType = RankedTensorType::get({ATrows, ATrows}, elementType);
    accumulator = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{output}).result();
    return rewriter.create<linalg::MatmulOp>(loc, matmulType, ValueRange{AT, interim}, accumulator).getResult(0);
  }

  static Value insertSlice(Value transformed,
                          Value outputSlice,
                          TensorState &state,
                          Location loc,
                          PatternRewriter &rewriter) {
    return rewriter.create<tensor::InsertSliceOp>(loc, transformed, outputSlice,
        state.offsets, state.sizes, state.strides).getResult();
  }

  static LogicalResult applySchedule(IREE::Flow::WinogradOutputTransformOp inputOp,
                                     TilingSchedule &schedule, PatternRewriter &rewriter) {

    // Get input info
    auto loc = inputOp.getLoc();
    auto input = inputOp.getOutput();
    auto inputType = input.getType().cast<ShapedType>();
    auto inputShape = inputType.getShape();
    auto elementType = inputType.getElementType();

    // Get output info
    auto output = inputOp.getResult();
    auto outputType = output.getType().cast<ShapedType>();
    auto outputShape = outputType.getShape();

    // Initialize tensor state
    TensorState inputState, outputState;
    initState(inputState, inputShape, rewriter);
    initState(outputState, outputShape, rewriter);

    // Generate workgroup loop nest
    auto funcOp = inputOp->getParentOfType<func::FuncOp>();
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    SmallVector<Value> lbs, ubs, steps, tiles;
    size_t numWorkgroups{0};
    computeWorkgroupLoopParams(lbs, ubs, steps, numWorkgroups, "input", inputShape, schedule, loc, rewriter);
    // Create additional constants
    auto zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elementType));
    /*---------------------------------------------------------*/
    // Output filter constants
    SmallVector<float> AT{
      1,1, 1, 1,  1,     1,      1,  0,
      0,1,-1, 2, -2,  1./2,  -1./2,  0,
      0,1, 1, 4,  4,  1./4,   1./4,  0,
      0,1,-1, 8, -8,  1./8,  -1./8,  0,
      0,1, 1,16, 16, 1./16,  1./16,  0,
      0,1,-1,32,-32, 1./32, -1./32,  1
    };
    int ATrows = 6;
    int ATcols = 8;
    SmallVector<float> A;
    transpose(AT, A, ATrows, ATcols);
    auto scratch = rewriter.create<tensor::EmptyOp>(loc, SmallVector<int64_t>{ATcols, ATrows}, elementType);
    auto ATValue = rewriter.create<arith::ConstantOp>(loc, DenseFPElementsAttr::get(
      RankedTensorType::get({ATrows, ATcols}, rewriter.getF32Type()), AT));
    auto AValue = rewriter.create<arith::ConstantOp>(loc, DenseFPElementsAttr::get(
      RankedTensorType::get({ATcols, ATrows}, rewriter.getF32Type()), A));
    /*---------------------------------------------------------*/
    rewriter.setInsertionPoint(inputOp);
    SmallVector<Value> ids, counts;
    generateWorkgroupIdsAndCounts(ids, counts, numWorkgroups, loc, rewriter);

    auto workgroupLoopNest = scf::buildLoopNest(rewriter, loc, lbs, ubs, steps, ValueRange({}),
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange outputIvs, ValueRange iterArgs) -> scf::ValueVector {
        return {};
    });
    updateWorkgroupLoops(workgroupLoopNest, numWorkgroups, ids, counts, schedule, loc, rewriter);
    updateStateAfterWorkgroupTiling(inputState, "input", schedule, workgroupLoopNest, rewriter);
    updateStateAfterWorkgroupTiling(outputState, "output", schedule, workgroupLoopNest, rewriter);

    // Generate flow loads
    rewriter.setInsertionPoint(&workgroupLoopNest.loops.back().getBody()->front());
    Value inputSlice = generateFlowLoads(inputOp, "input", inputState, elementType, loc, rewriter);
    Value outputSlice = generateFlowLoads(inputOp, "output", outputState, elementType, loc, rewriter);

    // Generate rest of loops
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    lbs.clear();
    ubs.clear();
    steps.clear();
    computeLoopParams(lbs, ubs, steps, tiles, numWorkgroups, "input", inputShape, schedule, loc, rewriter);
    rewriter.setInsertionPoint(&workgroupLoopNest.loops.back().getBody()->back());
    auto loopNest = scf::buildLoopNest(rewriter, loc, lbs, ubs, steps, ValueRange({outputSlice}),
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange outputIvs, ValueRange iterArgs) -> scf::ValueVector {
        return {iterArgs[0]};
    });

    // Add spirv attributes to loops
    const char *attrName = "iree.spirv.distribute_dim";
    for (int i = loopNest.loops.size() - 2, dim = 0; i >= 0; --i) {
      loopNest.loops[i]->setAttr(attrName, rewriter.getIndexAttr(dim++));
    }

    // Generate flow stores
    generateFlowStores(inputOp, loopNest.getResults()[0], outputState, loc, rewriter);

    updateLoopsAndState(loopNest, outputShape, tiles, numWorkgroups, schedule, inputState,
                        outputState, loc, rewriter);

    // Extract slices
    SmallVector<int64_t> rankReducedSize;
    // Remove unit dims from size during extraction
    for (auto size : inputState.currentSize) {
      if (size == 1) continue;
      rankReducedSize.push_back(size);
    }
    inputSlice = extractInputSlice(rankReducedSize, inputState, inputSlice, elementType, schedule, tiles, zero, loc, rewriter);

    SmallVector<int64_t> rankReducedOutputSize{outputTileSize, outputTileSize};
    Value iterArg = loopNest.loops.back().getRegionIterArg(0);
    outputSlice = extractOutputSlice(rankReducedOutputSize, outputState, iterArg, elementType, loc, rewriter);

    // Do compute
    auto result = computeTransform(inputSlice, outputSlice, scratch, zero, ATrows,
                                   ATcols, AValue, ATValue, elementType, loc, rewriter);

    // Extract slice from result
    //SmallVector<OpFoldResult> strides{rankReducedOutputSize.size(), rewriter.getIndexAttr(1)};
    //SmallVector<OpFoldResult> offsets{rankReducedOutputSize.size(), rewriter.getIndexAttr(0)};
    //SmallVector<OpFoldResult> sizes{rankReducedOutputSize.size(), rewriter.getIndexAttr(1)};
    //sizes[0] = outputState.sizes[1];
    //sizes[1] = outputState.sizes[2];
    //auto resultSlice = rewriter.create<tensor::ExtractSliceOp>(loc, 
    //      RankedTensorType::get(rankReducedOutputSize, elementType), result, offsets, sizes, strides);

    // Insert slice and update yielded value
    auto updatedTensor = insertSlice(result, iterArg, outputState, loc, rewriter);
    if (scf::YieldOp yieldOp = dyn_cast<scf::YieldOp>(loopNest.loops.back().getBody()->getTerminator())) {
      rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, updatedTensor);
    }

    // Remove original ops
    auto tensorStoreOp = getTensorStoreOp(inputOp);
    rewriter.eraseOp(tensorStoreOp);
    auto tensorLoadOp = getTensorLoadOp(inputOp);
    rewriter.eraseOp(inputOp);
    rewriter.eraseOp(tensorLoadOp);

    return success();
  }

  LogicalResult matchAndRewrite(IREE::Flow::WinogradOutputTransformOp inputOp,
                                PatternRewriter &rewriter) const override {

    TilingSchedule schedule;
    schedule.tensorFormat = {{"input", "ptnhwc"}, {"output", "nHWc"}};
    schedule.tilingInfo = {
      /* dim, lo, hi, step, tile, is_workgroup */
      {'c', 0, -1, 32, 32,  1},
      {'h', 0, -1,  1,  1,  0},
      {'w', 0, -1,  1,  1,  0},
      {'c', 0, 32,  1,  1,  0},
      {'n', 0, -1,  1,  1,  0},
    };
    if (failed(applySchedule(inputOp, schedule, rewriter)))
      return failure();
  
    return success();
  }
};

}


namespace {
struct LowerWinogradOutputTransformPass
    : public LowerWinogradOutputTransformBase<
          LowerWinogradOutputTransformPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, IREE::Flow::FlowDialect,
                    IREE::HAL::HALDialect, linalg::LinalgDialect,
                    scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
}  // namespace

void LowerWinogradOutputTransformPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(&getContext());
  patterns.insert<ConvertWinogradOutputTransform>(
      context);
  if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                          std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createLowerWinogradOutputTransformPass() {
  return std::make_unique<LowerWinogradOutputTransformPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
