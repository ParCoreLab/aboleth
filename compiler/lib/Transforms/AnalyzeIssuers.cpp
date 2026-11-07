#include "aboleth/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <array>
#include <limits>
#include <optional>

namespace mlir::aboleth {

#define GEN_PASS_DEF_ABOLETHANALYZEISSUERSPASS
#include "aboleth/Transforms/Passes.h.inc"

namespace {

struct AnalyzeIssuers
    : public impl::AbolethAnalyzeIssuersPassBase<AnalyzeIssuers> {
  void runOnOperation() override;
};

// Triton keeps JIT helpers as private functions until after these passes. A
// helper argument can carry a different value per CTA; it is not a uniform
// kernel argument. Resolve only a unique call-site SSA value. Ambiguous or
// recursive callers remain unknown rather than licensing exclusivity.
Value argumentOrigin(Value value) {
  llvm::SmallDenseSet<Value, 8> visited;
  while (auto argument = dyn_cast<BlockArgument>(value)) {
    if (!visited.insert(value).second)
      return value;
    auto function = dyn_cast<triton::FuncOp>(argument.getOwner()->getParentOp());
    if (!function || argument.getOwner() != &function.getBody().front())
      return value;
    auto module = function->getParentOfType<ModuleOp>();
    if (!module)
      return value;
    Value source;
    bool ambiguous = false;
    module.walk([&](triton::CallOp call) {
      if (call.getCallee() != function.getSymName())
        return;
      Value actual = call.getOperand(argument.getArgNumber());
      if (source && source != actual)
        ambiguous = true;
      source = actual;
    });
    if (!source || ambiguous)
      return value;
    value = source;
  }
  return value;
}

// A value every program of the launch computes identically: constants, kernel
// arguments, and arithmetic over them. Loads are excluded; memory may differ
// between the moments two programs read it.
bool isUniform(Value value, unsigned depth = 0) {
  if (depth > 64)
    return false;
  value = argumentOrigin(value);
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto function = dyn_cast<triton::FuncOp>(argument.getOwner()->getParentOp());
    if (!function || SymbolTable::getSymbolVisibility(function) !=
                         SymbolTable::Visibility::Public)
      return false;
    bool called = false;
    function->getParentOfType<ModuleOp>().walk([&](triton::CallOp call) {
      called |= call.getCallee() == function.getSymName();
    });
    return !called;
  }
  Operation *producer = value.getDefiningOp();
  if (!producer || isa<triton::GetProgramIdOp>(producer))
    return false;
  if (isa<arith::ConstantOp>(producer))
    return true;
  if (producer->getDialect() != producer->getContext()->getLoadedDialect(
                                    arith::ArithDialect::getDialectNamespace()))
    return false;
  return llvm::all_of(producer->getOperands(), [&](Value operand) {
    return isUniform(operand, depth + 1);
  });
}

bool isNonZeroConstant(Value value) {
  APInt constant;
  auto op = value.getDefiningOp<arith::ConstantOp>();
  auto attribute = op ? dyn_cast<IntegerAttr>(op.getValue()) : IntegerAttr{};
  return attribute && !attribute.getValue().isZero();
}

// The program-id axis `value` is an injective function of, if any. Index
// arithmetic is assumed not to overflow, as for any address computation.
std::optional<int> injectiveAxis(Value value, unsigned depth = 0) {
  if (depth > 64)
    return std::nullopt;
  value = argumentOrigin(value);
  Operation *producer = value.getDefiningOp();
  if (!producer)
    return std::nullopt;
  if (auto programId = dyn_cast<triton::GetProgramIdOp>(producer))
    return static_cast<int>(programId.getAxis());
  if (isa<arith::ExtSIOp, arith::ExtUIOp>(producer))
    return injectiveAxis(producer->getOperand(0), depth + 1);
  if (isa<arith::AddIOp, arith::XOrIOp>(producer)) {
    Value lhs = producer->getOperand(0), rhs = producer->getOperand(1);
    if (isUniform(rhs))
      return injectiveAxis(lhs, depth + 1);
    if (isUniform(lhs))
      return injectiveAxis(rhs, depth + 1);
    return std::nullopt;
  }
  if (isa<arith::SubIOp>(producer)) {
    Value lhs = producer->getOperand(0), rhs = producer->getOperand(1);
    if (isUniform(rhs))
      return injectiveAxis(lhs, depth + 1);
    if (isUniform(lhs))
      return injectiveAxis(rhs, depth + 1);
    return std::nullopt;
  }
  if (isa<arith::MulIOp>(producer)) {
    Value lhs = producer->getOperand(0), rhs = producer->getOperand(1);
    if (isNonZeroConstant(rhs))
      return injectiveAxis(lhs, depth + 1);
    if (isNonZeroConstant(lhs))
      return injectiveAxis(rhs, depth + 1);
  }
  return std::nullopt;
}

std::optional<int64_t> constantInteger(Value value) {
  auto op = value.getDefiningOp<arith::ConstantOp>();
  auto attr = op ? dyn_cast<IntegerAttr>(op.getValue()) : IntegerAttr{};
  if (!attr || attr.getValue().getBitWidth() > 64)
    return std::nullopt;
  return attr.getInt();
}

// A coordinate ranges from zero to its grid extent. A modulo rotation is a
// permutation only when the grid extent equals its period; retain that fact
// for a launch check instead of assuming the client's intended grid.
struct Coordinate {
  int axis;
  int64_t period;
};

std::optional<Coordinate> coordinate(Value value) {
  value = argumentOrigin(value);
  if (auto pid = value.getDefiningOp<triton::GetProgramIdOp>())
    return Coordinate{static_cast<int>(pid.getAxis()), 0};
  Operation *op = value.getDefiningOp();
  if (!op || !isa<arith::RemSIOp, arith::RemUIOp>(op))
    return std::nullopt;
  auto period = constantInteger(op->getOperand(1));
  if (!period || *period <= 0 ||
      *period > std::numeric_limits<int32_t>::max())
    return std::nullopt;
  Value numerator = op->getOperand(0);
  int64_t shift = 0;
  if (auto add = numerator.getDefiningOp<arith::AddIOp>()) {
    auto offset = constantInteger(add.getRhs());
    Value pid = add.getLhs();
    if (!offset) {
      offset = constantInteger(add.getLhs());
      pid = add.getRhs();
    }
    if (!offset)
      return std::nullopt;
    shift = *offset;
    numerator = pid;
  }
  auto pid = numerator.getDefiningOp<triton::GetProgramIdOp>();
  // Signed remainder of a negative value is not a rotation in [0, period).
  // Also prove the addition does not wrap before applying the remainder.
  if (!pid || shift < 0 ||
      shift > std::numeric_limits<int32_t>::max() - (*period - 1))
    return std::nullopt;
  return Coordinate{static_cast<int>(pid.getAxis()), *period};
}

struct IssuerProof {
  SmallVector<int32_t, 2> axes;
  std::array<int64_t, 3> grid = {0, 0, 0};
};

std::optional<IssuerProof> proveIndex(Value index) {
  index = argumentOrigin(index);
  if (auto item = coordinate(index)) {
    IssuerProof proof;
    proof.axes.push_back(item->axis);
    proof.grid[item->axis] = item->period;
    return proof;
  }
  // lane * width + peer, including a rotated peer coordinate. The peer
  // extent must equal width: a larger launch aliases the next lane's range.
  // A smaller launch is also rejected rather than inventing a different
  // endpoint-table layout. The lowering bounds the complete index in i32.
  if (auto add = index.getDefiningOp<arith::AddIOp>()) {
    for (int side = 0; side < 2; ++side) {
      Value term = argumentOrigin(add->getOperand(side));
      auto product = term.getDefiningOp<arith::MulIOp>();
      auto peer = coordinate(add->getOperand(1 - side));
      if (!peer)
        continue;
      // Canonicalization removes lane * 1. The remaining sum of two axes
      // is injective with the lower (peer) axis constrained to extent one.
      std::optional<int64_t> width = product ? constantInteger(product.getRhs())
                                           : std::optional<int64_t>{1};
      Value laneValue = product ? product.getLhs() : term;
      if (product && !width) {
        width = constantInteger(product.getLhs());
        laneValue = product.getRhs();
      }
      laneValue = argumentOrigin(laneValue);
      auto lane = laneValue.getDefiningOp<triton::GetProgramIdOp>();
      if (!lane || !width || *width <= 0 ||
          *width > std::numeric_limits<int32_t>::max() ||
          static_cast<int>(lane.getAxis()) == peer->axis ||
          (peer->period && peer->period != *width) ||
          (!product && lane && static_cast<int>(lane.getAxis()) < peer->axis))
        continue;
      IssuerProof proof;
      proof.axes = {static_cast<int32_t>(lane.getAxis()), peer->axis};
      llvm::sort(proof.axes);
      proof.grid[peer->axis] = *width;
      return proof;
    }
  }
  if (auto axis = injectiveAxis(index)) {
    IssuerProof proof;
    proof.axes.push_back(*axis);
    return proof;
  }
  // An arbitrary table load remains unproven. A runtime permutation fact
  // must be validated and included in artifact identity before using it.
  return std::nullopt;
}

// Separate JIT helper specializations can spell the same endpoint expression
// with distinct SSA values. Compare only the proven arithmetic vocabulary;
// loads and arbitrary calls never become registration/permutation facts.
bool equivalentIndex(Value first, Value second, unsigned depth = 0) {
  first = argumentOrigin(first);
  second = argumentOrigin(second);
  if (first == second)
    return true;
  if (depth > 64)
    return false;
  Operation *a = first.getDefiningOp(), *b = second.getDefiningOp();
  if (!a || !b || a->getName() != b->getName())
    return false;
  if (auto pid = dyn_cast<triton::GetProgramIdOp>(a))
    return pid.getAxis() == cast<triton::GetProgramIdOp>(b).getAxis();
  if (auto constant = dyn_cast<arith::ConstantOp>(a))
    return constant.getValue() == cast<arith::ConstantOp>(b).getValue();
  if (!isa<arith::ExtSIOp, arith::ExtUIOp, arith::AddIOp, arith::SubIOp,
           arith::MulIOp, arith::XOrIOp, arith::RemSIOp, arith::RemUIOp>(a) ||
      a->getNumOperands() != b->getNumOperands() || first.getType() != second.getType())
    return false;
  for (auto [lhs, rhs] : llvm::zip_equal(a->getOperands(), b->getOperands()))
    if (!equivalentIndex(lhs, rhs, depth + 1))
      return false;
  return true;
}

} // namespace

void AnalyzeIssuers::runOnOperation() {
  // Every selection from one table must use one index: two different indices,
  // even each injective, can name the same endpoint from two programs.
  llvm::DenseMap<Value, SmallVector<EndpointOp>> byTable;
  getOperation().walk(
      [&](EndpointOp op) { byTable[argumentOrigin(op.getTable())].push_back(op); });
  for (auto &[table, selections] : byTable) {
    for (EndpointOp selection : selections) {
      selection.removeExclusiveAxisAttr();
      selection.removeExclusiveAxesAttr();
      selection.removeExclusiveGridShapeAttr();
    }
    Value index = selections.front().getIndex();
    if (!isUniform(table))
      continue;
    if (llvm::any_of(selections,
                     [&](EndpointOp op) { return !equivalentIndex(op.getIndex(), index); }))
      continue;
    auto proof = proveIndex(index);
    if (!proof)
      continue;
    Builder builder(&getContext());
    for (EndpointOp selection : selections) {
      if (proof->axes.size() == 1)
        selection.setExclusiveAxisAttr(
            builder.getI32IntegerAttr(proof->axes.front()));
      else
        selection.setExclusiveAxesAttr(builder.getDenseI32ArrayAttr(proof->axes));
      if (llvm::any_of(proof->grid, [](int64_t size) { return size != 0; }))
        selection.setExclusiveGridShapeAttr(builder.getDenseI64ArrayAttr(proof->grid));
      for (Operation *user : selection->getUsers()) {
        if (auto submission = dyn_cast<SubmitOp>(user))
          submission.setExclusiveIssuerAttr(builder.getUnitAttr());
        else if (auto drain = dyn_cast<DrainOp>(user))
          drain.setExclusiveIssuerAttr(builder.getUnitAttr());
      }
    }
  }
}

} // namespace mlir::aboleth
