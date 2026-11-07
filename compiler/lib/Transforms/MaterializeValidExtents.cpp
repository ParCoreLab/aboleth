#include "aboleth/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::aboleth {

#define GEN_PASS_DEF_ABOLETHMATERIALIZEVALIDEXTENTSPASS
#include "aboleth/Transforms/Passes.h.inc"

namespace {

Value constant(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIntOp::create(builder, loc, value, 64);
}

Value validCount(OpBuilder &builder, Location loc, Value count, int64_t capacity) {
  Value nonnegative = arith::MaxSIOp::create(builder, loc, count, constant(builder, loc, 0));
  return arith::MinSIOp::create(builder, loc, nonnegative, constant(builder, loc, capacity));
}

class MaterializeValidExtents
    : public impl::AbolethMaterializeValidExtentsPassBase<MaterializeValidExtents> {
public:
  using Base::Base;
  void runOnOperation() override {
    SmallVector<ValidExtentOp> counts;
    SmallVector<BindValidLocalRegionOp> regions;
    getOperation().walk([&](ValidExtentOp op) { counts.push_back(op); });
    getOperation().walk([&](BindValidLocalRegionOp op) { regions.push_back(op); });
    for (ValidExtentOp op : counts) {
      OpBuilder builder(op);
      op.getNormalized().replaceAllUsesWith(validCount(builder, op.getLoc(), op.getCount(), op.getCapacity()));
      op.erase();
    }
    for (BindValidLocalRegionOp region : regions) {
      SmallVector<PutOp> puts;
      for (Operation *user : region.getRegion().getUsers())
        puts.push_back(cast<PutOp>(user));
      OpBuilder regionBuilder(region);
      auto ordinary = BindLocalRegionOp::create(regionBuilder, region.getLoc(),
          LocalRegionType::get(region.getContext()), region.getPointer(), region.getKey());
      for (PutOp put : puts) {
        OpBuilder builder(put);
        Location loc = put.getLoc();
        Value count = trimDeadPadding
            ? validCount(builder, loc, region.getCount(), region.getCapacity())
            : constant(builder, loc, region.getCapacity());
        Value end = arith::MulIOp::create(builder, loc, count,
                                         constant(builder, loc, region.getItemBytes()));
        Value zero = constant(builder, loc, 0);
        Value remaining = arith::SubIOp::create(builder, loc, end, region.getByteOffset());
        remaining = arith::MaxSIOp::create(builder, loc, remaining, zero);
        Value forward = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                              region.getByteOffset(), zero);
        remaining = arith::SelectOp::create(builder, loc, forward, remaining, zero);
        Value requested = arith::ExtSIOp::create(builder, loc, builder.getI64Type(), put.getBytes());
        requested = arith::MaxSIOp::create(builder, loc, requested, zero);
        Value bytes = arith::MinSIOp::create(builder, loc, requested, remaining);
        bytes = arith::TruncIOp::create(builder, loc, builder.getI32Type(), bytes);
        put.getBytesMutable().assign(bytes);
      }
      region.getRegion().replaceAllUsesWith(ordinary.getRegion());
      region.erase();
    }
  }
};

} // namespace
} // namespace mlir::aboleth
