#include "aboleth/Transforms/Passes.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <limits>
#include <system_error>

using namespace mlir;
using namespace mlir::aboleth;

LogicalResult ShapeFamilyOp::verify() {
  int64_t minimum = getMinimumBoundAttr().getInt();
  int64_t maximum = getMaximumBoundAttr().getInt();
  int64_t slots = getSlotsAttr().getInt();
  if (getName().empty() || minimum <= 0 || maximum < minimum || slots <= 0)
    return emitOpError("requires a name, positive bounds/slots and minimumBound <= maximumBound");
  if (maximum > (int64_t{1} << 62))
    return emitOpError("power-of-two bound rounding would overflow signed i64");
  if (auto bound = getIterationBoundAttr(); bound && bound.getInt() < minimum)
    return emitOpError("iterationBound must be at least minimumBound");
  return success();
}

namespace mlir::aboleth {
#define GEN_PASS_DEF_ABOLETHPLANSHAPESPASS
#include "aboleth/Transforms/Passes.h.inc"
namespace {
int64_t ceilPowerOfTwo(int64_t value) {
  int64_t bound = 1;
  while (bound < value)
    bound *= 2;
  return bound;
}
class PlanShapes : public impl::AbolethPlanShapesPassBase<PlanShapes> {
public:
  using Base::Base;
  void runOnOperation() override {
    llvm::json::Array families;
    bool invalid = false;
    getOperation().walk([&](ShapeFamilyOp family) {
      int64_t largest = ceilPowerOfTwo(family.getMaximumBoundAttr().getInt());
      int64_t slots = family.getSlotsAttr().getInt();
      SmallVector<int64_t> bounds;
      if (specialize && family.getUniformSelector()) {
        // Specialize the declared repeated-use domain and retain one general
        // fallback. Inputs above that domain remain legal; initialization or
        // mixed phases need not instantiate every intermediate capacity.
        int64_t limit = family.getMaximumBoundAttr().getInt();
        if (auto iteration = family.getIterationBoundAttr())
          limit = std::min(limit, iteration.getInt());
        int64_t specialized = ceilPowerOfTwo(limit);
        int64_t bound = ceilPowerOfTwo(family.getMinimumBoundAttr().getInt());
        for (;;) {
          bounds.push_back(bound);
          if (bound == specialized)
            break;
          bound *= 2;
        }
        if (bounds.back() != largest)
          bounds.push_back(largest);
      } else {
        bounds.push_back(largest);
      }
      if (bounds.size() > uint64_t(std::numeric_limits<int64_t>::max()/slots)) {
        family.emitOpError("private control domain count overflows signed i64");
        invalid = true;
        return;
      }
      Builder builder(family.getContext());
      family->setAttr("variantBounds", builder.getDenseI64ArrayAttr(bounds));
      family->setAttr("controlDomains", builder.getI64IntegerAttr(bounds.size()*slots));
      llvm::json::Array variants;
      for (auto [index, bound] : llvm::enumerate(bounds)) {
        llvm::json::Object variant;
        variant["capacity"] = bound;
        // Keep compact affine metadata: domain = base + the client's slot.
        // Slot ownership is a client fact, not an Aboleth stream schedule.
        variant["control_base"] = int64_t(index)*slots;
        variant["control_count"] = slots;
        variants.push_back(std::move(variant));
      }
      llvm::json::Object result;
      result["name"] = family.getName();
      result["maximum_bound"] = family.getMaximumBoundAttr().getInt();
      result["slots"] = slots;
      result["selector"] = "first_capacity_at_least_runtime_bound";
      if (auto iteration = family.getIterationBoundAttr())
        result["iteration_bound"] = iteration.getInt();
      result["uniform_required"] = bounds.size() > 1;
      result["private_control"] = "per_variant_slot";
      result["mutually_exclusive_within_slot"] = true;
      result["payload_aliases_proven"] = false;
      result["variants"] = std::move(variants);
      families.push_back(std::move(result));
    });
    if (invalid)
      return signalPassFailure();
    if (!emitPlanJson.empty()) {
      std::error_code error;
      llvm::raw_fd_ostream stream(emitPlanJson, error);
      if (error) {
        getOperation().emitError("cannot write shape-family plan: ") << error.message();
        return signalPassFailure();
      }
      llvm::json::Object document;
      document["families"] = std::move(families);
      llvm::json::OStream json(stream, 2);
      json.value(llvm::json::Value(std::move(document)));
      stream << '\n';
      stream.close();
      if (stream.has_error()) {
        getOperation().emitError("failed to write shape-family plan");
        signalPassFailure();
      }
    }
  }
};
} // namespace
} // namespace mlir::aboleth
