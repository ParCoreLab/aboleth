#include "aboleth/IR/Dialect.h"
#include "aboleth/Target/Passes.h"
#include "aboleth/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassRegistry.h"
#include "triton/Tools/PluginUtils.h"
#include "llvm/Support/ErrorHandling.h"

#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

using mlir::PassManager;
using mlir::Value;
using namespace mlir::aboleth;

constexpr const char *kPluginName = "Aboleth";
constexpr const char *kPluginVersion = "0.1.0";

void requireArity(const std::vector<Value> &operands, size_t minimum,
                  size_t maximum, const char *operation) {
  if (operands.size() < minimum || operands.size() > maximum)
    llvm::report_fatal_error(llvm::Twine(operation) +
                             " received an invalid operand count");
}

using PassFactory = std::unique_ptr<mlir::Pass> (*)();

// A pass without options ignores its arguments.
template <PassFactory create>
void addPass(PassManager *manager, const std::vector<std::string> &) {
  manager->addPass(create());
}

// Every pass registers under its default options for triton-opt.
template <PassFactory create>
void registerPass() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> { return create(); });
}

void addSimplifyDataflow(PassManager *manager,
                         const std::vector<std::string> &args) {
  AbolethSimplifyCommunicationDataflowPassOptions options;
  if (!args.empty() && args.front() == "unoptimized")
    options.storageAliases = false;
  if (args.size() > 1 && args[1] == "keep-dead")
    options.deadCommunication = false;
  manager->addPass(createAbolethSimplifyCommunicationDataflowPass(options));
}

void addForwardInlinePayloads(PassManager *manager,
                              const std::vector<std::string> &args) {
  AbolethForwardInlinePayloadsPassOptions options;
  if (!args.empty())
    options.inlineBytes = std::stoll(args.front());
  manager->addPass(createAbolethForwardInlinePayloadsPass(options));
}

void addPlanTransport(PassManager *manager,
                      const std::vector<std::string> &args) {
  AbolethPlanTransportPassOptions options;
  if (!args.empty())
    options.emitPlanJson = args.front();
  manager->addPass(createAbolethPlanTransportPass(options));
}

void addPlanShapes(PassManager *manager,
                   const std::vector<std::string> &args) {
  AbolethPlanShapesPassOptions options;
  if (!args.empty())
    options.emitPlanJson = args.front();
  if (args.size() > 1 && args[1] == "unoptimized")
    options.specialize = false;
  manager->addPass(createAbolethPlanShapesPass(options));
}

void addPlanEndpoints(PassManager *manager,
                      const std::vector<std::string> &args) {
  AbolethPlanEndpointsPassOptions options;
  if (!args.empty())
    options.emitPlanJson = args.front();
  if (args.size() > 1 && args[1] == "unoptimized")
    options.fanout = false;
  if (args.size() > 2 && args[2] == "ungrouped")
    options.coalesceReceives = false;
  manager->addPass(createAbolethPlanEndpointsPass(options));
}

void addPlanStorage(PassManager *manager,
                    const std::vector<std::string> &args) {
  AbolethPlanStoragePassOptions options;
  if (!args.empty())
    options.emitPlanJson = args.front();
  if (args.size() > 1 && args[1] == "unoptimized")
    options.optimizeBounds = false;
  manager->addPass(createAbolethPlanStoragePass(options));
}

void addMaterializeTransportPhase(
    PassManager *manager, const std::vector<std::string> &args) {
  AbolethMaterializeTransportPhasePassOptions options;
  if (!args.empty())
    options.planPath = args.front();
  manager->addPass(
      createAbolethMaterializeTransportPhasePass(options));
}

void addMaterializeCompletion(PassManager *manager,
                              const std::vector<std::string> &args) {
  // Positional: the module-wide interval an external plan chose (may be
  // empty), the kernel's transport, its endpoints' queue capacity, and
  // "unplanned" to ignore declared reuse.
  AbolethMaterializeCompletionPassOptions options;
  if (!args.empty() && !args[0].empty())
    options.completionInterval = std::stoll(args[0]);
  if (args.size() > 1)
    options.transport = args[1];
  if (args.size() > 2 && !args[2].empty())
    options.queueCapacity = std::stoll(args[2]);
  if (args.size() > 3)
    options.planReuse = args[3] != "unplanned";
  manager->addPass(createAbolethMaterializeCompletionPass(options));
}

void addMaterializeValidExtents(PassManager *manager,
                                const std::vector<std::string> &args) {
  AbolethMaterializeValidExtentsPassOptions options;
  if (!args.empty() && args[0] == "unoptimized")
    options.trimDeadPadding = false;
  manager->addPass(createAbolethMaterializeValidExtentsPass(options));
}

void addCoalesceSubmissions(PassManager *manager,
                            const std::vector<std::string> &args) {
  AbolethCoalesceSubmissionsPassOptions options;
  if (!args.empty())
    options.transport = args.front();
  manager->addPass(createAbolethCoalesceSubmissionsPass(options));
}

void addDeferAnnouncement(PassManager *manager,
                          const std::vector<std::string> &args) {
  AbolethDeferAnnouncementPassOptions options;
  if (!args.empty())
    options.transport = args.front();
  manager->addPass(createAbolethDeferAnnouncementPass(options));
}

void addOrderPublications(PassManager *manager,
                          const std::vector<std::string> &args) {
  AbolethOrderPublicationsPassOptions options;
  if (!args.empty())
    options.transport = args.front();
  manager->addPass(createAbolethOrderPublicationsPass(options));
}

void addElideEpochs(PassManager *manager,
                    const std::vector<std::string> &args) {
  AbolethElideEpochsPassOptions options;
  if (!args.empty())
    options.transport = args.front();
  manager->addPass(createAbolethElideEpochsPass(options));
}

void addPlanPublication(PassManager *manager,
                        const std::vector<std::string> &args) {
  // Positional: the kernel's transport (may be empty), then "strict" when every
  // receiver registration is strictly ordered.
  AbolethPlanPublicationPassOptions options;
  if (!args.empty())
    options.transport = args[0];
  if (args.size() > 1 && !args[1].empty()) {
    if (args[1] != "strict")
      llvm::report_fatal_error(
          llvm::Twine("aboleth_plan_publication: unknown ordering ") +
          args[1]);
    options.strictRegistrations = true;
  }
  manager->addPass(createAbolethPlanPublicationPass(options));
}

void addLowerP2P(PassManager *manager,
                 const std::vector<std::string> &args) {
  AbolethLowerP2PPassOptions options;
  // Positional: args[0] is the copy mode (may be empty), args[1] the
  // emission mode, mirroring aboleth_lower_rdma's argument shape.
  if (!args.empty())
    options.copyMode = args.front();
  if (args.size() > 1 && !args[1].empty()) {
    if (args[1] != "detached")
      llvm::report_fatal_error(
          llvm::Twine("aboleth_lower_p2p: unknown mode ") + args[1]);
    options.detached = true;
  }
  if (args.size() > 2)
    options.hipDeviceLibrary = args[2];
  manager->addPass(createAbolethLowerP2PPass(options));
}

void addLowerProxy(PassManager *manager,
                   const std::vector<std::string> &args) {
  AbolethLowerProxyPassOptions options;
  if (!args.empty()) options.hipDeviceLibrary = args.front();
  if (args.size() > 1) options.detached = args[1] == "detached";
  manager->addPass(createAbolethLowerProxyPass(options));
}

void addLowerRDMA(PassManager *manager,
                  const std::vector<std::string> &args) {
  // Named options, spelled as for --aboleth-lower-rdma: `name=value`, or a
  // bare `name` for a boolean that is set.
  AbolethLowerRDMAPassOptions options;
  for (const std::string &arg : args) {
    if (arg.empty())
      continue;
    auto [name, value] = llvm::StringRef(arg).split('=');
    bool set = value.empty() || value == "true" || value == "1";
    if (!set && value != "false" && value != "0" &&
        (name == "detached" || name == "exclusive" ||
         name == "gpu-scope-doorbell" || name == "invariant-descriptors" ||
         name == "vector-wqe-stores" || name == "self-reclaim"))
      llvm::report_fatal_error(llvm::Twine("aboleth_lower_rdma: ") + name +
                               " takes a boolean, not " + value);
    if (name == "bitcode-path")
      options.bitcodePath = value.str();
    else if (name == "detached")
      options.detached = set;
    else if (name == "exclusive")
      options.exclusive = set;
    else if (name == "gpu-scope-doorbell")
      options.gpuScopeDoorbell = set;
    else if (name == "self-reclaim")
      options.selfReclaim = set;
    else if (name == "invariant-descriptors")
      options.invariantDescriptors = set;
    else if (name == "vector-wqe-stores")
      options.vectorWqeStores = set;
    else
      llvm::report_fatal_error(
          llvm::Twine("aboleth_lower_rdma: unknown option ") + arg);
  }
  manager->addPass(createAbolethLowerRDMAPass(options));
}

void addLowerIonic(PassManager *manager, const std::vector<std::string> &args) {
  AbolethLowerIonicPassOptions options;
  if (!args.empty()) options.bitcodePath = args.front();
  if (args.size() > 1) options.detached = args[1] == "detached";
  if (args.size() > 2)
    options.shareProducerFence = args[2] == "share-producer-fence";
  if (args.size() > 3)
    options.fencedPublication = args[3] == "fenced-publication";
  manager->addPass(createAbolethLowerIonicPass(options));
}

void addLowerEFA(PassManager *manager, const std::vector<std::string> &args) {
  AbolethLowerEFAPassOptions options;
  if (!args.empty())
    options.publicationScheme = args.front();
  if (args.size() > 1)
    options.bitcodePath = args[1];
  manager->addPass(createAbolethLowerEFAPass(options));
}

void registerDialect(mlir::DialectRegistry *registry) {
  registry->insert<AbolethDialect>();
}

// A frontend endpoint is either a descriptor pointer, bound here, or an
// endpoint `aboleth.endpoint` already selected from a table.
Value bindEndpoint(::TritonOpBuilder &builder, Value endpoint) {
  if (mlir::isa<EndpointType>(endpoint.getType()))
    return endpoint;
  return builder.create<BindEndpointOp>(
      EndpointType::get(builder.getContext()), endpoint);
}

// A signal publishes an integer epoch by value, or the epoch a registered
// local cell holds.
Value signalSource(::TritonOpBuilder &builder, Value source, Value key) {
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(source.getType())) {
    if (integer.getWidth() == 64)
      return source;
    return builder.create<mlir::arith::ExtUIOp>(
        builder.getBuilder().getI64Type(), source);
  }
  return builder.create<BindLocalRegionOp>(
      LocalRegionType::get(builder.getContext()), source, key);
}

void createEndpoint(::TritonOpBuilder &builder, std::vector<Value> &operands) {
  requireArity(operands, 3, 3, "aboleth.endpoint");
  operands[0] = builder.create<EndpointOp>(
      EndpointType::get(builder.getContext()), operands[1], operands[2],
      /*exclusiveAxis=*/mlir::IntegerAttr{},
      /*exclusiveAxes=*/mlir::DenseI32ArrayAttr{},
      /*exclusiveGridShape=*/mlir::DenseI64ArrayAttr{});
}

void createBeginGroup(::TritonOpBuilder &builder,
                      std::vector<Value> &operands) {
  requireArity(operands, 2, 2, "aboleth.begin_group");
  auto *context = builder.getContext();
  Value endpoint = bindEndpoint(builder, operands[1]);
  operands[0] =
      builder.create<BeginGroupOp>(GroupType::get(context), endpoint);
}

int64_t constexprInteger(Value value, const char *name) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  auto integer = constant ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()) : mlir::IntegerAttr{};
  if (!integer)
    llvm::report_fatal_error(llvm::Twine(name) + " must be a constexpr integer");
  return integer.getInt();
}

void createValidExtent(::TritonOpBuilder &builder, std::vector<Value> &operands) {
  requireArity(operands, 3, 3, "aboleth.valid_extent");
  operands[0] = builder.create<ValidExtentOp>(builder.getBuilder().getI64Type(), operands[1],
      builder.getBuilder().getI64IntegerAttr(constexprInteger(operands[2], "capacity")));
}

void createValidRegion(::TritonOpBuilder &builder, std::vector<Value> &operands) {
  requireArity(operands, 7, 7, "aboleth.valid_region");
  operands[0] = builder.create<BindValidLocalRegionOp>(LocalRegionType::get(builder.getContext()),
      operands[1], operands[2], operands[3], operands[4],
      builder.getBuilder().getI64IntegerAttr(constexprInteger(operands[5], "capacity")),
      builder.getBuilder().getI64IntegerAttr(constexprInteger(operands[6], "item_bytes")));
}

void createPut(::TritonOpBuilder &builder, std::vector<Value> &operands) {
  requireArity(operands, 8, 9, "aboleth.put");
  Value after = operands.size() == 9 ? operands[8] : Value{};
  auto *context = builder.getContext();
  Value endpoint = bindEndpoint(builder, operands[1]);
  Value destination = builder.create<BindRemoteRegionOp>(
      RemoteRegionType::get(context), operands[2], operands[6]);
  Value source = operands[3];
  if (!mlir::isa<LocalRegionType>(source.getType()))
    source = builder.create<BindLocalRegionOp>(
        LocalRegionType::get(context), source, operands[5]);
  operands[0] = builder.create<PutOp>(
      TokenType::get(context), endpoint, destination, source, operands[4],
      operands[7], after);
}

// Payload values are variadic, so the optional ordering token is recognized by
// its type instead of by position: a token can never be a payload value.
void createPutInline(::TritonOpBuilder &builder,
                     std::vector<Value> &operands) {
  requireArity(operands, 6, std::numeric_limits<size_t>::max(),
               "aboleth.put_inline");
  Value after;
  size_t valueEnd = operands.size();
  if (mlir::isa<TokenType>(operands.back().getType())) {
    after = operands.back();
    --valueEnd;
  }
  if (valueEnd < 6)
    llvm::report_fatal_error(
        "aboleth.put_inline requires at least one payload value");
  auto *context = builder.getContext();
  Value endpoint = bindEndpoint(builder, operands[1]);
  Value destination = builder.create<BindRemoteRegionOp>(
      RemoteRegionType::get(context), operands[2], operands[3]);
  std::vector<Value> values(
      operands.begin() + 5, operands.begin() + valueEnd);
  operands[0] = builder.create<PutInlineOp>(
      TokenType::get(context), endpoint, destination,
      mlir::ValueRange(values), operands[4], after);
}

void createSignal(::TritonOpBuilder &builder, std::vector<Value> &operands) {
  requireArity(operands, 6, 7, "aboleth.signal");
  Value after = operands.size() == 7 ? operands[6] : Value{};
  auto *context = builder.getContext();
  Value endpoint = bindEndpoint(builder, operands[1]);
  Value destination = builder.create<BindRemoteRegionOp>(
      RemoteRegionType::get(context), operands[2], operands[5]);
  Value source = signalSource(builder, operands[3], operands[4]);
  operands[0] = builder.create<SignalOp>(
      TokenType::get(context), endpoint, destination, source, after,
      Value{}, /*weakPublication=*/mlir::UnitAttr{},
      /*fencedSignal=*/mlir::UnitAttr{}, /*completedPrefix=*/mlir::UnitAttr{});
}

void createGroupSignal(::TritonOpBuilder &builder,
                       std::vector<Value> &operands) {
  requireArity(operands, 7, 8, "aboleth.group_signal");
  Value after = operands.size() == 8 ? operands[6] : Value{};
  Value group = operands.back();
  auto *context = builder.getContext();
  Value endpoint = bindEndpoint(builder, operands[1]);
  Value destination = builder.create<BindRemoteRegionOp>(
      RemoteRegionType::get(context), operands[2], operands[5]);
  Value source = signalSource(builder, operands[3], operands[4]);
  operands[0] = builder.create<SignalOp>(
      TokenType::get(context), endpoint, destination, source, after, group,
      /*weakPublication=*/mlir::UnitAttr{},
      /*fencedSignal=*/mlir::UnitAttr{}, /*completedPrefix=*/mlir::UnitAttr{});
}

void createWait(::TritonOpBuilder &builder, std::vector<Value> &operands) {
  requireArity(operands, 2, 2, "aboleth.wait");
  builder.create<WaitOp>(operands[1], Value{}, /*selfReclaim=*/mlir::UnitAttr{});
}

void createWaitIf(::TritonOpBuilder &builder,
                  std::vector<Value> &operands) {
  requireArity(operands, 3, 3, "aboleth.wait_if");
  // The kernel's own predicate: nothing bounds what it defers by the queue.
  builder.create<WaitOp>(operands[1], operands[2],
                         /*selfReclaim=*/builder.getBuilder().getUnitAttr());
}

void createPlannedWait(::TritonOpBuilder &builder,
                       std::vector<Value> &operands) {
  requireArity(operands, 4, 4, "aboleth.planned_wait");
  builder.create<PlannedWaitOp>(operands[1], operands[2], operands[3],
                                /*reuseDistance=*/mlir::IntegerAttr{},
                                /*completionInterval=*/mlir::IntegerAttr{},
                                /*queueCapacity=*/mlir::IntegerAttr{});
}

// `wait(t, iteration=, reuse=)`: the reuse distance is a compile-time constant
// the frontend materialized as one.
void createReuseWait(::TritonOpBuilder &builder,
                     std::vector<Value> &operands) {
  requireArity(operands, 4, 4, "aboleth.reuse_wait");
  auto constant = operands[3].getDefiningOp<mlir::arith::ConstantOp>();
  auto distance =
      constant ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())
               : mlir::IntegerAttr{};
  if (!distance)
    llvm::report_fatal_error("aboleth.wait: reuse must be a constexpr");
  builder.create<PlannedWaitOp>(
      operands[1], operands[2], /*iterations=*/Value{},
      builder.getBuilder().getI64IntegerAttr(distance.getInt()),
      /*completionInterval=*/mlir::IntegerAttr{},
      /*queueCapacity=*/mlir::IntegerAttr{});
}

void createInvokeTransportPhase(::TritonOpBuilder &builder,
                                std::vector<Value> &operands) {
  requireArity(operands, 3, std::numeric_limits<size_t>::max(),
               "aboleth.invoke_transport_phase");
  builder.create<InvokeTransportPhaseOp>(
      operands[1], mlir::ValueRange(operands).drop_front(2));
}

void createWaitGroup(::TritonOpBuilder &builder,
                     std::vector<Value> &operands) {
  requireArity(operands, 2, 2, "aboleth.wait_group");
  builder.create<WaitGroupOp>(operands[1]);
}

void createDrain(::TritonOpBuilder &builder,
                 std::vector<Value> &operands) {
  requireArity(operands, 2, 2, "aboleth.drain");
  auto *context = builder.getContext();
  Value endpoint = bindEndpoint(builder, operands[1]);
  builder.create<DrainOp>(endpoint,
                         builder.getBuilder().getStringAttr("cta"),
                         /*exclusiveIssuer=*/mlir::UnitAttr{});
}

void createSystemFence(
    ::TritonOpBuilder &builder,
    std::vector<Value> &operands) {
  requireArity(
      operands, 2, 2,
      "aboleth.system_fence");
  builder.create<SystemFenceOp>(operands[1]);
}

void createWaitSignal(::TritonOpBuilder &builder,
                      std::vector<Value> &operands) {
  requireArity(operands, 3, 3, "aboleth.wait_signal");
  auto *context = builder.getContext();
  auto &opBuilder = builder.getBuilder();
  Value zero = builder.create<mlir::arith::ConstantOp>(
      opBuilder.getI32Type(), opBuilder.getI32IntegerAttr(0));
  Value signal = builder.create<BindLocalRegionOp>(
      LocalRegionType::get(context), operands[1], zero);
  builder.create<WaitSignalOp>(signal, operands[2]);
}

} // namespace

TRITON_PLUGIN_API mlir::triton::plugin::PluginInfo *tritonGetPluginInfo() {
  using namespace mlir::triton;

  static plugin::PassInfo passes[] = {
      {"aboleth_simplify_communication_dataflow", kPluginVersion,
       addSimplifyDataflow,
       registerPass<createAbolethSimplifyCommunicationDataflowPass>},
      {"aboleth_forward_inline_payloads", kPluginVersion,
       addForwardInlinePayloads,
       registerPass<createAbolethForwardInlinePayloadsPass>},
      {"aboleth_form_batches", kPluginVersion,
       addPass<createAbolethFormBatchesPass>,
       registerPass<createAbolethFormBatchesPass>},
      {"aboleth_order_producers", kPluginVersion,
       addPass<createAbolethOrderProducersPass>,
       registerPass<createAbolethOrderProducersPass>},
      {"aboleth_analyze_issuers", kPluginVersion,
       addPass<createAbolethAnalyzeIssuersPass>,
       registerPass<createAbolethAnalyzeIssuersPass>},
      {"aboleth_plan_transport", kPluginVersion, addPlanTransport,
       registerPass<createAbolethPlanTransportPass>},
      {"aboleth_plan_storage", kPluginVersion, addPlanStorage,
       registerPass<createAbolethPlanStoragePass>},
      {"aboleth_plan_endpoints", kPluginVersion, addPlanEndpoints,
       registerPass<createAbolethPlanEndpointsPass>},
      {"aboleth_plan_shapes", kPluginVersion, addPlanShapes,
       registerPass<createAbolethPlanShapesPass>},
      {"aboleth_coalesce_submissions", kPluginVersion,
       addCoalesceSubmissions,
       registerPass<createAbolethCoalesceSubmissionsPass>},
      {"aboleth_defer_announcement", kPluginVersion,
       addDeferAnnouncement,
       registerPass<createAbolethDeferAnnouncementPass>},
      {"aboleth_order_publications", kPluginVersion,
       addOrderPublications,
       registerPass<createAbolethOrderPublicationsPass>},
      {"aboleth_elide_epochs", kPluginVersion, addElideEpochs,
       registerPass<createAbolethElideEpochsPass>},
      {"aboleth_plan_publication", kPluginVersion,
       addPlanPublication,
       registerPass<createAbolethPlanPublicationPass>},
      {"aboleth_materialize_transport_phase", kPluginVersion,
       addMaterializeTransportPhase,
       registerPass<createAbolethMaterializeTransportPhasePass>},
      {"aboleth_materialize_completion", kPluginVersion,
       addMaterializeCompletion,
       registerPass<createAbolethMaterializeCompletionPass>},
      {"aboleth_materialize_valid_extents", kPluginVersion,
       addMaterializeValidExtents,
       registerPass<createAbolethMaterializeValidExtentsPass>},
      {"aboleth_lower_p2p", kPluginVersion, addLowerP2P,
       registerPass<createAbolethLowerP2PPass>},
      {"aboleth_lower_proxy", kPluginVersion,
       addLowerProxy,
       registerPass<createAbolethLowerProxyPass>},
      {"aboleth_lower_rdma", kPluginVersion, addLowerRDMA,
       registerPass<createAbolethLowerRDMAPass>},
      {"aboleth_lower_ionic", kPluginVersion, addLowerIonic,
       registerPass<createAbolethLowerIonicPass>},
      {"aboleth_lower_efa", kPluginVersion, addLowerEFA,
       registerPass<createAbolethLowerEFAPass>},
  };
  static plugin::DialectInfo dialects[] = {
      {"AbolethDialect", kPluginVersion, registerDialect},
  };
  static plugin::OpInfo operations[] = {
      {"aboleth_endpoint", createEndpoint},
      {"aboleth_begin_group", createBeginGroup},
      {"aboleth_put", createPut},
      {"aboleth_valid_extent", createValidExtent},
      {"aboleth_valid_region", createValidRegion},
      {"aboleth_put_inline", createPutInline},
      {"aboleth_signal", createSignal},
      {"aboleth_group_signal", createGroupSignal},
      {"aboleth_wait", createWait},
      {"aboleth_wait_if", createWaitIf},
      {"aboleth_planned_wait", createPlannedWait},
      {"aboleth_reuse_wait", createReuseWait},
      {"aboleth_invoke_transport_phase", createInvokeTransportPhase},
      {"aboleth_wait_group", createWaitGroup},
      {"aboleth_drain", createDrain},
      {"aboleth_system_fence", createSystemFence},
      {"aboleth_wait_signal", createWaitSignal},
  };
  static plugin::PluginInfo info = {
      TRITON_PLUGIN_API_VERSION,
      kPluginName,
      kPluginVersion,
      passes,
      std::size(passes),
      dialects,
      std::size(dialects),
      operations,
      std::size(operations),
      TRITON_VERSION,
  };
  return &info;
}
