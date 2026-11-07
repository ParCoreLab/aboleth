// RUN: aboleth-opt %s --pass simplify-communication-dataflow | FileCheck %s

module {
  aboleth.communication_dataflow "workset" {
    %source = aboleth.region {
      name = "source", elementType = bf16,
      shape = array<i64: 8, 16>, strides = array<i64: 16, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "local", immutable = false, escapes = false
    } : !aboleth.logical_region
    %destination = aboleth.region {
      name = "destination", elementType = bf16,
      shape = array<i64: 8, 16>, strides = array<i64: 16, 1>,
      indexMap = affine_map<(d0, d1) -> (d0, d1)>,
      visibility = "peer", immutable = false, escapes = true,
      registered = true, donated = true,
      reuseInterval = 1 : i64, poolDepth = 2 : i64
    } : !aboleth.logical_region
    %sourceRows = aboleth.region {
      name = "work.source_rows", elementType = i32,
      shape = array<i64: 8>, strides = array<i64: 1>,
      indexMap = affine_map<(d0) -> (d0)>,
      visibility = "local", immutable = true, escapes = false
    } : !aboleth.logical_region
    %peers = aboleth.region {
      name = "work.peers", elementType = i32,
      shape = array<i64: 8>, strides = array<i64: 1>,
      indexMap = affine_map<(d0) -> (d0)>,
      visibility = "local", immutable = true, escapes = false
    } : !aboleth.logical_region
    %destinationRows = aboleth.region {
      name = "work.destination_rows", elementType = i32,
      shape = array<i64: 8>, strides = array<i64: 1>,
      indexMap = affine_map<(d0) -> (d0)>,
      visibility = "local", immutable = true, escapes = false
    } : !aboleth.logical_region
    %work = "aboleth.workset"(
      %sourceRows, %peers, %destinationRows
    ) <{
      id = "work.declaration",
      name = "work",
      fieldNames = [
        "source_row", "destination_peer", "destination_row"
      ],
      fieldLowerBounds = array<i64: 0, 0, 0>,
      fieldUpperBounds = array<i64: 8, 2, 8>,
      maxItems = 8 : i64,
      uniqueDestinations = true,
      ordered = false,
      operandSegmentSizes = array<i32: 3, 0>
    }> : (
      !aboleth.logical_region,
      !aboleth.logical_region,
      !aboleth.logical_region
    ) -> !aboleth.workset
    %output = "aboleth.transfer_each"(
      %work, %source, %destination
    ) <{
      id = "send", outputId = "output",
      sourceIndexField = "source_row",
      destinationPeerField = "destination_peer",
      destinationIndexField = "destination_row"
    }> : (
      !aboleth.workset,
      !aboleth.logical_region,
      !aboleth.logical_region
    ) -> !aboleth.logical_region
    aboleth.use %output {
      name = "consumer", id = "consumer"
    } : !aboleth.logical_region
    ub.unreachable
  }
}

// CHECK: "aboleth.workset"
// CHECK-SAME: fieldLowerBounds = array<i64: 0, 0, 0>
// CHECK-SAME: fieldNames = ["source_row", "destination_peer", "destination_row"]
// CHECK: "aboleth.transfer_each"
// CHECK-SAME: destinationIndexField = "destination_row"
