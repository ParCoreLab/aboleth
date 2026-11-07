// RUN: not aboleth-opt %s 2>&1 | FileCheck %s

module {
  aboleth.communication_dataflow "workset-invalid" {
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
      visibility = "peer", immutable = false, escapes = false
    } : !aboleth.logical_region
    %rows = aboleth.region {
      name = "rows", elementType = i32,
      shape = array<i64: 8>, strides = array<i64: 1>,
      indexMap = affine_map<(d0) -> (d0)>,
      visibility = "local", immutable = true, escapes = false
    } : !aboleth.logical_region
    %work = "aboleth.workset"(%rows) <{
      id = "work.declaration", name = "work",
      fieldNames = ["destination_row"],
      fieldLowerBounds = array<i64: 0>,
      fieldUpperBounds = array<i64: 8>,
      maxItems = 8 : i64,
      uniqueDestinations = true,
      ordered = false,
      operandSegmentSizes = array<i32: 1, 0>
    }> : (!aboleth.logical_region) -> !aboleth.workset
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
    ub.unreachable
  }
}

// CHECK: error: 'aboleth.transfer_each' op references unknown required workset field 'source_row'
