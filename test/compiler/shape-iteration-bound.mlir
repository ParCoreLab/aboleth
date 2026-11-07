module {
  aboleth.shape_family {name = "iteration-with-initialization", minimumBound = 1 : i64,
    maximumBound = 4096 : i64, iterationBound = 8 : i64, slots = 2 : i64, uniformSelector = true}
  aboleth.shape_family {name = "overall-bound-is-stronger", minimumBound = 1 : i64,
    maximumBound = 8 : i64, iterationBound = 9223372036854775807 : i64,
    slots = 1 : i64, uniformSelector = true}
}
// CHECK: controlDomains = 10 : i64
// CHECK-SAME: variantBounds = array<i64: 1, 2, 4, 8, 4096>
// CHECK: controlDomains = 4 : i64
// CHECK-SAME: variantBounds = array<i64: 1, 2, 4, 8>
// GENERIC: controlDomains = 2 : i64
// GENERIC-SAME: variantBounds = array<i64: 4096>
// GENERIC: controlDomains = 1 : i64
// GENERIC-SAME: variantBounds = array<i64: 8>
