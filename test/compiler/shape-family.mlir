module {
  aboleth.shape_family {name = "uniform", minimumBound = 1 : i64,
    maximumBound = 513 : i64, slots = 2 : i64, uniformSelector = true}
  aboleth.shape_family {name = "unknown-uniformity", minimumBound = 1 : i64,
    maximumBound = 513 : i64, slots = 2 : i64, uniformSelector = false}
}
// CHECK: controlDomains = 22 : i64
// CHECK-SAME: variantBounds = array<i64: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024>
// CHECK: controlDomains = 2 : i64
// CHECK-SAME: variantBounds = array<i64: 1024>
// GENERIC: controlDomains = 2 : i64
// GENERIC-SAME: variantBounds = array<i64: 1024>
// GENERIC: controlDomains = 2 : i64
// GENERIC-SAME: variantBounds = array<i64: 1024>
