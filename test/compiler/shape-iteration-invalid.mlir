module {
  aboleth.shape_family {name = "invalid-iteration", minimumBound = 4 : i64,
    maximumBound = 4096 : i64, iterationBound = 2 : i64, slots = 2 : i64, uniformSelector = true}
}
// CHECK: error: 'aboleth.shape_family' op iterationBound must be at least minimumBound
