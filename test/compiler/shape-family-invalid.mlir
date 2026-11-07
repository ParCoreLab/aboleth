module {
  aboleth.shape_family {name = "overflow", minimumBound = 1 : i64,
    maximumBound = 4611686018427387905 : i64, slots = 2 : i64, uniformSelector = true}
}
// CHECK: error: 'aboleth.shape_family' op power-of-two bound rounding would overflow signed i64
