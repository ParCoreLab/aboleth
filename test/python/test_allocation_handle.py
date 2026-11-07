import unittest

from aboleth.runtime import AllocationHandle, AllocationProvider


def verbs_handle(**overrides):
    fields = {
        "provider": AllocationProvider.IB_VERBS,
        "peer": 1,
        "device": 0,
        "generation": 3,
        "address": 0x10000,
        "allocation_length": 4096,
        "allocation_size": 4096,
        "mapping_offset": 0,
        "offset": 64,
        "length": 128,
        "rkey": 0x1234,
    }
    return AllocationHandle(**{**fields, **overrides})


class AllocationHandleWireTest(unittest.TestCase):
    def test_strict_ordering_round_trips(self):
        for strict in (False, True):
            with self.subTest(strict=strict):
                handle = verbs_handle(strictly_ordered=strict)
                self.assertEqual(
                    AllocationHandle.from_bytes(handle.to_bytes()), handle
                )

    def test_an_unflagged_handle_reads_as_relaxed(self):
        # Encoders that predate the flag zero the field it travels in, and an
        # unknown registration must not license a fenced write.
        encoded = verbs_handle().to_bytes()
        self.assertEqual(encoded[2:4], b"\0\0")
        self.assertFalse(
            AllocationHandle.from_bytes(encoded).strictly_ordered
        )


if __name__ == "__main__":
    unittest.main()
