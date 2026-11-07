import unittest
from dataclasses import dataclass

from aboleth.planning import CommunicationDataflow


@dataclass(frozen=True)
class TritonLogicalRegion:
    name: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    visibility: str
    escapes: bool = False

    def declare(self, flow: CommunicationDataflow) -> str:
        return flow.region(
            self.name,
            element_type="bf16",
            shape=self.shape,
            strides=self.strides,
            visibility=self.visibility,
            escapes=self.escapes,
            registered=(
                self.visibility == "peer" and
                self.escapes
            ),
            donated=(
                self.visibility == "peer" and
                self.escapes
            ),
            reuse_interval=1 if self.escapes else 0,
            pool_depth=1 if self.escapes else 0,
        )


def build_triton_style_dataflow(
    *,
    cardinality: int,
    final_visibility: str,
):
    flow = CommunicationDataflow("triton-logical-tensors")
    source = TritonLogicalRegion(
        "triton.source", (32, 128), (128, 1), "local"
    ).declare(flow)
    staging = TritonLogicalRegion(
        "triton.staging", (32, 128), (128, 1), "peer"
    ).declare(flow)
    final = TritonLogicalRegion(
        "triton.final",
        (32, 128),
        (128, 1),
        final_visibility,
        escapes=True,
    ).declare(flow)
    produced = (
        flow.transfer(
            "triton.producer",
            source,
            staging,
            output="triton.produced",
        )
        if cardinality == 0
        else flow.reduction(
            "triton.producer",
            source,
            staging,
            cardinality=cardinality,
            operator="add",
            output="triton.produced",
        )
    )
    output = flow.copy(
        "triton.copy",
        produced,
        final,
        output="triton.output",
    )
    flow.use("triton.consumer", output)
    return flow.simplify()


class TritonDataflowFrontendTest(unittest.TestCase):
    def test_uses_generic_destination_rewrite(self):
        direct = build_triton_style_dataflow(
            cardinality=0,
            final_visibility="peer",
        )
        self.assertEqual(
            direct.operation("triton.producer").destination,
            "triton.final",
        )
        self.assertIsNone(direct.operation("triton.copy"))

        singleton = build_triton_style_dataflow(
            cardinality=1,
            final_visibility="peer",
        )
        self.assertEqual(
            singleton.operation("triton.producer").kind,
            "transfer",
        )
        self.assertIsNone(singleton.operation("triton.copy"))

        staged = build_triton_style_dataflow(
            cardinality=2,
            final_visibility="local",
        )
        self.assertEqual(
            staged.operation("triton.producer").kind,
            "reduction",
        )
        self.assertIsNotNone(staged.operation("triton.copy"))


if __name__ == "__main__":
    unittest.main()
