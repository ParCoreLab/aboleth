"""No-GPU regression for KV copy submission on HIP and CUDA."""
import ctypes
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[2] / 'python/aboleth/kv/copy.py'
spec = importlib.util.spec_from_file_location('_kv_copy_runtime_test', SOURCE)
copy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = copy
spec.loader.exec_module(copy)


class Function:
    def __init__(self, status=0):
        self.calls = []
        self.status = status

    def __call__(self, *args):
        self.calls.append(args)
        return self.status


class CopyRuntimeTest(unittest.TestCase):
    def tearDown(self):
        copy._copy_api = None

    def check_submission(self, hip):
        copy._copy_api = None
        function = Function()
        name = 'hipMemcpyAsync' if hip else 'cudaMemcpyAsync'
        library = SimpleNamespace(**{name: function})
        expected = 'libamdhip64.so.7' if hip else 'libcudart.so'
        with patch('torch.version.hip', '7.14' if hip else None), patch.object(
            copy.ctypes, 'CDLL', return_value=library
        ) as load:
            # Addresses, size and stream all exceed32 bits. No real allocation,
            # transfer, device selection or synchronization is performed.
            copy.cuda_memcpy_async(1 << 40, 2 << 40, 3 << 40, 4 << 40)
            copy.cuda_memcpy_async(5 << 40, 6 << 40, 7 << 40, 8 << 40)
        load.assert_called_once_with(expected)
        self.assertEqual(len(function.calls), 2)
        args = function.calls[0]
        self.assertEqual([a.value for a in args[:3]], [1 << 40, 2 << 40, 3 << 40])
        self.assertEqual(args[3], 3)
        self.assertEqual(args[4].value, 4 << 40)
        self.assertIs(type(args[0]), ctypes.c_void_p)
        self.assertIs(type(args[2]), ctypes.c_size_t)

    def test_hip_submission(self):
        self.check_submission(True)

    def test_cuda_submission(self):
        self.check_submission(False)

    def test_runtime_error_keeps_backend_status(self):
        for hip, name, label in [(True, 'hipMemcpyAsync', 'HIP'),
                                 (False, 'cudaMemcpyAsync', 'CUDA')]:
            with self.subTest(backend=label):
                copy._copy_api = None
                library = SimpleNamespace(**{name: Function(status=17)})
                with patch('torch.version.hip', '7.14' if hip else None), patch.object(
                    copy.ctypes, 'CDLL', return_value=library
                ), self.assertRaisesRegex(RuntimeError, name + ' failed with ' + label + ' status 17'):
                    copy.cuda_memcpy_async(8, 16, 32, 64)


if __name__ == '__main__':
    unittest.main()
