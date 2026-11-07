"""configure_plugin's registration check, after triton._C.libtriton is loaded
(importing aboleth loads it): a plugin reached through a symlink is the
registered plugin, not a second one."""

import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from aboleth import _plugin


class ConfigurePluginTest(unittest.TestCase):
    def setUp(self):
        self.assertIn("triton._C.libtriton", sys.modules)
        root = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        (root / "build" / "lib").mkdir(parents=True)
        self.real = root / "build" / "lib" / "libAbolethPlugin.so"
        self.real.touch()
        (root / "link").symlink_to(root / "build")
        self.linked = root / "link" / "lib" / "libAbolethPlugin.so"

    def configure(self, path, registered):
        environ = {"TRITON_PLUGIN_PATHS": ":".join(map(str, registered))}
        with mock.patch.dict(os.environ, environ), \
                mock.patch.object(_plugin, "plugin_path", return_value=path):
            return _plugin.configure_plugin(), os.environ["TRITON_PLUGIN_PATHS"]

    def test_a_symlinked_plugin_is_the_registered_one(self):
        path, registered = self.configure(self.linked, [self.real.resolve()])
        self.assertEqual(path, self.real.resolve())
        self.assertEqual(registered, str(self.real.resolve()))

    def test_an_unregistered_plugin_is_still_refused(self):
        other = self.real.with_name("other.so")
        other.touch()
        with self.assertRaises(ImportError):
            self.configure(self.linked, [other])


if __name__ == "__main__":
    unittest.main()
