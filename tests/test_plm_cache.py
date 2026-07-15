"""Stdlib-only tests for the features.plm disk-cache layer + missing-lib guard."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from features import plm

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_ABLANG2 = importlib.util.find_spec("ablang2") is not None


class PlmCacheTest(unittest.TestCase):
    def setUp(self):
        self._orig_root = plm.CACHE_ROOT
        self._tmp = tempfile.mkdtemp()
        plm.CACHE_ROOT = Path(self._tmp)

    def tearDown(self):
        plm.CACHE_ROOT = self._orig_root
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_save_load_roundtrip(self):
        vec = [1.0, -2.5, 3.25, 0.0]
        plm.save_cached(plm.MODEL_TAG_ESM2, "ABC", vec)
        self.assertEqual(plm.load_cached(plm.MODEL_TAG_ESM2, "ABC"), vec)

    def test_load_miss_returns_none(self):
        self.assertIsNone(plm.load_cached(plm.MODEL_TAG_ESM2, "NOPE"))

    def test_cache_path_deterministic_and_distinct(self):
        p = plm.cache_path(plm.MODEL_TAG_ESM2, "ABC")
        self.assertEqual(p, plm.cache_path(plm.MODEL_TAG_ESM2, "ABC"))
        self.assertNotEqual(p, plm.cache_path(plm.MODEL_TAG_ABLANG2, "ABC"))
        self.assertNotEqual(p, plm.cache_path(plm.MODEL_TAG_ESM2, "XYZ"))

    def test_embed_combined_concatenates_and_guards(self):
        seq = "SEQ"
        esm = [1.0, 2.0, 3.0]
        ab = [4.0, 5.0]
        plm.save_cached(plm.MODEL_TAG_ESM2, seq, esm)
        plm.save_cached(plm.MODEL_TAG_ABLANG2, seq, ab)
        combined = plm.embed_combined(seq)
        self.assertEqual(combined, esm + ab)
        self.assertEqual(len(combined), len(esm) + len(ab))

        # Missing one tag -> RuntimeError.
        plm.save_cached(plm.MODEL_TAG_ESM2, "ONLY_ESM", esm)
        with self.assertRaises(RuntimeError):
            plm.embed_combined("ONLY_ESM")

    @unittest.skipIf(_HAS_TORCH, "torch installed; real runner would not hit the guard")
    def test_embed_esm2_miss_raises_naming_package(self):
        with self.assertRaises(RuntimeError) as ctx:
            plm.embed_esm2(["MISSSEQ"])
        self.assertIn("torch", str(ctx.exception))

    @unittest.skipIf(_HAS_ABLANG2, "ablang2 installed; real runner would not hit the guard")
    def test_embed_ablang2_miss_raises_naming_package(self):
        with self.assertRaises(RuntimeError) as ctx:
            plm.embed_ablang2(["MISSSEQ"])
        self.assertIn("ablang2", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
