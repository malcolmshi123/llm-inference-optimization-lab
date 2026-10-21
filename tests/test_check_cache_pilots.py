import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bench.cache_workloads import create_manifest
from bench.check_cache_pilots import load_pilots
from tests.test_cache_workloads import WordTokenizer
from tests.test_compare_cache import create_run


class CheckCachePilotsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        manifest = create_manifest(WordTokenizer(), count=2)
        create_run(self.root / "pilot-off", manifest, expected=False, session=0)
        create_run(self.root / "pilot-on", manifest, expected=True, session=1)

    def test_valid_paired_pilots_are_loaded(self):
        pilots = load_pilots(self.root)
        self.assertEqual(len(pilots), 2)
        self.assertTrue(all(len(rows) == 4 for _, _, rows in pilots))

    def test_completed_status_does_not_bypass_raw_snapshot_checks(self):
        (self.root / "pilot-on" / "shared-c1-after.prom").unlink()
        with self.assertRaises(OSError):
            load_pilots(self.root)

    def test_different_workloads_are_rejected(self):
        create_run(
            self.root / "other" / "pilot-off",
            create_manifest(WordTokenizer(), count=2),
            expected=False,
            session=0,
        )
        create_run(
            self.root / "other" / "pilot-on",
            create_manifest(WordTokenizer(), seed=43, count=2),
            expected=True,
            session=1,
        )
        with self.assertRaisesRegex(ValueError, "same workload"):
            load_pilots(self.root / "other")


if __name__ == "__main__":
    unittest.main()
