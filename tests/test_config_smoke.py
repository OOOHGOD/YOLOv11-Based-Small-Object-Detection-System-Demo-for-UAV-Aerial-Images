"""Dependency-free smoke checks for the GUI detection configuration."""

from pathlib import Path
import runpy
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "UAV Aerial Perspective Small Target Detection System" / "Config.py"


class ConfigSmokeTest(unittest.TestCase):
    def test_detection_class_configuration_is_complete_and_portable(self):
        """Catch missing class labels or a model path tied to one machine."""
        config = runpy.run_path(CONFIG_PATH)

        self.assertEqual(set(config["names"]), set(range(10)))
        self.assertEqual(len(config["CH_names"]), 10)
        self.assertFalse(Path(config["model_path"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
