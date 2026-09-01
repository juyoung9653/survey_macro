import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from src.processor import _prepare_analysis_output_paths, _runtime_directory


class AnalysisOutputPathTests(unittest.TestCase):
    def test_output_paths_use_timestamp_under_result_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            paths = _prepare_analysis_output_paths(
                base,
                datetime(2026, 8, 18, 19, 7, 23),
            )

            self.assertEqual(paths.result_folder, base / "결과")
            self.assertEqual(
                paths.excel_path.name,
                "설문결과_2026.08.18.19.07.23.xlsx",
            )
            self.assertEqual(
                paths.comment_path.name,
                "설문결과_2026.08.18.19.07.23_자유기입.pdf",
            )
            self.assertEqual(
                paths.review_folder.name,
                "설문결과_2026.08.18.19.07.23_검토용",
            )
            self.assertTrue(paths.result_folder.is_dir())
            self.assertTrue(paths.review_folder.is_dir())

    def test_repeated_timestamp_uses_a_collision_safe_suffix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            now = datetime(2026, 8, 18, 19, 7, 23)
            first = _prepare_analysis_output_paths(temp_dir, now)
            second = _prepare_analysis_output_paths(temp_dir, now)

            self.assertNotEqual(first.review_folder, second.review_folder)
            self.assertEqual(
                second.excel_path.name,
                "설문결과_2026.08.18.19.07.23_2.xlsx",
            )
            self.assertTrue(second.review_folder.is_dir())

    def test_packaged_runtime_uses_executable_directory(self):
        executable = Path("C:/portable/survey/설문지스캔.exe")
        with (
            patch.object(sys, "frozen", True, create=True),
            patch.object(sys, "executable", str(executable)),
        ):
            runtime_directory = _runtime_directory()

        self.assertEqual(runtime_directory, executable.resolve().parent)


if __name__ == "__main__":
    unittest.main()
