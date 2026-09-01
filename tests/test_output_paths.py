import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from src.processor import (
    _prepare_analysis_output_paths,
    _prune_old_result_runs,
    _runtime_directory,
)


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
                paths.run_folder,
                base / "결과" / "설문결과_2026.08.18.19.07.23",
            )
            self.assertEqual(paths.excel_path.name, "설문결과.xlsx")
            self.assertEqual(paths.comment_path.name, "자유기입.pdf")
            self.assertEqual(paths.review_folder.name, "검토용")
            self.assertTrue(paths.result_folder.is_dir())
            self.assertTrue(paths.run_folder.is_dir())
            self.assertTrue(paths.review_folder.is_dir())
            self.assertEqual(paths.excel_path.parent, paths.run_folder)
            self.assertEqual(paths.comment_path.parent, paths.run_folder)

    def test_repeated_timestamp_uses_a_collision_safe_suffix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            now = datetime(2026, 8, 18, 19, 7, 23)
            first = _prepare_analysis_output_paths(temp_dir, now)
            second = _prepare_analysis_output_paths(temp_dir, now)

            self.assertNotEqual(first.run_folder, second.run_folder)
            self.assertEqual(
                second.run_folder.name,
                "설문결과_2026.08.18.19.07.23_2",
            )
            self.assertTrue(second.review_folder.is_dir())

    def test_only_the_latest_thirty_complete_runs_are_kept(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            start = datetime(2026, 1, 1, 9, 0, 0)
            run_names = []
            for offset in range(32):
                paths = _prepare_analysis_output_paths(
                    temp_dir,
                    start + timedelta(seconds=offset),
                )
                run_names.append(paths.run_folder.name)
                paths.excel_path.write_bytes(f"xlsx-{offset}".encode())
                (paths.review_folder / "검토.pdf").write_bytes(
                    f"pdf-{offset}".encode()
                )

            result_folder = Path(temp_dir) / "결과"
            deleted, errors = _prune_old_result_runs(result_folder)

            self.assertEqual(errors, [])
            self.assertEqual(
                {path.name for path in deleted},
                set(run_names[:2]),
            )
            remaining_folders = {
                path.name for path in result_folder.iterdir() if path.is_dir()
            }
            self.assertEqual(remaining_folders, set(run_names[2:]))
            self.assertEqual(list(result_folder.glob("*.zip")), [])

    def test_deletion_failure_preserves_the_source_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _prepare_analysis_output_paths(
                temp_dir,
                datetime(2026, 1, 1, 9, 0, 0),
            )
            paths.excel_path.write_bytes(b"xlsx")
            (paths.review_folder / "검토.pdf").write_bytes(b"pdf")

            with patch("src.processor.shutil.rmtree", side_effect=OSError("busy")):
                deleted, errors = _prune_old_result_runs(
                    paths.result_folder,
                    keep_count=0,
                )

            self.assertEqual(deleted, [])
            self.assertEqual(len(errors), 1)
            self.assertTrue(paths.run_folder.is_dir())

    def test_legacy_incomplete_and_unrelated_items_are_not_deleted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_folder = Path(temp_dir) / "결과"
            legacy_folder = result_folder / "설문결과_2026.01.01.09.00.00_검토용"
            legacy_folder.mkdir(parents=True)
            legacy_file = result_folder / "설문결과_2026.01.01.09.00.00.xlsx"
            legacy_file.write_bytes(b"legacy")
            unrelated_zip = result_folder / "설문결과_2025.01.01.00.00.00.zip"
            unrelated_zip.write_bytes(b"existing")
            incomplete = result_folder / "설문결과_2026.01.02.09.00.00"
            (incomplete / "검토용").mkdir(parents=True)

            deleted, errors = _prune_old_result_runs(
                result_folder,
                keep_count=0,
            )

            self.assertEqual(deleted, [])
            self.assertEqual(errors, [])
            self.assertTrue(legacy_folder.is_dir())
            self.assertEqual(legacy_file.read_bytes(), b"legacy")
            self.assertTrue(incomplete.is_dir())
            self.assertEqual(unrelated_zip.read_bytes(), b"existing")

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
