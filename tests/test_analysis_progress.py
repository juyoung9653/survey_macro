import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
import numpy as np

from src.models import TemplatePreset
from src.processor import _file_key, run_analysis


class _ResourceControllerStub:
    def start(self):
        pass

    def close(self):
        pass

    def checkpoint(self, required_memory_bytes=0, stage="", status_cb=None):
        pass


class AnalysisProgressTests(unittest.TestCase):
    def test_single_file_reports_each_template_sample_and_survey(self):
        progress_events: list[tuple[float, str]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "survey.pdf"
            document = fitz.open()
            for _ in range(3):
                document.new_page(width=80, height=60)
            document.save(pdf_path)
            document.close()

            samples = {0: [b"sample"] * 3}
            template = {0: np.full((8, 8), 220, np.uint8)}

            def collect_samples(fpath, *_args, progress_cb=None, **_kwargs):
                for done in range(1, 4):
                    progress_cb(done, 3)
                return _file_key(fpath), samples

            def analyze_file(
                _fpath, file_label, *_args, progress_cb=None, **_kwargs
            ):
                for done in range(1, 4):
                    progress_cb(done, 3)
                return file_label, [], []

            previous_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                with (
                    patch(
                        "src.processor._collect_template_samples",
                        side_effect=collect_samples,
                    ),
                    patch(
                        "src.processor.generate_dynamic_templates",
                        return_value=template,
                    ),
                    patch(
                        "src.processor._analyze_single_file",
                        side_effect=analyze_file,
                    ),
                    patch("src.processor._insert_img_into_pdf"),
                    patch("src.processor.export_to_excel", return_value=True),
                ):
                    success = run_analysis(
                        [str(pdf_path)],
                        [np.full((8, 8), 255, np.uint8)],
                        TemplatePreset(page_count=1),
                        progress_cb=lambda value, message: progress_events.append(
                            (float(value), message)
                        ),
                        resource_controller=_ResourceControllerStub(),
                    )
            finally:
                os.chdir(previous_cwd)

        values = [value for value, _message in progress_events]
        messages = [message for _value, message in progress_events]
        self.assertTrue(success)
        self.assertEqual(values[-1], 100.0)
        self.assertTrue(all(left <= right for left, right in zip(values, values[1:])))
        self.assertGreaterEqual(len(set(values)), 10)
        self.assertTrue(any("템플릿 표본 처리 중 (1/3)" in msg for msg in messages))
        self.assertTrue(any("설문 분석 중 (1/3)" in msg for msg in messages))


if __name__ == "__main__":
    unittest.main()
