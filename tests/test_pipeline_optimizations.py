import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import fitz
import numpy as np

from src.processor import (
    _build_file_labels,
    _build_file_templates,
    _decode_sampled_survey,
    _file_key,
    _load_ui_template_cache,
    _median_uint8_inplace,
    _save_ui_template_cache,
    run_analysis,
)
from src.models import TemplatePreset
from src.vision import load_pdf_pages


class PipelineOptimizationTests(unittest.TestCase):
    def test_uint8_median_matches_numpy(self):
        rng = np.random.default_rng(17)
        for count in (1, 2, 3, 4, 15, 31):
            with self.subTest(count=count):
                stack = rng.integers(0, 256, (count, 23, 19), dtype=np.uint8)
                expected = np.median(stack, axis=0).astype(np.uint8)

                actual = _median_uint8_inplace(stack.copy())

                self.assertTrue(np.array_equal(actual, expected))

    def test_selected_pdf_pages_preserve_order_and_duplicates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, "pages.pdf")
            document = fitz.open()
            for page_index in range(4):
                page = document.new_page(width=80, height=60)
                shade = (page_index + 1) / 5
                page.draw_rect(page.rect, fill=(shade, shade, shade), color=None)
            document.save(pdf_path)
            document.close()

            selected = load_pdf_pages(
                pdf_path,
                dpi=72,
                max_workers=2,
                gray=True,
                page_indices=[2, 99, 0, -1, 2],
            )
            empty = load_pdf_pages(pdf_path, dpi=72, gray=True, page_indices=[])

            self.assertEqual(len(selected), 3)
            self.assertTrue(np.array_equal(selected[0], selected[2]))
            self.assertFalse(np.array_equal(selected[0], selected[1]))
            self.assertEqual(empty, [])

    def test_phase_one_png_samples_can_be_reused(self):
        first = np.full((30, 20), 40, np.uint8)
        second = np.full((30, 20), 180, np.uint8)
        first_bytes = cv2.imencode(".png", first)[1].tobytes()
        second_bytes = cv2.imencode(".png", second)[1].tobytes()
        samples = {0: [first_bytes, second_bytes]}

        decoded = _decode_sampled_survey(samples, survey_idx=1, expected_pages=1)

        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertTrue(np.array_equal(decoded[0], second))
        self.assertIsNone(
            _decode_sampled_survey(samples, survey_idx=2, expected_pages=1)
        )

    def test_batch_templates_keep_their_existing_alignment(self):
        first_path = "first.pdf"
        second_path = "second.pdf"
        first = np.full((30, 20), 220, np.uint8)
        second = np.full((30, 20), 180, np.uint8)
        first_bytes = cv2.imencode(".png", first)[1].tobytes()
        second_bytes = cv2.imencode(".png", second)[1].tobytes()
        sample_results = {
            _file_key(first_path): {0: [first_bytes]},
            _file_key(second_path): {0: [second_bytes]},
        }

        with patch("src.processor.ImageAligner") as aligner_cls:
            reference, templates = _build_file_templates(
                [first_path, second_path], sample_results
            )

        aligner_cls.assert_not_called()
        self.assertIsNotNone(reference)
        assert reference is not None
        self.assertTrue(np.array_equal(reference[0], first))
        self.assertTrue(
            np.array_equal(templates[_file_key(second_path)][0], second)
        )

    def test_batch_analysis_finishes_each_file_before_collecting_the_next(self):
        file_paths = ["first.pdf", "second.pdf", "third.pdf"]
        sample_sets = {path: {0: [path.encode()]} for path in file_paths}
        sample_names = {id(samples): path for path, samples in sample_sets.items()}
        events: list[tuple[str, str]] = []
        exported_rows: list[dict] = []

        def collect_samples(fpath, *_args, **_kwargs):
            file_index = file_paths.index(fpath)
            if file_index > 0:
                self.assertEqual(sample_sets[file_paths[file_index - 1]], {})
            events.append(("collect", fpath))
            return _file_key(fpath), sample_sets[fpath]

        def build_template(samples):
            path = sample_names[id(samples)]
            events.append(("template", path))
            shade = 200 - file_paths.index(path) * 20
            return {0: np.full((8, 8), shade, np.uint8)}

        def analyze_file(fpath, file_label, *_args, sample_pages=None, **_kwargs):
            self.assertIs(sample_pages, sample_sets[fpath])
            events.append(("analyze", fpath))
            return file_label, [{"파일명": file_label, "페이지": "1p"}], []

        def export_rows(results, _config):
            exported_rows.extend(results)
            return True

        config = TemplatePreset(page_count=1)
        template_page = np.full((8, 8), 255, np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir:
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
                        side_effect=build_template,
                    ),
                    patch(
                        "src.processor._analyze_single_file",
                        side_effect=analyze_file,
                    ),
                    patch("src.processor._insert_img_into_pdf"),
                    patch("src.processor.export_to_excel", side_effect=export_rows),
                ):
                    success = run_analysis(file_paths, [template_page], config)
            finally:
                os.chdir(previous_cwd)

        self.assertTrue(success)
        self.assertEqual(
            events,
            [
                ("collect", "first.pdf"),
                ("template", "first.pdf"),
                ("analyze", "first.pdf"),
                ("collect", "second.pdf"),
                ("template", "second.pdf"),
                ("analyze", "second.pdf"),
                ("collect", "third.pdf"),
                ("template", "third.pdf"),
                ("analyze", "third.pdf"),
            ],
        )
        self.assertEqual(
            [row["파일명"] for row in exported_rows],
            ["first", "second", "third"],
        )

    def test_batch_analysis_defers_files_until_a_reference_template_exists(self):
        file_paths = ["broken.pdf", "valid.pdf"]
        valid_samples = {0: [b"valid"]}
        template = {0: np.full((8, 8), 210, np.uint8)}
        events: list[tuple[str, str, bool]] = []
        exported_rows: list[dict] = []

        def collect_samples(fpath, *_args, **_kwargs):
            if fpath == "broken.pdf":
                return _file_key(fpath), {}
            return _file_key(fpath), valid_samples

        def analyze_file(
            fpath,
            file_label,
            _config,
            file_template,
            reference_templates,
            *_args,
            sample_pages=None,
            **_kwargs,
        ):
            self.assertIs(file_template, template)
            self.assertIs(reference_templates, template)
            events.append(("analyze", fpath, sample_pages is None))
            return file_label, [{"파일명": file_label, "페이지": "1p"}], []

        def export_rows(results, _config):
            exported_rows.extend(results)
            return True

        with tempfile.TemporaryDirectory() as temp_dir:
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
                    patch("src.processor.export_to_excel", side_effect=export_rows),
                ):
                    success = run_analysis(
                        file_paths,
                        [np.full((8, 8), 255, np.uint8)],
                        TemplatePreset(page_count=1),
                    )
            finally:
                os.chdir(previous_cwd)

        self.assertTrue(success)
        self.assertEqual(
            events,
            [
                ("analyze", "broken.pdf", True),
                ("analyze", "valid.pdf", False),
            ],
        )
        self.assertEqual(
            [row["파일명"] for row in exported_rows], ["broken", "valid"]
        )

    def test_ui_template_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "templates.npz"
            templates = {0: np.arange(20, dtype=np.uint8).reshape(4, 5)}

            _save_ui_template_cache(cache_path, templates)
            loaded = _load_ui_template_cache(cache_path)

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertTrue(np.array_equal(loaded[0], templates[0]))

    def test_missing_png_keeps_survey_index_and_duplicate_stems_get_labels(self):
        samples = {0: [b"", cv2.imencode(".png", np.ones((4, 4), np.uint8))[1].tobytes()]}

        self.assertIsNone(
            _decode_sampled_survey(samples, survey_idx=0, expected_pages=1)
        )
        self.assertEqual(
            _build_file_labels(["a/result.pdf", "b/result.pdf", "c/other.pdf"]),
            ["result_1", "result_2", "other"],
        )
        collision_labels = _build_file_labels(
            ["a/result.pdf", "b/result.pdf", "c/result_1.pdf", "d/Result.pdf"]
        )
        self.assertEqual(len({os.path.normcase(v) for v in collision_labels}), 4)


if __name__ == "__main__":
    unittest.main()
