import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import fitz
import numpy as np

from src.processor import (
    _analyze_single_file,
    _build_file_labels,
    _build_file_templates,
    _build_vector_page,
    _collect_template_samples,
    _decode_sampled_survey,
    _encode_jpeg,
    _file_key,
    _load_ui_template_cache,
    _median_uint8_inplace,
    _prepare_checkbox_template_interiors,
    _replace_single_sample_templates,
    _sampled_survey_is_available,
    _save_ui_template_cache,
    _select_ui_detection_templates,
    _ui_template_sample_progress,
    _validate_analysis_page_geometry,
    _validate_analysis_template_layout,
    extract_ink_info_from_mask,
    extract_checkbox_ink_info,
    extract_pure_ink_mask,
    generate_dynamic_templates,
    run_analysis,
)
from src.models import Box, Field, TemplatePreset
from src.vision import load_pdf_pages


class _ResourceControllerStub:
    def __init__(self):
        self.started = False
        self.closed = False
        self.checkpoints = []

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def checkpoint(self, required_memory_bytes=0, stage="", status_cb=None):
        self.checkpoints.append((required_memory_bytes, stage))


class PipelineOptimizationTests(unittest.TestCase):
    @staticmethod
    def _make_detection_grid() -> np.ndarray:
        image = np.full((700, 900), 255, np.uint8)
        x0, y0, cell_w, cell_h = 120, 100, 100, 80
        for column in range(6):
            x = x0 + column * cell_w
            cv2.line(image, (x, y0), (x, y0 + 5 * cell_h), 0, 2)
        for row in range(6):
            y = y0 + row * cell_h
            cv2.line(image, (x0, y), (x0 + 5 * cell_w, y), 0, 2)
        return image

    def test_ui_detection_template_prefers_sample_with_more_valid_boxes(self):
        clean = self._make_detection_grid()
        broken = clean.copy()
        for row in (3, 4):
            center_y = 100 + row * 80 + 40
            cv2.rectangle(
                broken,
                (617, center_y - 3),
                (623, center_y + 3),
                255,
                -1,
            )
        encoded_ok, encoded = cv2.imencode(".png", clean)
        self.assertTrue(encoded_ok)

        selected = _select_ui_detection_templates(
            {0: [encoded.tobytes()]},
            {0: broken},
        )

        self.assertTrue(np.array_equal(selected[0], clean))

    def test_ui_detection_template_keeps_median_on_equal_score(self):
        median = self._make_detection_grid()
        encoded_ok, encoded = cv2.imencode(".png", median)
        self.assertTrue(encoded_ok)

        selected = _select_ui_detection_templates(
            {0: [encoded.tobytes()]},
            {0: median},
        )

        self.assertIs(selected[0], median)

    def test_ui_detection_samples_cover_the_end_of_a_merged_batch(self):
        clean = self._make_detection_grid()
        broken = clean.copy()
        cv2.rectangle(broken, (617, 210), (623, 230), 255, -1)
        broken_ok, broken_png = cv2.imencode(".png", broken)
        clean_ok, clean_png = cv2.imencode(".png", clean)
        self.assertTrue(broken_ok)
        self.assertTrue(clean_ok)
        samples = [broken_png.tobytes()] * 9 + [clean_png.tobytes()]

        selected = _select_ui_detection_templates(
            {0: samples},
            {0: broken},
        )

        self.assertTrue(np.array_equal(selected[0], clean))

    def test_prepared_checkbox_template_interior_matches_per_survey_refinement(self):
        template = np.full((180, 240), 255, np.uint8)
        box = Box(page_idx=0, x=70, y=60, w=28, h=28)
        cv2.rectangle(template, (70, 60), (98, 88), 0, 2)
        target = template.copy()
        cv2.line(target, (76, 77), (83, 84), 0, 3)
        cv2.line(target, (83, 84), (94, 68), 0, 3)
        field = Field(name="Q", boxes=[box])
        plans = [(field, [box], [box], True)]

        prepared = _prepare_checkbox_template_interiors(
            plans, {0: template}, {0: target.shape}
        )
        expected = extract_checkbox_ink_info(target, box, template)
        actual = extract_checkbox_ink_info(
            target, box, template_interior=next(iter(prepared.values()))
        )

        self.assertEqual(actual.ink_pixels, expected.ink_pixels)
        self.assertEqual(actual.area, expected.area)
        self.assertEqual(actual.mark_strength, expected.mark_strength)
        self.assertEqual(actual.residual_pixels, expected.residual_pixels)
        self.assertEqual(actual.mask_bounds, expected.mask_bounds)
        self.assertTrue(np.array_equal(actual.ink_mask, expected.ink_mask))

    def test_batched_vector_annotations_render_like_individual_page_calls(self):
        base_img = np.full((180, 240), 235, np.uint8)
        cv2.circle(base_img, (120, 90), 25, 80, 3)
        annotations = [
            (20, 35, 45, 28, "A", False),
            (120, 105, 55, 32, "B", True),
        ]

        expected_doc = fitz.open()
        expected_page = expected_doc.new_page(width=240, height=180)
        image_bytes = _encode_jpeg(base_img)
        assert image_bytes is not None
        expected_page.insert_image(expected_page.rect, stream=image_bytes)
        for bx, by, bw, bh, label, is_ticked in annotations:
            color = (0, 1, 0) if is_ticked else (1, 0, 0)
            expected_page.draw_rect(
                fitz.Rect(bx, by, bx + bw, by + bh), color=color, width=2
            )
            expected_page.insert_text(
                fitz.Point(bx, max(0, by - 5)),
                label,
                fontsize=8,
                color=color,
            )

        actual_doc = fitz.open()
        _build_vector_page(actual_doc, base_img, annotations)
        expected_pixmap = expected_doc[0].get_pixmap(dpi=144, alpha=False)
        actual_pixmap = actual_doc[0].get_pixmap(dpi=144, alpha=False)
        try:
            self.assertEqual(
                (actual_pixmap.width, actual_pixmap.height),
                (expected_pixmap.width, expected_pixmap.height),
            )
            self.assertEqual(actual_pixmap.samples, expected_pixmap.samples)
        finally:
            actual_doc.close()
            expected_doc.close()

    def test_analysis_reuses_debug_jpeg_for_comment_output(self):
        page_image = np.full((30, 20), 245, np.uint8)
        ink_image = np.full((30, 20), 255, np.uint8)
        encoded_page = cv2.imencode(".png", page_image)[1].tobytes()
        config = TemplatePreset(page_count=1)

        def process_survey(survey_data, *_args, **_kwargs):
            title = survey_data["row_title"]
            return (
                {"파일명": "sample", "페이지": title, "Q": "checked"},
                {0: page_image},
                {0: ink_image},
                {0: []},
                {0: []},
                {0: page_image},
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "sample.pdf"
            document = fitz.open()
            document.new_page(width=20, height=30)
            document.save(pdf_path)
            document.close()
            template = {0: np.full((30, 20), 255, np.uint8)}

            with (
                patch("src.processor._build_page_aligners", return_value=[object()]),
                patch("src.processor._remap_checkbox_layout", return_value=config),
                patch(
                    "src.processor._checkbox_layout_is_trustworthy",
                    return_value=True,
                ),
                patch("src.processor._prepare_field_plans", return_value=[]),
                patch("src.processor.process_survey_data", side_effect=process_survey),
                patch(
                    "src.processor._encode_jpeg", wraps=_encode_jpeg
                ) as encode_jpeg,
            ):
                _, rows, comments = _analyze_single_file(
                    str(pdf_path),
                    "sample",
                    config,
                    template,
                    template,
                    [template[0]],
                    Path(temp_dir),
                    sample_pages={0: [encoded_page]},
                )

        self.assertEqual(len(rows), 1)
        self.assertEqual(len(comments), 1)
        self.assertEqual(encode_jpeg.call_count, 2)

    def test_uint8_median_matches_numpy(self):
        rng = np.random.default_rng(17)
        for count in (1, 2, 3, 4, 15, 31):
            with self.subTest(count=count):
                stack = rng.integers(0, 256, (count, 23, 19), dtype=np.uint8)
                expected = np.median(stack, axis=0).astype(np.uint8)

                actual = _median_uint8_inplace(stack.copy())

                self.assertTrue(np.array_equal(actual, expected))

    def test_single_response_uses_clean_preset_instead_of_its_own_mark(self):
        blank = np.full((160, 240), 255, np.uint8)
        box = Box(page_idx=0, x=30, y=60, w=160, h=40)
        cv2.rectangle(blank, (30, 60), (190, 100), 0, 2)
        marked = blank.copy()
        cv2.line(marked, (80, 90), (120, 70), 0, 5)
        config = TemplatePreset(
            page_count=1,
            fields=[Field(name="Q", boxes=[box])],
        )
        templates = generate_dynamic_templates({0: [marked]}, config=config)

        absorbed_ink = extract_pure_ink_mask(marked, templates[0], 0.0)
        absorbed_score, _ = extract_ink_info_from_mask(absorbed_ink, box)
        replaced = _replace_single_sample_templates(
            templates,
            {0: [marked]},
            [blank],
        )
        recovered_ink = extract_pure_ink_mask(marked, templates[0], 0.0)
        recovered_score, _ = extract_ink_info_from_mask(recovered_ink, box)

        self.assertEqual(absorbed_score, 0)
        self.assertEqual(replaced, {0})
        self.assertGreater(recovered_score, 50)

    def test_multiple_responses_keep_the_file_specific_template(self):
        file_template = np.full((30, 20), 230, np.uint8)
        templates = {0: file_template}

        replaced = _replace_single_sample_templates(
            templates,
            {0: [b"first", b"second"]},
            [np.full_like(file_template, 255)],
        )

        self.assertEqual(replaced, set())
        self.assertIs(templates[0], file_template)

    def test_ui_template_progress_distinguishes_sample_from_full_pdf(self):
        events = []
        callback = _ui_template_sample_progress(
            lambda value, message: events.append((value, message)),
            sample_page_count=31,
            total_page_count=50,
        )
        self.assertIsNotNone(callback)
        assert callback is not None

        callback(0, "PDF 로딩 시작")
        callback(48, "PDF 로딩... (15/31)")
        callback(100, "PDF 로딩... (31/31)")

        self.assertEqual(
            events,
            [
                (
                    0,
                    "체크박스 위치 찾기용 페이지 준비 시작 "
                    "(전체 50쪽 중 31쪽 사용)",
                ),
                (
                    48,
                    "체크박스 위치 찾기용 페이지 읽는 중... "
                    "(15/31쪽 · 전체 50쪽)",
                ),
                (
                    100,
                    "체크박스 위치 찾기용 페이지 준비 완료 "
                    "(전체 50쪽 중 31쪽 사용)",
                ),
            ],
        )

    def test_response_aware_template_does_not_absorb_majority_mark(self):
        blank = np.full((160, 240), 255, np.uint8)
        box = Box(page_idx=0, x=30, y=60, w=160, h=40)
        cv2.rectangle(blank, (30, 60), (190, 100), 0, 2)
        marked = blank.copy()
        cv2.line(marked, (80, 90), (120, 70), 0, 5)
        samples = [marked.copy() for _ in range(16)] + [
            blank.copy() for _ in range(15)
        ]
        config = TemplatePreset(
            page_count=1,
            fields=[Field(name="Q", boxes=[box])],
        )

        ordinary = generate_dynamic_templates({0: [item.copy() for item in samples]})[0]
        cleaned = generate_dynamic_templates(
            {0: [item.copy() for item in samples]}, config=config
        )[0]
        ordinary_ink = extract_pure_ink_mask(marked, ordinary, 0.0)
        cleaned_ink = extract_pure_ink_mask(marked, cleaned, 0.0)
        ordinary_score, _ = extract_ink_info_from_mask(ordinary_ink, box)
        cleaned_score, _ = extract_ink_info_from_mask(cleaned_ink, box)

        self.assertEqual(ordinary_score, 0)
        self.assertGreater(cleaned_score, 50)

    def test_response_aware_template_leaves_comment_regions_on_median(self):
        blank = np.full((160, 240), 255, np.uint8)
        box = Box(page_idx=0, x=30, y=60, w=160, h=40)
        cv2.rectangle(blank, (30, 60), (190, 100), 0, 2)
        marked = blank.copy()
        cv2.line(marked, (80, 90), (120, 70), 0, 5)
        samples = [marked.copy() for _ in range(16)] + [
            blank.copy() for _ in range(15)
        ]
        config = TemplatePreset(
            page_count=1,
            fields=[Field(name="comment", boxes=[box], is_comment=True)],
        )

        ordinary = generate_dynamic_templates({0: [item.copy() for item in samples]})[0]
        cleaned = generate_dynamic_templates(
            {0: [item.copy() for item in samples]}, config=config
        )[0]

        self.assertTrue(np.array_equal(cleaned, ordinary))

    def test_response_aware_template_preserves_large_response_box_borders(self):
        box = Box(page_idx=0, x=30, y=60, w=160, h=40)
        samples = []
        for index in range(31):
            page = np.full((160, 240), 255, np.uint8)
            shift = (index % 5) - 2
            cv2.rectangle(page, (30 + shift, 60), (190 + shift, 100), 0, 2)
            if index < 16:
                cv2.line(page, (80 + shift, 90), (120 + shift, 70), 0, 5)
            samples.append(page)
        config = TemplatePreset(
            page_count=1,
            fields=[Field(name="Q", boxes=[box])],
        )

        cleaned = generate_dynamic_templates(
            {0: [item.copy() for item in samples]}, config=config
        )[0]
        blank_ink = extract_pure_ink_mask(samples[19], cleaned, 0.0)
        blank_score, _ = extract_ink_info_from_mask(blank_ink, box)

        self.assertEqual(blank_score, 0)

    def test_response_aware_template_preserves_internal_printed_grid_rules(self):
        box = Box(page_idx=0, x=30, y=50, w=170, h=70)
        samples = []
        for index in range(31):
            page = np.full((170, 240), 255, np.uint8)
            shift = (index % 5) - 2
            cv2.line(page, (95 + shift, 50), (95 + shift, 120), 0, 2)
            if index < 16:
                cv2.line(page, (140, 100), (175, 70), 0, 5)
            samples.append(page)
        config = TemplatePreset(
            page_count=1,
            fields=[Field(name="Q", boxes=[box])],
        )

        cleaned = generate_dynamic_templates(
            {0: [item.copy() for item in samples]}, config=config
        )[0]
        blank_ink = extract_pure_ink_mask(samples[19], cleaned, 0.0)
        blank_score, _ = extract_ink_info_from_mask(blank_ink, box)

        self.assertEqual(blank_score, 0)

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

    def test_raw_phase_one_sample_can_be_reused_without_decode(self):
        sample = np.full((30, 20), 125, np.uint8)
        samples = {0: [sample]}

        decoded = _decode_sampled_survey(
            samples, survey_idx=0, expected_pages=1
        )

        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertIs(decoded[0], sample)
        self.assertTrue(_sampled_survey_is_available(samples, 0, 1))

    def test_phase_one_uses_dynamic_page_pipeline_and_preserves_order(self):
        class ParallelControllerStub:
            cpu_count = 4

            def __init__(self):
                self.pending_tasks = []

            def parallel_checkpoint(
                self,
                _required_memory,
                pending_tasks,
                stage="",
                status_cb=None,
                coordinator_threads=0,
                coordinator_memory_bytes=0,
            ):
                self.pending_tasks.append(pending_tasks)
                return SimpleNamespace(worker_count=2, total_cpu_threads=3)

        active = 0
        max_active = 0
        active_lock = threading.Lock()

        class NewThreadExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers
                self.threads = []

            def submit(self, function, *args):
                future = Future()

                def run():
                    try:
                        future.set_result(function(*args))
                    except BaseException as error:
                        future.set_exception(error)

                thread = threading.Thread(target=run)
                thread.start()
                self.threads.append(thread)
                return future

            def shutdown(self, wait=True):
                if wait:
                    for thread in self.threads:
                        thread.join()

        class FakeAligner:
            def align(self, image):
                nonlocal active, max_active
                with active_lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.02)
                with active_lock:
                    active -= 1
                return image

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "samples.pdf"
            document = fitz.open()
            for shade in (0.2, 0.4, 0.6, 0.8):
                page = document.new_page(width=20, height=30)
                page.draw_rect(
                    page.rect, fill=(shade, shade, shade), color=None
                )
            document.save(pdf_path)
            document.close()

            controller = ParallelControllerStub()
            progress = []
            with (
                patch(
                    "src.processor._build_page_aligners",
                    side_effect=lambda *_args: [FakeAligner()],
                ) as build_aligners,
                patch("src.processor.ThreadPoolExecutor", NewThreadExecutor),
            ):
                _, samples = _collect_template_samples(
                    str(pdf_path),
                    TemplatePreset(page_count=1),
                    [np.full((30, 20), 255, np.uint8)],
                    dpi=72,
                    resource_controller=controller,
                    progress_cb=lambda done, total: progress.append(
                        (done, total)
                    ),
                )

        decoded_means = [
            float(
                cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
                .mean()
            )
            for data in samples[0]
        ]
        self.assertGreater(max_active, 1)
        self.assertEqual(build_aligners.call_count, 2)
        self.assertEqual(controller.pending_tasks, [4])
        self.assertEqual(progress, [(1, 4), (2, 4), (3, 4), (4, 4)])
        self.assertEqual(decoded_means, sorted(decoded_means))
        self.assertTrue(all(isinstance(data, bytes) for data in samples[0]))

    def test_phase_one_retains_raw_samples_when_memory_headroom_allows(self):
        class HighMemoryController:
            cpu_count = 4

            def parallel_checkpoint(self, *_args, pending_tasks, **_kwargs):
                return SimpleNamespace(
                    worker_count=min(2, pending_tasks),
                    total_cpu_threads=3,
                    available_memory_bytes=2**40,
                    reserve_memory_bytes=0,
                    safety_memory_bytes=0,
                )

        class IdentityAligner:
            def align(self, image):
                return image

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "raw-samples.pdf"
            document = fitz.open()
            for shade in (0.3, 0.7):
                page = document.new_page(width=20, height=30)
                page.draw_rect(
                    page.rect, fill=(shade, shade, shade), color=None
                )
            document.save(pdf_path)
            document.close()

            with patch(
                "src.processor._build_page_aligners",
                return_value=[IdentityAligner()],
            ):
                _, samples = _collect_template_samples(
                    str(pdf_path),
                    TemplatePreset(page_count=1),
                    [np.full((30, 20), 255, np.uint8)],
                    dpi=72,
                    resource_controller=HighMemoryController(),
                )

        self.assertEqual(len(samples[0]), 2)
        self.assertTrue(
            all(isinstance(sample, np.ndarray) for sample in samples[0])
        )

    def test_sample_availability_requires_every_expected_page(self):
        samples = {0: [b"first", b"second"], 1: [b"first"]}

        self.assertTrue(_sampled_survey_is_available(samples, 0, 2))
        self.assertFalse(_sampled_survey_is_available(samples, 1, 2))
        self.assertFalse(_sampled_survey_is_available(samples, 0, 0))

    def test_file_analysis_uses_dynamic_plan_and_preserves_survey_order(self):
        class ParallelControllerStub:
            cpu_count = 8

            def __init__(self):
                self.pending_tasks = []
                self.checkpoints = []

            def parallel_checkpoint(
                self,
                _required_memory,
                pending_tasks,
                stage="",
                status_cb=None,
                coordinator_threads=0,
            ):
                self.pending_tasks.append(pending_tasks)
                return SimpleNamespace(
                    worker_count=min(2, pending_tasks),
                    total_cpu_threads=3,
                )

            def checkpoint(
                self, required_memory_bytes=0, stage="", status_cb=None
            ):
                self.checkpoints.append((required_memory_bytes, stage))

        active = 0
        max_active = 0
        active_lock = threading.Lock()
        third_started = threading.Event()
        analysis_overlapped_output = False
        page_image = np.full((30, 20), 255, np.uint8)

        def process_survey(survey_data, *_args, **_kwargs):
            nonlocal active, max_active
            if survey_data["row_title"] == "sample_3p":
                third_started.set()
            with active_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with active_lock:
                active -= 1
            title = survey_data["row_title"]
            return (
                {"파일명": "sample", "페이지": title, "Q": title},
                {0: page_image},
                {},
                {0: []},
                {},
                {},
            )

        def build_page(*_args, **_kwargs):
            nonlocal analysis_overlapped_output
            if not analysis_overlapped_output:
                analysis_overlapped_output = third_started.wait(0.5)

        encoded_page = cv2.imencode(
            ".png", np.full((30, 20), 255, np.uint8)
        )[1].tobytes()
        sample_pages = {0: [encoded_page] * 4}
        config = TemplatePreset(page_count=1)
        controller = ParallelControllerStub()

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "sample.pdf"
            document = fitz.open()
            for _ in range(4):
                document.new_page(width=20, height=30)
            document.save(pdf_path)
            document.close()

            template = {0: np.full((30, 20), 255, np.uint8)}
            with (
                patch("src.processor._build_page_aligners", return_value=[object()]),
                patch("src.processor._remap_checkbox_layout", return_value=config),
                patch(
                    "src.processor._checkbox_layout_is_trustworthy",
                    return_value=True,
                ),
                patch("src.processor._prepare_field_plans", return_value=[]),
                patch("src.processor.process_survey_data", side_effect=process_survey),
                patch(
                    "src.processor._build_encoded_vector_page",
                    side_effect=build_page,
                ),
            ):
                _, rows, comments = _analyze_single_file(
                    str(pdf_path),
                    "sample",
                    config,
                    template,
                    template,
                    [template[0]],
                    Path(temp_dir),
                    sample_pages=sample_pages,
                    resource_controller=controller,
                )

        self.assertGreater(max_active, 1)
        self.assertTrue(analysis_overlapped_output)
        self.assertEqual(controller.pending_tasks, [4])
        self.assertEqual(controller.checkpoints, [])
        self.assertEqual(
            [row["페이지"] for row in rows],
            ["sample_1p", "sample_2p", "sample_3p", "sample_4p"],
        )
        self.assertEqual(comments, [])

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

        def build_template(samples, **_kwargs):
            path = sample_names[id(samples)]
            events.append(("template", path))
            shade = 200 - file_paths.index(path) * 20
            return {0: np.full((8, 8), shade, np.uint8)}

        def analyze_file(fpath, file_label, *_args, sample_pages=None, **_kwargs):
            self.assertIs(sample_pages, sample_sets[fpath])
            events.append(("analyze", fpath))
            return file_label, [{"파일명": file_label, "페이지": "1p"}], []

        def export_rows(results, _config, _out_path):
            exported_rows.extend(results)
            return True

        config = TemplatePreset(page_count=1)
        template_page = np.full((8, 8), 255, np.uint8)
        resource_controller = _ResourceControllerStub()

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
                    patch("src.processor._validate_analysis_page_geometry"),
                    patch("src.processor._insert_img_into_pdf"),
                    patch("src.processor.export_to_excel", side_effect=export_rows),
                ):
                    success = run_analysis(
                        file_paths,
                        [template_page],
                        config,
                        resource_controller=resource_controller,
                        output_base_dir=temp_dir,
                    )
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
        self.assertTrue(resource_controller.started)
        self.assertTrue(resource_controller.closed)
        self.assertEqual(len(resource_controller.checkpoints), 3)

    def test_batch_analysis_stops_before_answers_when_a_template_is_missing(self):
        file_paths = ["broken.pdf", "valid.pdf"]
        valid_samples = {0: [b"valid"]}
        template = {0: np.full((8, 8), 210, np.uint8)}
        analyze_file = patch("src.processor._analyze_single_file").start()
        self.addCleanup(patch.stopall)

        def collect_samples(fpath, *_args, **_kwargs):
            if fpath == "broken.pdf":
                return _file_key(fpath), {}
            return _file_key(fpath), valid_samples

        with tempfile.TemporaryDirectory() as temp_dir:
            resource_controller = _ResourceControllerStub()
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
                    patch("src.processor._validate_analysis_page_geometry"),
                    patch("src.processor._insert_img_into_pdf"),
                ):
                    with self.assertRaisesRegex(
                        ValueError, "자동 생성 템플릿.*1쪽"
                    ):
                        run_analysis(
                            file_paths,
                            [np.full((8, 8), 255, np.uint8)],
                            TemplatePreset(page_count=1),
                            resource_controller=resource_controller,
                            output_base_dir=temp_dir,
                        )
            finally:
                os.chdir(previous_cwd)

        analyze_file.assert_not_called()
        self.assertTrue(resource_controller.started)
        self.assertTrue(resource_controller.closed)

    def test_page_geometry_allows_dpi_changes_with_the_same_ratio(self):
        reference = np.full((1000, 707), 255, np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "same-ratio.pdf"
            doc = fitz.open()
            doc.new_page(width=595, height=842)
            doc.new_page(width=1190, height=1684)
            doc.save(pdf_path)
            doc.close()

            _validate_analysis_page_geometry(
                [str(pdf_path)],
                [reference],
                TemplatePreset(page_count=1),
            )

    def test_page_geometry_blocks_a_materially_different_ratio(self):
        reference = np.full((1000, 707), 255, np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "wrong-ratio.pdf"
            doc = fitz.open()
            doc.new_page(width=612, height=792)
            doc.save(pdf_path)
            doc.close()

            with self.assertRaisesRegex(ValueError, "페이지 비율.*다릅니다"):
                _validate_analysis_page_geometry(
                    [str(pdf_path)],
                    [reference],
                    TemplatePreset(page_count=1),
                )

    def test_analysis_layout_blocks_an_incompatible_checkbox_template(self):
        template = np.full((100, 80), 255, np.uint8)
        config = TemplatePreset(
            page_count=1,
            fields=[Field(name="Q1", boxes=[Box(0, 10, 10, 10, 10)])],
        )
        incompatible = SimpleNamespace(expected_boxes=1, compatible=False)

        with patch(
            "src.processor.remap_preset_to_detected_layout",
            return_value=incompatible,
        ):
            with self.assertRaisesRegex(ValueError, "정합 신뢰도가 부족"):
                _validate_analysis_template_layout(
                    config,
                    {0: template},
                    [template],
                    "다른양식",
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
