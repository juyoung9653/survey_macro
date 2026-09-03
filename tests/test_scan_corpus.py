import json
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import openpyxl

from src.models import Box, Field, TemplatePreset
from src.processor import (
    generate_ui_templates,
    generate_ui_templates_multi,
    remap_preset_to_detected_layout,
    run_analysis,
    _validate_analysis_page_geometry,
)
from src.vision import (
    auto_detect_checkboxes,
    estimate_deskew_angle,
    load_pdf_pages,
)


_RUN_CORPUS_TESTS = os.getenv("RUN_SCAN_CORPUS_TESTS") == "1"
_SCAN_DIR = Path(
    os.getenv("SURVEY_SCAN_CORPUS", r"C:\Users\Public\scan")
)
_PRESET_DIR = Path(
    os.getenv(
        "SURVEY_PRESET_DIR",
        str(Path(os.getenv("LOCALAPPDATA", "")) / "CheckFinder" / "presets"),
    )
)

_ONE_PAGE_PDFS = (
    "보드게임.pdf",
    "색깔.pdf",
    "운영실무.pdf",
    "웰다잉.pdf",
    "자활창업.pdf",
)
_TWO_PAGE_PDFS = (
    "거점.pdf",
    "급식.pdf",
    "도서관.pdf",
    "라라.pdf",
    "매점.pdf",
    "물류.pdf",
    "세차분식.pdf",
    "청소.pdf",
    "청수.pdf",
    "카페주거.pdf",
    "편의점.pdf",
    "헤이클린.pdf",
)

_ADDITIONAL_CORPUS_PDFS = (
    "S25C-0i26090316340.pdf",
    "법정의무.pdf",
)

_TWO_PAGE_ROW_SIGNATURES = (
    (2, 5, 4, 4, 2, 2, 4, 2, 2, 3, 3, 2, 2),
    (2, 1, 4, 1, 1, 1, 1, 1, 1, 2, 2, 1, 4, 4),
)


def _group_boxes_by_row(boxes):
    rows = []
    current = []
    for box in sorted(boxes, key=lambda item: item[1]):
        if not current:
            current.append(box)
            continue
        previous = current[-1]
        tolerance = max(box[3], previous[3]) * 0.5
        if abs(box[1] - previous[1]) <= tolerance:
            current.append(box)
        else:
            rows.append(current)
            current = [box]
    if current:
        rows.append(current)
    return rows


def _representative_groups(paths):
    groups = []
    for size in (2, 3):
        for start in range(0, len(paths), size):
            group = tuple(paths[start : start + size])
            if len(group) < 2:
                group = tuple(paths[-size:])
            if group not in groups:
                groups.append(group)
    all_paths = tuple(paths)
    if all_paths not in groups:
        groups.append(all_paths)
    return groups


@unittest.skipUnless(
    _RUN_CORPUS_TESTS,
    "set RUN_SCAN_CORPUS_TESTS=1 to run the external PDF corpus",
)
class ScanCorpusDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _SCAN_DIR.is_dir():
            raise AssertionError(f"scan corpus directory is missing: {_SCAN_DIR}")
        expected = (
            set(_ONE_PAGE_PDFS)
            | set(_TWO_PAGE_PDFS)
            | set(_ADDITIONAL_CORPUS_PDFS)
        )
        actual = {path.name for path in _SCAN_DIR.glob("*.pdf")}
        if actual != expected:
            missing = sorted(expected - actual)
            unclassified = sorted(actual - expected)
            raise AssertionError(
                f"scan corpus changed; missing={missing}, unclassified={unclassified}"
            )

    @staticmethod
    def _page_angles(path: Path, page_count: int):
        pages = load_pdf_pages(
            str(path),
            page_indices=list(range(page_count)),
        )
        return [float(estimate_deskew_angle(page) or 0.0) for page in pages]

    def _assert_layout(self, templates, page_count: int):
        self.assertEqual(set(templates), set(range(page_count)))
        detected_by_page = {
            page: auto_detect_checkboxes(template)
            for page, template in templates.items()
        }

        if page_count == 1:
            rows = _group_boxes_by_row(detected_by_page[0])
            small_rows = [
                row
                for row in rows
                if np.median([box[2] for box in row]) <= 50
                and np.median([box[3] for box in row]) <= 50
            ]
            grid_rows = [
                row
                for row in rows
                if len(row) >= 3
                and np.median([box[2] for box in row]) >= 100
                and np.median([box[3] for box in row]) >= 100
            ]
            self.assertEqual([len(row) for row in small_rows], [2, 6, 4])
            self.assertEqual([len(row) for row in grid_rows], [5] * 6)
            return

        signatures = tuple(
            tuple(len(row) for row in _group_boxes_by_row(detected_by_page[page]))
            for page in range(page_count)
        )
        self.assertEqual(signatures, _TWO_PAGE_ROW_SIGNATURES)

    def test_every_pdf_individually(self):
        for page_count, names in (
            (1, _ONE_PAGE_PDFS),
            (2, _TWO_PAGE_PDFS),
        ):
            for name in names:
                with self.subTest(pdf=name, page_count=page_count):
                    path = _SCAN_DIR / name
                    angles = self._page_angles(path, page_count)
                    templates = generate_ui_templates(
                        str(path),
                        page_count,
                        -1,
                        0.0,
                        page_fine_angles=angles,
                    )
                    _validate_analysis_page_geometry(
                        [str(path)],
                        [templates[page] for page in range(page_count)],
                        TemplatePreset(page_count=page_count),
                    )
                    self._assert_layout(templates, page_count)

    def test_compatible_pdfs_in_pairs_triples_and_all(self):
        for page_count, names in (
            (1, _ONE_PAGE_PDFS),
            (2, _TWO_PAGE_PDFS),
        ):
            paths = [_SCAN_DIR / name for name in names]
            for group in _representative_groups(paths):
                labels = tuple(path.name for path in group)
                with self.subTest(pdfs=labels, page_count=page_count):
                    angles = self._page_angles(group[0], page_count)
                    templates = generate_ui_templates_multi(
                        [str(path) for path in group],
                        page_count,
                        -1,
                        0.0,
                        page_fine_angles=angles,
                    )
                    _validate_analysis_page_geometry(
                        [str(path) for path in group],
                        [templates[page] for page in range(page_count)],
                        TemplatePreset(page_count=page_count),
                    )
                    self._assert_layout(templates, page_count)

    def test_one_page_pdfs_accept_default_preset_with_current_box_geometry(self):
        preset_path = _PRESET_DIR / "기본.json"
        template_path = _PRESET_DIR / "기본_tpl_p0.png"
        if not preset_path.is_file() or not template_path.is_file():
            raise AssertionError(
                f"default preset fixture is missing: {_PRESET_DIR}"
            )

        data = json.loads(preset_path.read_text(encoding="utf-8"))
        config = TemplatePreset(
            page_count=int(data.get("page_count", 1)),
            fine_angle=float(data.get("fine_angle", 0.0)),
            rot_code=int(data.get("rot_code", -1)),
            reverse_numbering=bool(data.get("reverse_numbering", True)),
            template_dilate_pct=float(data.get("template_dilate_pct", 0.3)),
            fields=[Field.from_dict(field) for field in data.get("fields", [])],
            page_fine_angles=[
                float(value) for value in data.get("page_fine_angles", [])
            ],
        )
        source_template = cv2.imdecode(
            np.frombuffer(template_path.read_bytes(), np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(source_template)

        color_pdf_path = _SCAN_DIR / "색깔.pdf"
        color_angles = []
        color_detected = []
        color_mapped_geometry = set()
        for name in _ONE_PAGE_PDFS:
            with self.subTest(pdf=name):
                pdf_path = _SCAN_DIR / name
                angles = self._page_angles(pdf_path, 1)
                templates = generate_ui_templates(
                    str(pdf_path),
                    1,
                    config.rot_code,
                    config.fine_angle,
                    page_fine_angles=angles,
                )
                detected = [
                    Box(0, x, y, w, h)
                    for x, y, w, h in auto_detect_checkboxes(templates[0])
                ]

                result = remap_preset_to_detected_layout(
                    config,
                    templates,
                    source_templates={0: source_template},
                    detected_boxes_by_page={0: detected},
                )

                self.assertTrue(result.accepted)
                self.assertTrue(result.compatible)
                self.assertEqual(result.matched_boxes, result.expected_boxes)
                detected_geometry = {
                    (box.page_idx, box.x, box.y, box.w, box.h)
                    for box in detected
                }
                mapped_geometry = {
                    (box.page_idx, box.x, box.y, box.w, box.h)
                    for field in result.config.fields
                    if not field.is_comment
                    for box in field.boxes
                }
                self.assertTrue(mapped_geometry <= detected_geometry)
                if pdf_path == color_pdf_path:
                    color_angles = angles
                    color_detected = detected
                    color_mapped_geometry = mapped_geometry

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication

        from src.ui import MainWindow

        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        try:
            window.preset_dir = _PRESET_DIR
            window.file_paths = [str(color_pdf_path)]
            window.pages = load_pdf_pages(
                str(color_pdf_path), page_indices=[0]
            )
            window.preset = TemplatePreset(page_count=1)
            window._pages_are_canonical = False
            window._reset_state_for_new_pdf()
            window.preset.page_fine_angles = color_angles
            window._update_page_size()
            raw_page = window.pages[0]
            window.auto_detect(mark_dirty=False)
            self.assertEqual(
                sum(len(field.boxes) for field in window.preset.fields),
                len(color_detected),
            )
            self.assertIs(window.pages[0], raw_page)
            self.assertIs(
                window._canvas_base_page(raw_page, 0),
                window._inferred_display_templates[0],
            )

            window._apply_loaded_preset(data, preset_name="기본")

            self.assertEqual(
                [field.name for field in window.preset.fields],
                [field.name for field in config.fields],
            )
            gui_geometry = {
                (box.page_idx, box.x, box.y, box.w, box.h)
                for field in window.preset.fields
                if not field.is_comment
                for box in field.boxes
            }
            self.assertEqual(gui_geometry, color_mapped_geometry)
        finally:
            window.close()
            app.processEvents()

    def test_single_response_keeps_answers_with_clean_preset_fallback(self):
        preset_path = _PRESET_DIR / "기본.json"
        template_path = _PRESET_DIR / "기본_tpl_p0.png"
        data = json.loads(preset_path.read_text(encoding="utf-8"))
        config = TemplatePreset(
            page_count=int(data.get("page_count", 1)),
            fine_angle=float(data.get("fine_angle", 0.0)),
            rot_code=int(data.get("rot_code", -1)),
            reverse_numbering=bool(data.get("reverse_numbering", True)),
            template_dilate_pct=float(data.get("template_dilate_pct", 0.3)),
            fields=[Field.from_dict(field) for field in data.get("fields", [])],
            page_fine_angles=[
                float(value) for value in data.get("page_fine_angles", [])
            ],
        )
        saved_template = cv2.imdecode(
            np.frombuffer(template_path.read_bytes(), np.uint8),
            cv2.IMREAD_COLOR,
        )
        pdf_path = _SCAN_DIR / "S25C-0i26090316340.pdf"
        current_templates = generate_ui_templates(
            str(pdf_path),
            config.page_count,
            config.rot_code,
            config.fine_angle,
            page_fine_angles=config.page_fine_angles,
        )
        remap = remap_preset_to_detected_layout(
            config,
            current_templates,
            source_templates={0: saved_template},
        )
        self.assertTrue(remap.accepted)

        from src.ui import MainWindow

        clean_pages = MainWindow._build_single_sample_template_pages(
            [saved_template],
            current_templates,
            remap.page_transforms,
            remap.config.page_count,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertTrue(
                run_analysis(
                    [str(pdf_path)],
                    [current_templates[0]],
                    remap.config,
                    template_pages_preprocessed=True,
                    output_base_dir=temp_dir,
                    single_sample_template_pages=clean_pages,
                )
            )
            workbook_path = next(Path(temp_dir).rglob("*.xlsx"))
            workbook = openpyxl.load_workbook(workbook_path, data_only=False)
            headers = [cell.value for cell in workbook["결과"][1]]
            values = [cell.value for cell in workbook["결과"][2]]

        answer = dict(zip(headers, values))
        self.assertEqual(answer["성별"], "남")
        self.assertEqual(answer["연령"], "50대")
        self.assertEqual(answer["경력"], "1년차")
        self.assertEqual([answer[f"Q{i}"] for i in range(1, 6)], [5] * 5)
        self.assertEqual(answer["의견"], "있음")


if __name__ == "__main__":
    unittest.main()
