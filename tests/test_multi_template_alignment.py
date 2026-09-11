import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import cv2
import fitz
import numpy as np

from src.models import Box, Field, TemplatePreset
from src.processor import (
    _align_with_page_context,
    _analyze_single_file,
    _prepare_field_plans,
    remap_preset_to_detected_layout,
)
from src.vision import ImageAligner, PageOrientationError


def _asymmetric_form() -> np.ndarray:
    image = np.full((480, 360), 255, np.uint8)
    cv2.putText(image, "TOP", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    cv2.rectangle(image, (45, 180), (310, 390), 0, 3)
    for y in range(210, 370, 55):
        cv2.rectangle(image, (90, y), (120, y + 25), 0, 2)
        cv2.rectangle(image, (220, y), (250, y + 25), 0, 2)
    return image


def _checkbox_grid(rows: int, offset_x: int = 0, offset_y: int = 0) -> np.ndarray:
    image = np.full((600, 500), 255, np.uint8)
    for row in range(rows):
        for column in range(4):
            x = 70 + column * 90 + offset_x
            y = 90 + row * 75 + offset_y
            cv2.rectangle(image, (x, y), (x + 32, y + 32), 0, 2)
    return image


def _mixed_answer_layout(main_offset_x: int = 0) -> tuple[np.ndarray, list[Box], list[Box]]:
    image = np.full((700, 900), 255, np.uint8)
    small_boxes = []
    large_boxes = []
    for column in range(4):
        x = 80 + column * 65
        cv2.rectangle(image, (x, 70), (x + 28, 98), 0, 2)
        small_boxes.append(Box(0, x, 70, 28, 28))
    for row in range(5):
        for column in range(5):
            x = 260 + column * 105 + main_offset_x
            y = 250 + row * 70
            cv2.rectangle(image, (x, y), (x + 88, y + 56), 0, 2)
            large_boxes.append(Box(0, x, y, 88, 56))
    return image, small_boxes, large_boxes


class MultiTemplateAlignmentTests(unittest.TestCase):
    def test_analysis_snaps_each_shifted_large_grid_before_annotations(self):
        source, demographics, source_grid = _mixed_answer_layout()
        target, _target_demographics, target_grid = _mixed_answer_layout(22)
        cv2.line(target, (target_grid[0].x + 18, target_grid[0].y + 42),
                 (target_grid[0].x + 70, target_grid[0].y + 14), 0, 5)
        config = TemplatePreset(
            page_count=1,
            fields=[
                Field(name="demographics", boxes=demographics),
                Field(name="main_grid", boxes=source_grid),
            ],
        )
        file_config = remap_preset_to_detected_layout(
            config, {0: source}, source_templates={0: source}
        ).config
        expected = remap_preset_to_detected_layout(
            file_config, {0: target}, source_templates={0: source}
        ).config.fields[1].boxes[0]
        expected_working = _prepare_field_plans(
            remap_preset_to_detected_layout(
                file_config, {0: target}, source_templates={0: source}
            ).config
        )[1][2][0]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "sample.pdf"
            document = fitz.open()
            document.new_page(width=500, height=700)
            document.save(pdf_path)
            document.close()
            captured = []
            with patch("src.processor._render_survey_pages", return_value={0: target}), patch(
                "src.processor._build_encoded_vector_page",
                side_effect=lambda _doc, _image, annotations: captured.append(annotations),
            ):
                _analyze_single_file(
                    str(pdf_path), "sample", config, {0: source}, {0: source},
                    [source], root,
                )

        rectangles = [annotation for page in captured for annotation in page]
        self.assertTrue(rectangles)
        self.assertIn(
            (
                expected_working.x,
                expected_working.y,
                expected_working.w,
                expected_working.h,
            ),
            [rectangle[:4] for rectangle in rectangles],
        )

    def test_complete_frame_remap_moves_large_cells_without_moving_demographics(self):
        source, demographics, source_grid = _mixed_answer_layout()
        target, _target_demographics, target_grid = _mixed_answer_layout(22)
        config = TemplatePreset(
            page_count=1,
            fields=[
                Field(name="demographics", boxes=demographics),
                Field(name="main_grid", boxes=source_grid),
            ],
        )

        remap = remap_preset_to_detected_layout(
            config, {0: target}, source_templates={0: source}
        )
        source_remap = remap_preset_to_detected_layout(
            config, {0: source}, source_templates={0: source}
        )

        self.assertTrue(remap.accepted)
        self.assertTrue(remap.compatible)
        remapped_demographics = remap.config.fields[0].boxes
        remapped_grid = remap.config.fields[1].boxes
        self.assertEqual(
            [box.x for box in remapped_demographics],
            [box.x for box in source_remap.config.fields[0].boxes],
        )
        self.assertEqual(
            [box.x - source_box.x for box, source_box in zip(remapped_grid, source_remap.config.fields[1].boxes)],
            [22] * len(remapped_grid),
        )
        self.assertEqual(
            [box.y for box in remapped_grid],
            [box.y for box in source_remap.config.fields[1].boxes],
        )

    def test_checkbox_layout_accepts_a_shifted_grid_but_rejects_changed_count(self):
        reference = _checkbox_grid(4)
        shifted = _checkbox_grid(4, offset_x=12, offset_y=18)
        changed_count = _checkbox_grid(3, offset_x=12, offset_y=18)

        self.assertIsNotNone(
            ImageAligner(reference, refine_ecc=False).align_if_checkbox_layout_matches(shifted)
        )
        self.assertIsNone(
            ImageAligner(reference, refine_ecc=False).align_if_checkbox_layout_matches(changed_count)
        )

    def test_orb_and_ecc_fallback_rejects_an_upside_down_page(self):
        reference = _asymmetric_form()
        upside_down = cv2.rotate(reference, cv2.ROTATE_180)

        upright_aligner = ImageAligner(reference, refine_ecc=False)
        upright = upright_aligner.align_if_orb_confident(reference)
        flipped_aligner = ImageAligner(reference, refine_ecc=False)
        flipped = flipped_aligner.align_if_orb_confident(upside_down)

        self.assertIsNotNone(upright)
        self.assertGreaterEqual(upright_aligner.last_quick_score, 0.60)
        self.assertIsNone(flipped)

    def test_recovers_an_upright_variant_with_orb_and_ecc_evidence(self):
        automatic = Mock(last_orientation_status="untrustworthy")
        automatic.align.side_effect = PageOrientationError("untrustworthy")
        expected = np.full((3, 4), 255, np.uint8)
        automatic.align_if_orb_confident.return_value = expected

        actual = _align_with_page_context(
            automatic, np.zeros((3, 4), np.uint8), "variant.pdf", 0
        )

        self.assertIs(actual, expected)
        automatic.align_if_orb_confident.assert_called_once()

    def test_keeps_an_ambiguous_orientation_error(self):
        automatic = Mock(last_orientation_status="ambiguous")
        automatic.align.side_effect = PageOrientationError("ambiguous")

        with self.assertRaises(PageOrientationError):
            _align_with_page_context(
                automatic, np.zeros((3, 4), np.uint8), "variant.pdf", 0
            )

        automatic.align_if_orb_confident.assert_not_called()

    def test_rejects_a_variant_without_orb_and_ecc_evidence(self):
        automatic = Mock(last_orientation_status="untrustworthy")
        automatic.align.side_effect = PageOrientationError("untrustworthy")
        automatic.align_if_orb_confident.return_value = None
        automatic.align_if_checkbox_layout_matches.return_value = None

        with self.assertRaises(PageOrientationError):
            _align_with_page_context(
                automatic, np.zeros((3, 4), np.uint8), "variant.pdf", 0
            )
