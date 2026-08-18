import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.models import TemplatePreset
from src.processor import _prepare_alignment_references, _ui_template_cache_path
from src.vision import (
    _checkbox_cache_key,
    apply_rotation,
    estimate_deskew_angle,
)


def _make_grid(height: int = 900, width: int = 650) -> np.ndarray:
    image = np.full((height, width), 255, np.uint8)
    for y in range(100, height - 80, 80):
        cv2.line(image, (55, y), (width - 55, y), 0, 2)
        for x in range(80, width - 60, 95):
            cv2.rectangle(image, (x, y + 18), (x + 30, y + 48), 0, 2)
    cv2.line(image, (55, 70), (55, height - 70), 0, 2)
    cv2.line(image, (width - 55, 70), (width - 55, height - 70), 0, 2)
    return image


class DeskewTests(unittest.TestCase):
    def test_estimate_deskew_corrects_small_document_rotation(self):
        source = _make_grid()
        rotated = apply_rotation(source, -1, 1.2)

        estimated = estimate_deskew_angle(rotated)

        self.assertIsNotNone(estimated)
        assert estimated is not None
        self.assertAlmostEqual(estimated, -1.2, delta=0.25)
        corrected = apply_rotation(rotated, -1, estimated)
        remaining = estimate_deskew_angle(corrected)
        self.assertIsNotNone(remaining)
        assert remaining is not None
        self.assertAlmostEqual(remaining, 0.0, delta=0.25)

    def test_estimate_deskew_rejects_blank_page(self):
        blank = np.full((700, 500), 255, np.uint8)
        self.assertIsNone(estimate_deskew_angle(blank))

    def test_horizontal_form_rules_are_not_masked_by_zero_degree_columns(self):
        horizontal = np.full((900, 650), 255, np.uint8)
        for y in range(100, 820, 70):
            cv2.line(horizontal, (45, y), (605, y), 0, 2)
        horizontal = apply_rotation(horizontal, -1, 1.2)

        vertical = np.full_like(horizontal, 255)
        for x in range(65, 610, 75):
            cv2.line(vertical, (x, 80), (x, 830), 0, 2)
        mixed = np.minimum(horizontal, vertical)

        estimated = estimate_deskew_angle(mixed)

        self.assertIsNotNone(estimated)
        assert estimated is not None
        self.assertAlmostEqual(estimated, -1.2, delta=0.25)

    def test_page_angles_are_added_to_manual_correction(self):
        preset = TemplatePreset(
            fine_angle=0.2,
            page_fine_angles=[-0.55, 0.35],
        )

        self.assertAlmostEqual(preset.fine_angle_for_page(0), -0.35)
        self.assertAlmostEqual(preset.fine_angle_for_page(1), 0.55)
        self.assertAlmostEqual(preset.fine_angle_for_page(2), 0.2)

    def test_page_angles_are_part_of_template_and_checkbox_cache_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "survey.pdf"
            pdf_path.write_bytes(b"placeholder")

            first_template_key = _ui_template_cache_path(
                [str(pdf_path)], 2, -1, 0.0, "multi", [-0.5, 0.2]
            )
            second_template_key = _ui_template_cache_path(
                [str(pdf_path)], 2, -1, 0.0, "multi", [0.0, 0.2]
            )
            first_checkbox_key = _checkbox_cache_key(
                [str(pdf_path)],
                2,
                -1,
                0.0,
                page_fine_angles=[-0.5, 0.2],
            )
            second_checkbox_key = _checkbox_cache_key(
                [str(pdf_path)],
                2,
                -1,
                0.0,
                page_fine_angles=[0.0, 0.2],
            )

        self.assertNotEqual(first_template_key, second_template_key)
        self.assertNotEqual(first_checkbox_key, second_checkbox_key)

    def test_preprocessed_preset_template_is_not_rotated_twice(self):
        page = np.full((40, 30), 255, np.uint8)
        preset = TemplatePreset(
            page_count=1,
            fine_angle=0.4,
            page_fine_angles=[-0.7],
        )

        with patch("src.processor.apply_rotation") as rotate:
            references = _prepare_alignment_references(
                [page], preset, template_pages_preprocessed=True
            )

        self.assertIs(references[0], page)
        rotate.assert_not_called()

    def test_raw_template_pages_receive_their_combined_page_angles(self):
        pages = [
            np.full((40, 30), 250, np.uint8),
            np.full((40, 30), 240, np.uint8),
        ]
        preset = TemplatePreset(
            page_count=2,
            fine_angle=0.2,
            page_fine_angles=[-0.5, 0.3],
        )

        with patch(
            "src.processor.apply_rotation", side_effect=lambda page, *_args: page
        ) as rotate:
            references = _prepare_alignment_references(pages, preset)

        self.assertEqual(len(references), 2)
        self.assertAlmostEqual(rotate.call_args_list[0].args[2], -0.3)
        self.assertAlmostEqual(rotate.call_args_list[1].args[2], 0.5)


if __name__ == "__main__":
    unittest.main()
