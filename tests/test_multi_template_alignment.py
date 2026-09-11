import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from src.processor import _align_with_page_context
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


class MultiTemplateAlignmentTests(unittest.TestCase):
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
