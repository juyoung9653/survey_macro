import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.processor import _align_template_mask_by_coverage, extract_pure_ink_mask
from src.vision import ImageAligner, auto_detect_checkboxes


def _make_form(height: int = 700, width: int = 500) -> np.ndarray:
    image = np.full((height, width), 255, np.uint8)
    for y in range(90, height - 70, 70):
        cv2.line(image, (50, y), (width - 50, y), 0, 2)
        for x in range(70, width - 60, 70):
            cv2.rectangle(image, (x, y + 12), (x + 34, y + 46), 0, 2)
    return image


def _dark_mask(image: np.ndarray) -> np.ndarray:
    return cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)[1]


def _overlap(first: np.ndarray, second: np.ndarray) -> int:
    return cv2.countNonZero(cv2.bitwise_and(first, second))


class TemplateAlignmentTests(unittest.TestCase):
    def test_identity_alignment_is_unchanged(self):
        template = _dark_mask(_make_form())

        aligned = _align_template_mask_by_coverage(template, template.copy())

        self.assertTrue(np.array_equal(aligned, template))

    def test_coverage_alignment_improves_rotated_shifted_template(self):
        form = _make_form()
        height, width = form.shape
        template = _dark_mask(form)
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0.35, 1.0)
        matrix[0, 2] += 5
        matrix[1, 2] -= 4
        target = cv2.warpAffine(form, matrix, (width, height), borderValue=255)
        target_mask = _dark_mask(target)

        aligned = _align_template_mask_by_coverage(template, target_mask)

        self.assertGreater(_overlap(aligned, target_mask), _overlap(template, target_mask))

    def test_image_aligner_uses_one_coordinate_system_for_resized_pages(self):
        reference = _make_form()
        target = cv2.resize(reference, (750, 1050), interpolation=cv2.INTER_LINEAR)
        expected = cv2.resize(target, (500, 700), interpolation=cv2.INTER_AREA)
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)

        with (
            patch(
                "src.vision.cv2.estimateAffine2D",
                return_value=(identity.copy(), None),
            ) as estimate_affine,
            patch(
                "src.vision.cv2.findTransformECC",
                return_value=(0.999, identity.copy()),
            ) as find_ecc,
        ):
            aligned = ImageAligner(reference).align(target)

        estimate_affine.assert_called_once()
        find_ecc.assert_called_once()
        self.assertTrue(np.array_equal(aligned, expected))

    def test_ink_is_preserved_and_checkbox_detection_still_works(self):
        form = _make_form()
        ink = np.zeros_like(form)
        cv2.line(ink, (82, 120), (110, 150), 255, 6)
        cv2.line(ink, (110, 150), (135, 105), 255, 6)
        target = form.copy()
        target[ink > 0] = 0

        extracted = extract_pure_ink_mask(target, form, template_dilate_pct=0.0)
        boxes = auto_detect_checkboxes(cv2.cvtColor(form, cv2.COLOR_GRAY2BGR))

        self.assertGreater(_overlap(extracted, ink), 50)
        self.assertGreater(len(boxes), 0)


if __name__ == "__main__":
    unittest.main()
