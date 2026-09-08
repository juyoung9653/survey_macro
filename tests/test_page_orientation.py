import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.vision import ImageAligner, PageOrientationError


def _asymmetric_form(height: int = 760, width: int = 540) -> np.ndarray:
    page = np.full((height, width), 255, np.uint8)
    cv2.rectangle(page, (45, 42), (width - 45, height - 42), 0, 3)
    cv2.putText(page, "FORM A7", (72, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.15, 0, 3)
    cv2.putText(page, "TOP", (75, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    for row, y in enumerate(range(245, height - 75, 92), start=1):
        cv2.putText(page, str(row), (82, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, 0, 2)
        cv2.line(page, (125, y), (width - 78, y), 0, 2)
        for x in (155, 270, 385):
            cv2.rectangle(page, (x, y + 17), (x + 31, y + 48), 0, 2)
    cv2.circle(page, (width - 96, height - 88), 19, 0, -1)
    return page


def _scanner_distortion(reference: np.ndarray) -> np.ndarray:
    height, width = reference.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -0.7, 1.006)
    matrix[0, 2] += 5.0
    matrix[1, 2] -= 4.0
    return cv2.warpAffine(reference, matrix, (width, height), borderValue=255)


def _mean_error(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(first.astype(np.float32) - second.astype(np.float32))[35:-35, 35:-35]
        )
    )


class PageOrientationTests(unittest.TestCase):
    def test_rotates_upside_down_grayscale_before_fine_alignment(self):
        reference = _asymmetric_form()
        upside_down = cv2.rotate(_scanner_distortion(reference), cv2.ROTATE_180)

        aligner = ImageAligner(reference, auto_orient_180=True)
        aligned = aligner.align(upside_down)

        self.assertEqual(aligner.last_orientation_degrees, 180)
        self.assertEqual(aligner.last_orientation_status, "rotated_180")
        self.assertLess(_mean_error(aligned, reference), 18.0)

    def test_handles_bgr_input_when_ecc_refinement_is_disabled(self):
        reference = _asymmetric_form()
        upside_down = cv2.rotate(reference, cv2.ROTATE_180)
        bgr_upside_down = cv2.cvtColor(upside_down, cv2.COLOR_GRAY2BGR)

        aligner = ImageAligner(
            cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR),
            refine_ecc=False,
            auto_orient_180=True,
        )
        aligned = aligner.align(bgr_upside_down)

        self.assertEqual(aligner.last_orientation_degrees, 180)
        self.assertTrue(np.array_equal(aligned, cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)))

    def test_stable_mask_is_applied_in_reference_coordinates_after_rotation(self):
        reference = _asymmetric_form()
        stable_mask = np.zeros_like(reference)
        stable_mask[30 : reference.shape[0] // 2 + 80, 25 : reference.shape[1] - 25] = 255
        upside_down = cv2.rotate(_scanner_distortion(reference), cv2.ROTATE_180)

        aligner = ImageAligner(
            reference,
            stable_mask=stable_mask,
            auto_orient_180=True,
        )
        aligned = aligner.align(upside_down)

        self.assertEqual(aligner.last_orientation_degrees, 180)
        self.assertLess(_mean_error(aligned, reference), 18.0)

    def test_alternating_orientations_do_not_poison_cached_alignment(self):
        reference = _asymmetric_form()
        upright = _scanner_distortion(reference)
        upside_down = cv2.rotate(upright, cv2.ROTATE_180)
        aligner = ImageAligner(reference, auto_orient_180=True)

        outputs = []
        orientations = []
        for page in (upright, upside_down, upright, upside_down):
            outputs.append(aligner.align(page))
            orientations.append(aligner.last_orientation_degrees)

        self.assertEqual(orientations, [0, 180, 0, 180])
        self.assertLess(max(_mean_error(output, reference) for output in outputs), 20.0)

    def test_blank_and_symmetric_pages_fail_instead_of_guessing(self):
        blank = np.full((300, 220), 255, np.uint8)
        blank_aligner = ImageAligner(blank, auto_orient_180=True)
        with self.assertRaises(PageOrientationError):
            blank_aligner.align(blank.copy())
        self.assertEqual(blank_aligner.last_orientation_status, "untrustworthy")

        # Odd dimensions give ROTATE_180 an integer centre, so this is exactly
        # symmetric rather than merely visually close to symmetric.
        symmetric = np.full((301, 221), 255, np.uint8)
        cv2.rectangle(symmetric, (45, 55), (175, 245), 0, 3)
        cv2.circle(symmetric, (110, 150), 34, 0, 3)
        symmetric_aligner = ImageAligner(symmetric, auto_orient_180=True)
        with self.assertRaises(PageOrientationError):
            symmetric_aligner.align(symmetric.copy())
        self.assertEqual(symmetric_aligner.last_orientation_status, "ambiguous")

    def test_default_mode_does_not_attempt_orientation(self):
        reference = _asymmetric_form()
        aligner = ImageAligner(reference)

        aligner.align(reference)

        self.assertEqual(aligner.last_orientation_degrees, 0)
        self.assertEqual(aligner.last_orientation_status, "disabled")

    def test_confident_upright_page_skips_two_candidate_orientation_ecc(self):
        reference = _asymmetric_form()
        aligner = ImageAligner(reference, auto_orient_180=True)

        with patch.object(
            aligner,
            "_orientation_affine_score",
            wraps=aligner._orientation_affine_score,
        ) as affine_score:
            aligner.align(reference)

        affine_score.assert_not_called()
        self.assertEqual(aligner.last_orientation_status, "upright")


if __name__ == "__main__":
    unittest.main()
