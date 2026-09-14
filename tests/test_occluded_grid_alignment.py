import unittest
import os
from pathlib import Path

import cv2
import fitz
import numpy as np

from src.vision import ImageAligner


def _occluded_grid_form(
    *,
    include_header: bool,
    rows: int = 6,
    columns: int = 5,
    cell_width: int = 70,
    upright_marker: bool = True,
) -> np.ndarray:
    """A form whose header can be blank while its answer grid remains intact."""
    image = np.full((1200, 900), 255, np.uint8)
    if include_header:
        # These four non-answer frames make the normal all-frame count check
        # fail when the target's upper-left area is white.
        for index in range(4):
            left = 90 + index * 170
            cv2.rectangle(image, (left, 100), (left + 130, 150), 0, 2)
    for row in range(rows):
        for column in range(columns):
            left = 270 + column * 75
            top = 410 + row * 65
            cv2.rectangle(image, (left, top), (left + cell_width, top + 55), 0, 2)
    if upright_marker:
        cv2.putText(
            image, "UPRIGHT", (130, 930), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 0, 6
        )
    return image


class OccludedGridAlignmentTests(unittest.TestCase):
    def _align(self, reference: np.ndarray, candidate: np.ndarray) -> ImageAligner:
        aligner = ImageAligner(reference, refine_ecc=False)
        self.result = aligner.align_if_checkbox_layout_matches(candidate)
        return aligner

    def test_accepts_complete_grid_when_only_header_is_occluded(self):
        aligner = self._align(
            _occluded_grid_form(include_header=True),
            _occluded_grid_form(include_header=False),
        )

        self.assertIsNotNone(self.result)
        self.assertEqual(aligner.last_alignment_stage, "checkbox_layout_grid")
        self.assertEqual(aligner.last_alignment_diagnostics["grid_shape"], (6, 5))
        self.assertEqual(aligner.last_alignment_diagnostics["matched"], 30)

    def test_rejects_missing_answer_row(self):
        aligner = self._align(
            _occluded_grid_form(include_header=True),
            _occluded_grid_form(include_header=False, rows=5),
        )

        self.assertIsNone(self.result)
        self.assertEqual(
            aligner.last_alignment_diagnostics["grid_fallback_reason"],
            "grid_shape_mismatch",
        )

    def test_rejects_missing_answer_column(self):
        aligner = self._align(
            _occluded_grid_form(include_header=True),
            _occluded_grid_form(include_header=False, columns=4),
        )

        self.assertIsNone(self.result)
        self.assertEqual(
            aligner.last_alignment_diagnostics["grid_fallback_reason"],
            "grid_shape_mismatch",
        )

    def test_rejects_changed_answer_cell_edges(self):
        aligner = self._align(
            _occluded_grid_form(include_header=True),
            _occluded_grid_form(include_header=False, cell_width=85),
        )

        self.assertIsNone(self.result)
        self.assertEqual(
            aligner.last_alignment_diagnostics["grid_fallback_reason"],
            "grid_edges_mismatch",
        )

    def test_rejects_grid_without_upright_evidence(self):
        aligner = self._align(
            _occluded_grid_form(include_header=True, upright_marker=False),
            _occluded_grid_form(include_header=False, upright_marker=False),
        )

        self.assertIsNone(self.result)
        self.assertEqual(
            aligner.last_alignment_diagnostics["grid_fallback_reason"],
            "upright_evidence_missing",
        )

    def test_rejects_upside_down_complete_grid(self):
        reference = _occluded_grid_form(include_header=True)
        candidate = cv2.rotate(
            _occluded_grid_form(include_header=False), cv2.ROTATE_180
        )
        aligner = self._align(reference, candidate)

        self.assertIsNone(self.result)
        self.assertEqual(
            aligner.last_alignment_diagnostics["grid_fallback_reason"],
            "upright_evidence_missing",
        )

    def test_startup_pdf_occluded_first_page_uses_its_complete_grid(self):
        source = Path(os.getenv("SURVEY_SCAN_CORPUS", "C:/Users/Public/scan")) / "자활창업.pdf"
        self.assertTrue(source.is_file(), f"real PDF fixture is missing: {source}")
        with fitz.open(source) as document:
            self.assertGreaterEqual(len(document), 3)
            pages = []
            for index in (2, 0):
                pixmap = document[index].get_pixmap(dpi=100, colorspace=fitz.csGRAY)
                pages.append(
                    np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                        pixmap.h, pixmap.w
                    )
                )
        aligner = self._align(pages[0], pages[1])

        self.assertIsNotNone(self.result)
        self.assertEqual(aligner.last_alignment_stage, "checkbox_layout_grid")
        self.assertEqual(aligner.last_alignment_diagnostics["grid_shape"], (6, 5))
