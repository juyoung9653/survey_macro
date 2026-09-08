"""Production wiring and opt-in real mixed-orientation scan regression."""
import hashlib
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import cv2
import fitz
import numpy as np

from src.models import TemplatePreset
from src.processor import (
    _align_with_page_context, _build_page_aligners, _render_aligned_page,
    _render_pdf_page,
)
from src.vision import ImageAligner, PageOrientationError


class OrientationPipelineTests(unittest.TestCase):
    def test_analysis_enables_orientation_for_each_reference(self):
        references = [np.full((200, 150), 255, np.uint8)] * 2
        with patch("src.processor.ImageAligner") as constructor:
            _build_page_aligners(references, TemplatePreset(page_count=2))
        self.assertEqual(constructor.call_count, 2)
        for call in constructor.call_args_list:
            self.assertTrue(call.kwargs["auto_orient_180"])

    def test_orientation_error_identifies_file_and_one_based_page(self):
        aligner = Mock()
        aligner.align.side_effect = PageOrientationError("ambiguous")
        with self.assertRaisesRegex(PageOrientationError, "sample.pdf.*7쪽"):
            _align_with_page_context(aligner, np.zeros((2, 2)), "sample.pdf", 6)

    @unittest.skipUnless(os.getenv("RUN_BOARD_ORIENTATION_TEST") == "1",
                         "opt-in local board-game PDF")
    def test_all_29_board_pages_match_manually_oriented_alignment(self):
        source = Path("C:/Users/Public/scan/보드게임.pdf")
        self.assertTrue(source.is_file())
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        with fitz.open(source) as doc:
            self.assertEqual(len(doc), 29)
            reference = _render_pdf_page(doc, 0, 120)
            automatic = _build_page_aligners([reference], TemplatePreset())
            manual = ImageAligner(reference, sparse_lk=True)
            flipped_pages = []
            for index in range(len(doc)):
                _, actual = _render_aligned_page(
                    doc, index, 0, automatic, -1, 0.0, 120
                )
                raw = _render_pdf_page(doc, index, 120)
                if 6 <= index <= 10:
                    raw = cv2.rotate(raw, cv2.ROTATE_180)
                expected = manual.align(raw)
                with self.subTest(page=index + 1):
                    np.testing.assert_array_equal(actual, expected)
                if automatic[0].last_orientation_degrees == 180:
                    flipped_pages.append(index + 1)
        self.assertEqual(flipped_pages, [7, 8, 9, 10, 11])
        self.assertEqual(before, hashlib.sha256(source.read_bytes()).hexdigest())
        print("Board PDF: all 29 aligned pages equal manual correction; rotated 7-11; source unchanged")


if __name__ == "__main__":
    unittest.main()
