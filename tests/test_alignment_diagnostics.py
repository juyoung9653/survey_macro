import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from src.processor import _align_with_page_context
from src.vision import ImageAligner, PageOrientationError


def _grid(rows=4):
    image = np.full((600, 500), 255, np.uint8)
    for row in range(rows):
        for column in range(4):
            x, y = 70 + column * 90, 90 + row * 75
            cv2.rectangle(image, (x, y), (x + 32, y + 32), 0, 2)
    return image


class AlignmentDiagnosticsTests(unittest.TestCase):
    def test_blank_page_reports_missing_candidates_not_a_count_mismatch(self):
        aligner = ImageAligner(_grid())
        self.assertIsNone(aligner.align_if_checkbox_layout_matches(np.full((600, 500), 255, np.uint8)))
        self.assertEqual(aligner.last_alignment_diagnostics['reason'], 'too_few_detected_boxes')
        self.assertEqual(aligner.last_alignment_diagnostics['candidate_boxes'], 0)

    def test_removed_row_reports_count_difference(self):
        aligner = ImageAligner(_grid())
        self.assertIsNone(aligner.align_if_checkbox_layout_matches(_grid(3)))
        diagnostic = aligner.last_alignment_diagnostics
        self.assertEqual(diagnostic['reason'], 'box_count_mismatch')
        self.assertEqual((diagnostic['reference_boxes'], diagnostic['candidate_boxes']), (16, 12))

    def test_excessive_translation_reports_translation_not_rotation(self):
        aligner = ImageAligner(_grid())
        with patch('src.vision.cv2.phaseCorrelate', return_value=((150, 0), 1.0)):
            self.assertIsNone(aligner.align_if_checkbox_layout_matches(_grid()))
        self.assertEqual(aligner.last_alignment_diagnostics['reason'], 'translation_exceeds_limit')

    def test_error_includes_both_files_pages_and_required_matches(self):
        aligner = Mock(last_orientation_status='untrustworthy')
        aligner.reference_context = ('reference.pdf', 1)
        aligner.align.side_effect = PageOrientationError('페이지를 정렬하지 못했습니다.')
        aligner.align_if_orb_confident.return_value = None
        aligner.align_if_checkbox_layout_matches.return_value = None
        aligner.last_alignment_diagnostics = dict(reason='insufficient_matches', reference_boxes=44, candidate_boxes=44, matched=32, required_matches=37)
        with self.assertRaises(PageOrientationError) as raised:
            _align_with_page_context(aligner, _grid(), 'candidate.pdf', 3)
        message = str(raised.exception)
        self.assertIn("'candidate.pdf' 4쪽", message)
        self.assertIn("'reference.pdf' 2쪽", message)
        self.assertIn('기준 44개 / 현재 44개', message)
        self.assertIn('32개 / 필요한 대응 37개', message)
        self.assertIsInstance(raised.exception.__cause__, PageOrientationError)

    def test_new_page_clears_previous_failure_measurements(self):
        aligner = ImageAligner(_grid())
        aligner.last_alignment_diagnostics['reason'] = 'box_count_mismatch'
        aligner.align(_grid())
        self.assertNotIn('reason', aligner.last_alignment_diagnostics)
