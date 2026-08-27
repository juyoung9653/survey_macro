import unittest

import cv2
import numpy as np

from src.mark_analysis import (
    _comment_has_overprint_evidence,
    _comment_region_has_meaningful_ink,
    _expand_comment_box,
)
from src.models import Box


class CommentDetectionTests(unittest.TestCase):
    def test_only_long_comment_lines_gain_a_bounded_corridor(self):
        short = Box(0, 100, 100, 600, 90)
        long = Box(0, 314, 1898, 1120, 109)

        self.assertEqual(_expand_comment_box(short), short)
        self.assertEqual(
            _expand_comment_box(long),
            Box(0, 74, 1871, 1450, 332),
        )

    def test_comment_filter_rejects_compact_scan_residue(self):
        mask = np.zeros((120, 160), np.uint8)
        cv2.rectangle(mask, (40, 40), (50, 50), 255, -1)

        self.assertFalse(
            _comment_region_has_meaningful_ink(mask, Box(0, 0, 0, 160, 120))
        )

    def test_comment_filter_keeps_a_handwritten_tick(self):
        mask = np.zeros((120, 160), np.uint8)
        cv2.line(mask, (35, 65), (60, 90), 255, 3)
        cv2.line(mask, (60, 90), (115, 35), 255, 3)

        self.assertTrue(
            _comment_region_has_meaningful_ink(mask, Box(0, 0, 0, 160, 120))
        )

    def test_comment_filter_keeps_clustered_compact_handwriting(self):
        mask = np.zeros((120, 200), np.uint8)
        for x in (45, 80, 115):
            cv2.rectangle(mask, (x, 45), (x + 10, 55), 255, -1)

        self.assertTrue(
            _comment_region_has_meaningful_ink(mask, Box(0, 0, 0, 200, 120))
        )

    def test_comment_filter_rejects_a_long_sparse_rule(self):
        mask = np.zeros((80, 180), np.uint8)
        cv2.line(mask, (20, 35), (150, 35), 255, 1)
        cv2.line(mask, (85, 35), (85, 50), 255, 1)

        self.assertFalse(
            _comment_region_has_meaningful_ink(mask, Box(0, 0, 0, 180, 80))
        )

    def test_overprint_detects_localized_darkening_over_answer_label(self):
        template = np.full((650, 1200), 245, np.uint8)
        target = template.copy()
        box = Box(0, 200, 300, 900, 100)
        cv2.line(target, (185, 330), (220, 375), 35, 8)
        cv2.line(target, (220, 375), (250, 325), 35, 8)

        self.assertTrue(_comment_has_overprint_evidence(target, template, box))

    def test_overprint_rejects_uniform_page_darkening(self):
        template = np.full((650, 1200), 245, np.uint8)
        target = np.full_like(template, 215)
        box = Box(0, 200, 300, 900, 100)

        self.assertFalse(_comment_has_overprint_evidence(target, template, box))


if __name__ == "__main__":
    unittest.main()
