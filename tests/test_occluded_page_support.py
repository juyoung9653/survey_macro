import unittest

import cv2
import numpy as np

from src.models import Box, Field, TemplatePreset
from src.processor import _configured_layout_has_frame_support, _complete_local_reference, remap_preset_to_detected_layout


class OccludedPageSupportTests(unittest.TestCase):
    def setUp(self):
        self.source = np.full((1000, 1200), 255, np.uint8)
        demographics = [Box(0, 80 + column * 55, 80, 36, 36) for column in range(12)]
        grid = [Box(0, 450 + column * 110, 300 + row * 105, 100, 95)
                for row in range(5) for column in range(5)]
        self.config = TemplatePreset(fields=[Field("demographics", boxes=demographics), Field("answers", boxes=grid)])
        for box in demographics + grid:
            cv2.rectangle(self.source, (box.x, box.y), (box.x + box.w, box.y + box.h), 0, 2)
        cv2.putText(self.source, "SURVEY", (100, 920), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 5)
        self.occluded = self.source.copy()
        self.occluded[:180] = 255

    def accepts(self, image):
        return _configured_layout_has_frame_support(self.config, {0: image}, {0: self.source})

    def test_white_demographics_with_intact_answer_grid_are_allowed(self):
        self.assertTrue(self.accepts(self.occluded))

    def test_missing_all_demographic_matches_returns_unaccepted_without_crashing(self):
        result = remap_preset_to_detected_layout(self.config, {0: self.occluded}, source_templates={0: self.source})
        self.assertFalse(result.accepted)
        self.assertEqual(result.expected_boxes, 37)

    def test_reference_uses_verified_complete_scan_without_changing_original(self):
        original = self.occluded.copy()
        reference = _complete_local_reference([self.occluded, self.source])
        self.assertLess(float(reference[:180].mean()), float(original[:180].mean()))
        np.testing.assert_array_equal(self.occluded, original)

    def test_reference_does_not_adopt_different_answer_grid(self):
        other = self.source.copy()
        other[290:840, 880:1100] = 255
        self.assertIs(_complete_local_reference([self.occluded, other]), self.occluded)

    def test_unframed_ink_is_not_treated_as_white_occlusion(self):
        marked = self.occluded.copy()
        cv2.line(marked, (84, 85), (103, 104), 0, 3)
        self.assertFalse(self.accepts(marked))

    def test_missing_answer_row_does_not_qualify_as_occlusion(self):
        missing = self.occluded.copy()
        missing[285:403, 440:1110] = 255
        self.assertFalse(self.accepts(missing))

    def test_shifted_answer_grid_does_not_validate_original_coordinates(self):
        moved = cv2.warpAffine(self.occluded, np.float32([[1, 0, 25], [0, 1, 25]]), (1200, 1000), borderValue=255)
        self.assertFalse(self.accepts(moved))
