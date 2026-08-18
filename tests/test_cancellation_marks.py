import unittest

import cv2
import numpy as np

from src.models import Box, Field, TemplatePreset
from src.processor import (
    _cancellation_runner_up_index,
    process_survey_data,
)


class CancellationMarkTests(unittest.TestCase):
    def setUp(self):
        self.boxes = [Box(0, 10 + index * 180, 10, 180, 140) for index in range(5)]

    @staticmethod
    def _draw_check(mask: np.ndarray, box: Box, thickness: int = 6):
        cv2.line(
            mask,
            (box.x + 35, box.y + 75),
            (box.x + 65, box.y + 105),
            255,
            thickness,
        )
        cv2.line(
            mask,
            (box.x + 65, box.y + 105),
            (box.x + 135, box.y + 35),
            255,
            thickness,
        )

    @staticmethod
    def _draw_cancellation(mask: np.ndarray, box: Box):
        for offset in (-18, -9, 0, 9, 18):
            cv2.line(
                mask,
                (box.x + 20, box.y + 55 + offset),
                (box.x + 155, box.y + 85 + offset),
                255,
                6,
            )
            cv2.line(
                mask,
                (box.x + 20, box.y + 85 + offset),
                (box.x + 155, box.y + 55 + offset),
                255,
                6,
            )

    def _scores(self, mask: np.ndarray) -> tuple[list[int], list[int]]:
        inks = []
        areas = []
        for box in self.boxes:
            roi = mask[box.y : box.y + box.h, box.x : box.x + box.w]
            inks.append(cv2.countNonZero(roi))
            areas.append(box.w * box.h)
        return inks, areas

    def test_dense_cancellation_selects_independent_runner_up(self):
        mask = np.zeros((170, 920), np.uint8)
        self._draw_check(mask, self.boxes[2])
        self._draw_cancellation(mask, self.boxes[4])
        inks, areas = self._scores(mask)

        selected = _cancellation_runner_up_index(
            inks, areas, self.boxes, {0: mask}
        )

        self.assertEqual(selected, 2)

    def test_single_dense_mark_does_not_fabricate_runner_up(self):
        mask = np.zeros((170, 920), np.uint8)
        self._draw_cancellation(mask, self.boxes[4])
        inks, areas = self._scores(mask)

        selected = _cancellation_runner_up_index(
            inks, areas, self.boxes, {0: mask}
        )

        self.assertIsNone(selected)

    def test_two_similar_checks_are_not_treated_as_a_correction(self):
        mask = np.zeros((170, 920), np.uint8)
        self._draw_check(mask, self.boxes[1], thickness=6)
        self._draw_check(mask, self.boxes[2], thickness=8)
        inks, areas = self._scores(mask)

        selected = _cancellation_runner_up_index(
            inks, areas, self.boxes, {0: mask}
        )

        self.assertIsNone(selected)

    def test_multiple_independent_checks_are_not_auto_corrected(self):
        mask = np.zeros((170, 920), np.uint8)
        self._draw_check(mask, self.boxes[1])
        self._draw_check(mask, self.boxes[2])
        self._draw_cancellation(mask, self.boxes[4])
        inks, areas = self._scores(mask)

        selected = _cancellation_runner_up_index(
            inks, areas, self.boxes, {0: mask}
        )

        self.assertIsNone(selected)

    def test_process_survey_uses_runner_up_in_large_single_choice_field(self):
        ink_mask = np.zeros((170, 920), np.uint8)
        self._draw_check(ink_mask, self.boxes[2])
        self._draw_cancellation(ink_mask, self.boxes[4])
        template = np.full_like(ink_mask, 255)
        page = template.copy()
        page[ink_mask > 0] = 0
        config = TemplatePreset(
            page_count=1,
            reverse_numbering=True,
            template_dilate_pct=0.0,
            fields=[Field(name="Q", boxes=self.boxes)],
        )

        row, _, _, annotations, _, _ = process_survey_data(
            {
                "fname": "sample",
                "row_title": "sample_1p",
                "gray_pages": {0: page},
            },
            config,
            {0: template},
        )

        self.assertEqual(row["Q"], "3")
        self.assertEqual(
            [annotation[-1] for annotation in annotations[0]],
            [False, False, True, False, False],
        )


if __name__ == "__main__":
    unittest.main()
