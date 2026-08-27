import unittest

import cv2
import numpy as np

from src.mark_analysis import (
    _CheckboxHaloInfo,
    _CheckboxInkInfo,
    _checkbox_ambiguous_indices,
    _checkbox_cancellation_runner_up_index,
    _suppress_isolated_weak_checkbox_marks,
)
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

    @staticmethod
    def _checkbox_info(
        box: Box,
        mask: np.ndarray,
        ink: int,
        strength: float,
    ) -> _CheckboxInkInfo:
        return _CheckboxInkInfo(
            ink,
            144,
            box,
            1.0,
            (box.x, box.y, box.x + mask.shape[1], box.y + mask.shape[0]),
            mask,
            mark_strength=strength,
            stroke_span_ratio=1.0,
        )

    @staticmethod
    def _halo_info(box: Box, mask: np.ndarray, ink: int) -> _CheckboxHaloInfo:
        return _CheckboxHaloInfo(
            ink,
            400,
            box,
            (box.x, box.y, box.x + mask.shape[1], box.y + mask.shape[0]),
            mask,
        )

    def test_crossed_out_checkbox_allows_a_moderate_ink_ratio(self):
        boxes = [Box(0, 10, 10, 12, 12), Box(0, 50, 10, 12, 12)]
        crossed = np.zeros((20, 20), np.uint8)
        cv2.line(crossed, (1, 1), (18, 18), 255, 2)
        cv2.line(crossed, (18, 1), (1, 18), 255, 2)
        checked = np.zeros((20, 20), np.uint8)
        cv2.line(checked, (2, 10), (8, 16), 255, 2)
        cv2.line(checked, (8, 16), (18, 3), 255, 2)
        infos = [
            self._checkbox_info(boxes[0], crossed, 100, 0.70),
            self._checkbox_info(boxes[1], checked, 70, 0.45),
        ]
        halos = [
            self._halo_info(boxes[0], crossed, 80),
            self._halo_info(boxes[1], checked, 25),
        ]

        self.assertEqual(
            _checkbox_cancellation_runner_up_index(infos, halos, [True, True]),
            1,
        )

    def test_two_credible_linear_marks_are_left_for_manual_review(self):
        boxes = [Box(0, 10, 10, 12, 12), Box(0, 50, 10, 12, 12)]
        first = np.zeros((20, 20), np.uint8)
        second = np.zeros((20, 20), np.uint8)
        cv2.line(first, (1, 10), (18, 10), 255, 2)
        cv2.line(second, (2, 17), (17, 2), 255, 2)
        infos = [
            self._checkbox_info(boxes[0], first, 117, 0.88),
            self._checkbox_info(boxes[1], second, 62, 0.47),
        ]
        halos = [
            self._halo_info(boxes[0], first, 182),
            self._halo_info(boxes[1], second, 77),
        ]

        self.assertIsNone(
            _checkbox_cancellation_runner_up_index(infos, halos, [True, True])
        )
        self.assertEqual(
            _checkbox_ambiguous_indices(infos, halos, [True, True]),
            [0, 1],
        )

    def test_tiny_speck_is_removed_only_beside_a_strong_duplicate_mark(self):
        boxes = [Box(0, 10, 10, 12, 12), Box(0, 50, 10, 12, 12)]
        speck = np.zeros((12, 12), np.uint8)
        speck[5:7, 5:8] = 255
        checked = np.zeros((12, 12), np.uint8)
        cv2.line(checked, (1, 7), (5, 11), 255, 2)
        cv2.line(checked, (5, 11), (11, 1), 255, 2)
        infos = [
            self._checkbox_info(boxes[0], speck, 6, 0.043),
            self._checkbox_info(boxes[1], checked, 65, 0.45),
        ]
        infos[0].stroke_span_ratio = 0.25
        halos = [
            self._halo_info(boxes[0], np.zeros_like(speck), 0),
            self._halo_info(boxes[1], checked, 163),
        ]

        self.assertEqual(
            _suppress_isolated_weak_checkbox_marks(
                infos, halos, [True, True]
            ),
            [False, True],
        )

    def test_tiny_mark_is_not_removed_without_a_strong_sibling(self):
        boxes = [Box(0, 10, 10, 12, 12), Box(0, 50, 10, 12, 12)]
        speck = np.zeros((12, 12), np.uint8)
        speck[5:7, 5:8] = 255
        infos = [
            self._checkbox_info(boxes[0], speck, 6, 0.043),
            self._checkbox_info(boxes[1], speck, 6, 0.043),
        ]
        for info in infos:
            info.stroke_span_ratio = 0.25
        halos = [
            self._halo_info(box, np.zeros_like(speck), 0) for box in boxes
        ]

        self.assertEqual(
            _suppress_isolated_weak_checkbox_marks(
                infos, halos, [True, True]
            ),
            [True, True],
        )

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

    def test_process_survey_uses_runner_up_for_small_checkbox_correction(self):
        template = np.full((220, 400), 255, np.uint8)
        boxes = [Box(0, x, 90, 24, 24) for x in (50, 120, 190)]
        for box in boxes:
            cv2.rectangle(
                template,
                (box.x, box.y),
                (box.x + box.w, box.y + box.h),
                80,
                2,
            )

        page = template.copy()
        cancelled = boxes[0]
        for offset in (2, 8, 14):
            cv2.line(
                page,
                (cancelled.x + 2, cancelled.y + offset),
                (cancelled.x + cancelled.w - 2, cancelled.y + cancelled.h - offset),
                20,
                3,
            )
        intended = boxes[1]
        cv2.line(
            page,
            (intended.x + 4, intended.y + 13),
            (intended.x + 10, intended.y + 19),
            20,
            3,
        )
        cv2.line(
            page,
            (intended.x + 10, intended.y + 19),
            (intended.x + 21, intended.y + 4),
            20,
            3,
        )
        config = TemplatePreset(
            page_count=1,
            reverse_numbering=False,
            template_dilate_pct=0.0,
            fields=[Field(name="Q", boxes=boxes)],
        )

        row, _, _, annotations, _, _ = process_survey_data(
            {
                "fname": "sample",
                "row_title": "sample_1p",
                "gray_pages": {0: page},
            },
            config,
            {0: template},
            trust_checkbox_layout=True,
        )

        self.assertEqual(row["Q"], "2")
        self.assertEqual(
            [annotation[-1] for annotation in annotations[0]],
            [False, True, False],
        )


if __name__ == "__main__":
    unittest.main()
