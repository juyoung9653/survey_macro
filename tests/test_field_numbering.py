import unittest

import numpy as np

from src.models import Box, Field, TemplatePreset
from src.processor import _label_number, process_survey_data


def _field(values, reverse_numbering=None):
    return Field(
        "Q1",
        [Box(0, 10 * index, 0, 8, 8) for index in range(len(values))],
        values,
        reverse_numbering=reverse_numbering,
    )


def _analysis_values(field, preset_reverse):
    """The same label-index lookup used by processor scoring."""
    total = len(field.boxes)
    reverse = field.effective_reverse_numbering(preset_reverse)
    return [
        field.value_map[_label_number(total, index, reverse) - 1]
        for index in range(1, total + 1)
    ]


class FieldNumberingTests(unittest.TestCase):
    def test_legacy_none_inherits_both_global_directions(self):
        field = _field(["one", "two", "three"])

        self.assertIsNone(field.reverse_numbering)
        self.assertTrue(field.effective_reverse_numbering(True))
        self.assertFalse(field.effective_reverse_numbering(False))
        self.assertEqual(_analysis_values(field, True), ["three", "two", "one"])
        self.assertEqual(_analysis_values(field, False), ["one", "two", "three"])

    def test_mixed_fields_override_global_direction_for_labels_and_values(self):
        inherited = _field(["a", "b", "c"])
        forward = _field(["d", "e", "f"], False)
        reverse = _field(["g", "h", "i"], True)
        preset = TemplatePreset(reverse_numbering=True, fields=[inherited, forward, reverse])

        self.assertEqual(_analysis_values(inherited, preset.reverse_numbering), ["c", "b", "a"])
        self.assertEqual(_analysis_values(forward, preset.reverse_numbering), ["d", "e", "f"])
        self.assertEqual(_analysis_values(reverse, False), ["i", "h", "g"])
        self.assertEqual([_label_number(3, i, True) for i in (1, 2, 3)], [3, 2, 1])

    def test_serialization_round_trips_explicit_values_and_legacy_fallback(self):
        legacy = _field(["one", "two"])
        explicit_false = _field(["one", "two"], False)
        explicit_true = _field(["one", "two"], True)

        self.assertNotIn("reverse_numbering", legacy.to_dict())
        self.assertIsNone(Field.from_dict(legacy.to_dict()).reverse_numbering)
        self.assertIsNone(Field.from_dict({**legacy.to_dict(), "reverse_numbering": None}).reverse_numbering)
        self.assertFalse(Field.from_dict(explicit_false.to_dict()).reverse_numbering)
        self.assertTrue(Field.from_dict(explicit_true.to_dict()).reverse_numbering)
        self.assertEqual(explicit_false.to_dict()["value_map"], ["one", "two"])

    def test_process_survey_uses_each_field_direction_for_values_and_annotations(self):
        page = np.full((180, 180), 255, np.uint8)
        first_boxes = [Box(0, 20, 20, 20, 20), Box(0, 70, 20, 20, 20)]
        second_boxes = [Box(0, 20, 100, 20, 20), Box(0, 70, 100, 20, 20)]
        # Mark the left physical choice in both fields.
        page[22:38, 22:38] = 0
        page[102:118, 22:38] = 0
        first = Field("forward", first_boxes, ["A1", "A2"], reverse_numbering=False)
        second = Field("reverse", second_boxes, ["B1", "B2"], reverse_numbering=True)
        config = TemplatePreset(
            page_count=1,
            reverse_numbering=False,
            template_dilate_pct=0.0,
            fields=[first, second],
        )

        row, _, _, annotations, _, _ = process_survey_data(
            {"fname": "sample", "row_title": "sample_1p", "gray_pages": {0: page}},
            config,
            {0: np.full_like(page, 255)},
        )

        self.assertEqual(row["forward"], "A1")
        self.assertEqual(row["reverse"], "B2")
        self.assertEqual([entry[4] for entry in annotations[0]], ["1", "2", "2", "1"])


if __name__ == "__main__":
    unittest.main()
