import unittest

import cv2
import numpy as np

from src.models import Box, Field, TemplatePreset
from src.processor import remap_preset_to_detected_layout


def _layout(rows: int, offset_x: int = 0):
    image = np.full((700, 900), 255, np.uint8)
    demographics = []
    grid = []
    for column in range(4):
        x = 70 + column * 55
        cv2.rectangle(image, (x, 65), (x + 26, 91), 0, 2)
        demographics.append(Box(0, x, 65, 26, 26))
    for row in range(rows):
        for column in range(5):
            x = 280 + column * 100 + offset_x
            y = 230 + row * 72
            cv2.rectangle(image, (x, y), (x + 82, y + 54), 0, 2)
            grid.append(Box(0, x, y, 82, 54))
    return image, demographics, grid


class IndependentFileLayoutTests(unittest.TestCase):
    def test_local_template_remap_keeps_names_values_and_comments(self):
        canonical, demographics, grid = _layout(5)
        local, _local_demographics, local_grid = _layout(5, offset_x=24)
        config = TemplatePreset(
            page_count=1,
            fields=[
                Field(name="성별", boxes=demographics, value_map=["여", "남", "기타", ""]),
                Field(name="Q1", boxes=grid, value_map=[str(i) for i in range(25)]),
                Field(name="의견", boxes=[Box(0, 90, 620, 300, 40)], is_comment=True),
            ],
        )

        remap = remap_preset_to_detected_layout(
            config, {0: local}, source_templates={0: canonical}
        )

        self.assertTrue(remap.accepted)
        self.assertTrue(remap.compatible)
        self.assertEqual([field.name for field in remap.config.fields], ["성별", "Q1", "의견"])
        self.assertEqual(remap.config.fields[1].value_map, config.fields[1].value_map)
        self.assertTrue(remap.config.fields[2].is_comment)
        self.assertGreater(
            remap.config.fields[1].boxes[0].x,
            remap.config.fields[0].boxes[0].x + 150,
        )

    def test_incompatible_local_row_count_is_rejected(self):
        canonical, demographics, grid = _layout(5)
        incompatible, _demographics, _grid = _layout(3, offset_x=24)
        config = TemplatePreset(
            page_count=1,
            fields=[Field(name="Q1", boxes=grid)],
        )

        remap = remap_preset_to_detected_layout(
            config, {0: incompatible}, source_templates={0: canonical}
        )

        self.assertFalse(remap.accepted)
        self.assertFalse(remap.compatible)


if __name__ == "__main__":
    unittest.main()
