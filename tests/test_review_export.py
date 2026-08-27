import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import openpyxl

from src.export import export_to_excel
from src.models import Box, Field, TemplatePreset


class ReviewExportTests(unittest.TestCase):
    def test_ambiguous_answer_gets_a_review_sheet_and_counts_as_nonresponse(self):
        success, encoded = cv2.imencode(
            ".png", np.full((80, 240), 245, np.uint8)
        )
        self.assertTrue(success)
        results = [
            {
                "파일명": "sample",
                "페이지": "sample_1p",
                "Q": "검수필요(A/B)",
                "__review_items__": [
                    {
                        "파일명": "sample",
                        "페이지": "sample_1p",
                        "문항": "Q",
                        "후보": "A/B",
                        "사유": "두 표식 확인 필요",
                        "이미지": encoded.tobytes(),
                    }
                ],
            }
        ]
        config = TemplatePreset(
            page_count=1,
            reverse_numbering=False,
            fields=[
                Field(
                    name="Q",
                    boxes=[Box(0, 10, 10, 12, 12), Box(0, 40, 10, 12, 12)],
                    value_map=["A", "B"],
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "result.xlsx"
            self.assertTrue(export_to_excel(results, config, str(path)))
            workbook = openpyxl.load_workbook(path, data_only=False)

        self.assertEqual(
            tuple(cell.value for cell in workbook["결과"][1]),
            ("파일명", "페이지", "Q"),
        )
        self.assertEqual(workbook["검수필요"]["D2"].value, "A/B")
        self.assertEqual(len(workbook["검수필요"]._images), 1)
        self.assertIn("검수필요*", workbook["전체 통계"]["D4"].value)


if __name__ == "__main__":
    unittest.main()
