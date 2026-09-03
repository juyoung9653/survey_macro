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

    def test_average_sheet_uses_formulas_linked_to_result_cells(self):
        results = [
            {
                "파일명": "A.pdf",
                "페이지": "A_1p",
                "점수": "5",
                "만족도": "좋음",
            },
            {
                "파일명": "A.pdf",
                "페이지": "A_2p",
                "점수": "3",
                "만족도": "나쁨",
            },
            {
                "파일명": "B.pdf",
                "페이지": "B_1p",
                "점수": "1",
                "만족도": "검수필요(좋음/나쁨)",
            },
        ]
        config = TemplatePreset(
            page_count=1,
            reverse_numbering=False,
            fields=[
                Field(
                    name="점수",
                    boxes=[Box(0, index * 20, 10, 12, 12) for index in range(5)],
                    value_map=["", "", "", "", ""],
                    show_average=True,
                ),
                Field(
                    name="만족도",
                    boxes=[Box(0, 10, 40, 12, 12), Box(0, 40, 40, 12, 12)],
                    value_map=["나쁨", "좋음"],
                    show_average=True,
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "averages.xlsx"
            self.assertTrue(export_to_excel(results, config, str(path)))
            workbook = openpyxl.load_workbook(path, data_only=False)

        average_sheet = workbook["평균"]
        self.assertEqual(
            average_sheet["B2"].value,
            "=IFERROR(AVERAGE('결과'!$C$2:$C$4),0)",
        )
        self.assertEqual(
            average_sheet["C2"].value,
            '=IFERROR(AVERAGEIFS(\'결과\'!$C$2:$C$4,\'결과\'!$A$2:$A$4,"A.pdf"),0)',
        )
        self.assertTrue(average_sheet["B3"].value.startswith("=IFERROR(SUMPRODUCT("))
        self.assertIn(
            'MATCH(\'결과\'!$D$2:$D$4,{"나쁨","좋음"},0)',
            average_sheet["B3"].value,
        )
        self.assertIn(
            '--(\'결과\'!$A$2:$A$4="B.pdf")',
            average_sheet["D3"].value,
        )
        self.assertEqual(average_sheet["B2"].number_format, "0.##")
        self.assertEqual(workbook.calculation.calcMode, "auto")


if __name__ == "__main__":
    unittest.main()
