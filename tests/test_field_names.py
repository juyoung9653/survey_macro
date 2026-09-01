import tempfile
import unittest
from pathlib import Path

from src.models import Field, TemplatePreset, validate_field_names
from src.processor import run_analysis


class FieldNameSafetyTests(unittest.TestCase):
    def test_names_must_be_unique_ignoring_case_and_spaces(self):
        with self.assertRaisesRegex(ValueError, "중복된 이름: 만족도"):
            validate_field_names([" 만족도 ", "만족도"])

        with self.assertRaisesRegex(ValueError, "중복된 이름: Q1"):
            validate_field_names(["Q1", "q1"])

    def test_result_metadata_and_hidden_names_are_reserved(self):
        for name in ("파일명", "페이지", "__검수"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "쓸 수 없습니다"):
                    validate_field_names([name])

    def test_analysis_rejects_duplicate_names_before_creating_outputs(self):
        config = TemplatePreset(
            fields=[Field(name="Q1"), Field(name="q1")]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "중복된 이름"):
                run_analysis(
                    ["missing.pdf"],
                    [],
                    config,
                    output_base_dir=temp_dir,
                )
            self.assertFalse((Path(temp_dir) / "결과").exists())


if __name__ == "__main__":
    unittest.main()
