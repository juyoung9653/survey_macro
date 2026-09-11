import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import fitz
import numpy as np

from src.processor import generate_ui_templates_multi


class UiMultiDocumentSetupTests(unittest.TestCase):
    def make_pdf(self, path, pages=1):
        with fitz.open() as document:
            for _ in range(pages):
                document.new_page()
            document.save(path)

    def test_other_documents_do_not_move_the_first_document_editing_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [str(Path(folder) / name) for name in ('first.pdf', 'second.pdf', 'third.pdf')]
            for path in paths:
                self.make_pdf(path)
            expected = {0: np.full((30, 20, 3), 255, np.uint8)}
            progress = Mock()
            with patch('src.processor.generate_ui_templates', return_value=expected) as generate:
                actual = generate_ui_templates_multi(paths, 1, -1, 0.0, progress_cb=progress, page_fine_angles=[0.2])
            self.assertIs(actual, expected)
            generate.assert_called_once_with(paths[0], 1, -1, 0.0, progress_cb=progress, page_fine_angles=[0.2])

    def test_unreadable_second_document_is_reported_instead_of_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            first = str(Path(folder) / 'first.pdf')
            second = str(Path(folder) / 'missing.pdf')
            self.make_pdf(first)
            with patch('src.processor.generate_ui_templates') as generate:
                with self.assertRaisesRegex(ValueError, 'missing.pdf.*파일 확인 실패'):
                    generate_ui_templates_multi([first, second], 1, -1, 0.0)
                generate.assert_not_called()

    def test_incomplete_survey_explains_required_and_available_pages(self):
        with tempfile.TemporaryDirectory() as folder:
            first = str(Path(folder) / 'first.pdf')
            second = str(Path(folder) / 'short.pdf')
            self.make_pdf(first, 2)
            self.make_pdf(second)
            with self.assertRaisesRegex(ValueError, 'short.pdf.*2쪽.*1쪽'):
                generate_ui_templates_multi([first, second], 2, -1, 0.0)

    def test_failed_first_template_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as folder:
            first = str(Path(folder) / 'first.pdf')
            self.make_pdf(first)
            with patch('src.processor.generate_ui_templates', return_value={}):
                with self.assertRaisesRegex(ValueError, 'first.pdf.*양식을 만들지 못했습니다'):
                    generate_ui_templates_multi([first], 1, -1, 0.0)
