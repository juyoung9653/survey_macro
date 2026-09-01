import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QInputDialog,
    QMessageBox,
)

from src.localization import install_korean_translations
from src.models import Field, TemplatePreset
from src.ui import MainWindow, ValueMappingDialog


_APP = QApplication.instance() or QApplication([])


class _ProgressStub:
    def __init__(self):
        self.closed = False

    def setValue(self, _value):
        pass

    def setLabelText(self, _message):
        pass

    def close(self):
        self.closed = True


class UiSafetyTests(unittest.TestCase):
    def test_standard_dialog_buttons_are_korean(self):
        install_korean_translations(_APP)
        expected_text = {
            QMessageBox.StandardButton.Ok: "확인",
            QMessageBox.StandardButton.Save: "저장",
            QMessageBox.StandardButton.SaveAll: "모두 저장",
            QMessageBox.StandardButton.Open: "열기",
            QMessageBox.StandardButton.Yes: "예",
            QMessageBox.StandardButton.YesToAll: "모두 예",
            QMessageBox.StandardButton.No: "아니요",
            QMessageBox.StandardButton.NoToAll: "모두 아니요",
            QMessageBox.StandardButton.Abort: "중단",
            QMessageBox.StandardButton.Retry: "다시 시도",
            QMessageBox.StandardButton.Ignore: "무시",
            QMessageBox.StandardButton.Close: "닫기",
            QMessageBox.StandardButton.Cancel: "취소",
            QMessageBox.StandardButton.Discard: "저장 안 함",
            QMessageBox.StandardButton.Help: "도움말",
            QMessageBox.StandardButton.Apply: "적용",
            QMessageBox.StandardButton.Reset: "초기화",
            QMessageBox.StandardButton.RestoreDefaults: "기본값 복원",
        }

        for standard_button, expected in expected_text.items():
            with self.subTest(button=standard_button.name):
                message_box = QMessageBox(
                    standardButtons=standard_button,
                )
                self.assertEqual(message_box.button(standard_button).text(), expected)
                message_box.close()

    def test_input_and_file_dialog_actions_are_korean(self):
        install_korean_translations(_APP)

        input_dialog = QInputDialog()
        input_dialog.setLabelText("이름")
        input_dialog.show()
        file_dialog = QFileDialog()
        file_dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        file_dialog.show()
        _APP.processEvents()

        input_buttons = input_dialog.findChild(QDialogButtonBox)
        file_buttons = file_dialog.findChild(QDialogButtonBox)
        self.assertIsNotNone(input_buttons)
        self.assertIsNotNone(file_buttons)
        self.assertEqual(
            input_buttons.button(QDialogButtonBox.StandardButton.Ok).text(), "확인"
        )
        self.assertEqual(
            input_buttons.button(QDialogButtonBox.StandardButton.Cancel).text(), "취소"
        )
        self.assertTrue(
            file_buttons.button(QDialogButtonBox.StandardButton.Open)
            .text()
            .startswith("열기")
        )
        self.assertEqual(
            file_buttons.button(QDialogButtonBox.StandardButton.Cancel).text(), "취소"
        )
        input_dialog.close()
        file_dialog.close()

    def test_mapping_dialog_rejects_duplicate_question_names(self):
        fields = [Field(name="Q1"), Field(name="Q2")]
        dialog = ValueMappingDialog(None, fields, reverse_numbering=False)
        dialog.working_names = ["Q1", "q1"]

        with patch("src.ui.QMessageBox.warning") as warning:
            dialog.accept()

        warning.assert_called_once()
        self.assertNotEqual(dialog.result(), QDialog.DialogCode.Accepted)
        self.assertEqual([field.name for field in fields], ["Q1", "Q2"])
        dialog.close()

    def test_failed_pdf_load_restores_the_previous_document(self):
        window = MainWindow()
        window.file_paths = ["previous.pdf"]
        window.preset = TemplatePreset(
            page_count=1, fields=[Field(name="기존 문항")]
        )
        window.current_preset_name = "기존 프리셋"
        window._preset_dirty = False
        progress = _ProgressStub()

        with (
            patch(
                "src.ui.QFileDialog.getOpenFileNames",
                return_value=(["broken.pdf"], "PDF Files (*.pdf)"),
            ),
            patch("src.ui.QInputDialog.getInt", return_value=(1, True)),
            patch.object(window, "_show_progress_dialog", return_value=progress),
            patch("src.ui.load_pdf_pages", side_effect=RuntimeError("읽기 오류")),
            patch("src.ui.QMessageBox.critical") as critical,
        ):
            loaded = window.load_pdf()

        self.assertFalse(loaded)
        self.assertEqual(window.file_paths, ["previous.pdf"])
        self.assertEqual([field.name for field in window.preset.fields], ["기존 문항"])
        self.assertEqual(window.current_preset_name, "기존 프리셋")
        self.assertTrue(progress.closed)
        self.assertIn("읽기 오류", critical.call_args.args[2])
        window.close()

    def test_unsaved_change_prompt_honors_cancel_discard_and_save(self):
        window = MainWindow()
        window._preset_dirty = True

        with patch(
            "src.ui.QMessageBox.warning",
            return_value=QMessageBox.StandardButton.Cancel,
        ):
            self.assertFalse(window._confirm_save_or_discard_changes("이동합니다."))

        with patch(
            "src.ui.QMessageBox.warning",
            return_value=QMessageBox.StandardButton.Discard,
        ):
            self.assertTrue(window._confirm_save_or_discard_changes("이동합니다."))

        with (
            patch(
                "src.ui.QMessageBox.warning",
                return_value=QMessageBox.StandardButton.Save,
            ),
            patch.object(window, "save_preset", return_value=True) as save,
        ):
            self.assertTrue(window._confirm_save_or_discard_changes("이동합니다."))
        save.assert_called_once_with()
        window._preset_dirty = False
        window.close()

    def test_save_as_requires_confirmation_before_overwriting(self):
        window = MainWindow()
        with tempfile.TemporaryDirectory() as temp_dir:
            window.preset_dir = Path(temp_dir)
            existing = window.preset_dir / "기존.json"
            existing.write_text("keep", encoding="utf-8")
            with (
                patch(
                    "src.ui.QInputDialog.getText", return_value=("기존", True)
                ),
                patch(
                    "src.ui.QMessageBox.question",
                    return_value=QMessageBox.StandardButton.No,
                ),
            ):
                saved = window.save_preset_as()

            self.assertFalse(saved)
            self.assertEqual(existing.read_text(encoding="utf-8"), "keep")
        window.close()

    def test_primary_convenience_buttons_are_visible(self):
        window = MainWindow()

        self.assertEqual(window.save_preset_btn.text(), "프리셋 저장")
        self.assertEqual(window.open_results_btn.text(), "결과 폴더")
        self.assertEqual(window.help_btn.text(), "도움말")
        self.assertIn("#7E57C2", window.value_map_btn.styleSheet())
        self.assertLessEqual(window.centralWidget().sizeHint().width(), 1280)
        window.close()


if __name__ == "__main__":
    unittest.main()
