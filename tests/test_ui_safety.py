import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtCore import QPoint, Qt
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
from src.ui import MainCanvas, MainWindow, ValueMappingDialog


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
        raw_page = np.full((20, 15, 3), 180, np.uint8)
        display_template = np.full((20, 15, 3), 255, np.uint8)
        window.pages = [raw_page]
        window._update_page_size()
        window._set_inferred_display_templates({0: display_template})
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
        self.assertIs(window.pages[0], raw_page)
        self.assertIs(window._inferred_display_templates[0], display_template)
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

    def test_saving_preset_shows_completion_message(self):
        window = MainWindow()
        with tempfile.TemporaryDirectory() as temp_dir:
            window.preset_dir = Path(temp_dir)
            with patch("src.ui.QMessageBox.information") as information:
                saved = window._save_preset_to_name("저장한 프리셋")

        self.assertTrue(saved)
        information.assert_called_once_with(
            window,
            "프리셋 저장 완료",
            "프리셋 '저장한 프리셋'을(를) 저장했습니다.",
        )
        window.close()

    def test_primary_convenience_buttons_are_visible(self):
        window = MainWindow()

        self.assertEqual(window.save_preset_btn.text(), "프리셋 저장")
        self.assertEqual(window.open_results_btn.text(), "결과 폴더")
        self.assertEqual(window.help_btn.text(), "도움말")
        self.assertIn("#7E57C2", window.value_map_btn.styleSheet())
        self.assertLessEqual(window.centralWidget().sizeHint().width(), 1280)
        window.close()

    def test_canvas_uses_inferred_blank_template_without_replacing_source_page(self):
        window = MainWindow()
        raw_page = np.full((40, 30, 3), 180, np.uint8)
        blank_template = np.full((40, 30, 3), 255, np.uint8)
        raw_before = raw_page.copy()
        window.file_paths = ["survey.pdf"]
        window.pages = [raw_page]
        window.preset = TemplatePreset(page_count=1)
        window._update_page_size()

        accepted = window._set_inferred_display_templates({0: blank_template})

        self.assertTrue(accepted)
        self.assertIs(window._canvas_base_page(raw_page, 0), blank_template)
        self.assertIs(window.pages[0], raw_page)
        self.assertTrue(np.array_equal(raw_page, raw_before))
        self.assertEqual(window._analysis_reference_pages, [])
        self.assertIn("화면: 자동 생성 빈 양식", window.document_status_label.text())
        window.close()

    def test_mismatched_display_template_blocks_analysis_instead_of_hiding_error(self):
        window = MainWindow()
        raw_page = np.full((40, 30, 3), 180, np.uint8)
        wrong_size = np.full((41, 30, 3), 255, np.uint8)
        window.file_paths = ["survey.pdf"]
        window.pages = [raw_page]
        window.preset = TemplatePreset(
            page_count=1,
            fields=[Field(name="Q1")],
        )
        window._update_page_size()

        accepted = window._set_inferred_display_templates({0: wrong_size})

        self.assertFalse(accepted)
        self.assertEqual(window._inferred_display_templates, {})
        self.assertIn("크기", window._analysis_validation_error)
        self.assertIn("크기", window.document_status_label.toolTip())
        self.assertIs(window._canvas_base_page(raw_page, 0), raw_page)
        with (
            patch("src.ui.QMessageBox.warning") as warning,
            patch.object(window, "_show_progress_dialog") as show_progress,
        ):
            window.execute_analysis()
        self.assertEqual(warning.call_args.args[1], "분석 실행 불가")
        self.assertIn("크기", warning.call_args.args[2])
        show_progress.assert_not_called()
        window.close()

    def test_checkbox_cache_still_loads_the_inferred_display_template(self):
        window = MainWindow()
        raw_page = np.full((40, 30, 3), 180, np.uint8)
        blank_template = np.full((40, 30, 3), 255, np.uint8)
        window.file_paths = ["survey.pdf"]
        window.pages = [raw_page]
        window.preset = TemplatePreset(page_count=1)
        window._update_page_size()

        with (
            patch(
                "src.ui.generate_ui_templates",
                return_value={0: blank_template},
            ) as generate,
            patch(
                "src.ui.load_checkbox_cache",
                return_value={0: [(4, 5, 8, 8)]},
            ),
            patch("src.ui.save_checkbox_cache") as save_cache,
        ):
            window.auto_detect(mark_dirty=False)

        generate.assert_called_once()
        save_cache.assert_not_called()
        self.assertIs(window._inferred_display_templates[0], blank_template)
        self.assertEqual(len(window.preset.fields), 1)
        window.close()

    def test_draw_buttons_toggle_cancel_without_a_separate_select_button(self):
        window = MainWindow()

        self.assertFalse(hasattr(window, "select_tool_btn"))
        self.assertEqual(window.edit_mode, MainCanvas.MODE_SELECT)
        self.assertFalse(window.draw_box_tool_btn.isChecked())
        self.assertFalse(window.draw_comment_tool_btn.isChecked())

        window.draw_box_tool_btn.click()
        self.assertEqual(window.edit_mode, MainCanvas.MODE_DRAW_BOX)
        self.assertEqual(window.draw_box_tool_btn.text(), "그리기 취소")

        window.draw_comment_tool_btn.click()
        self.assertEqual(window.edit_mode, MainCanvas.MODE_DRAW_COMMENT)
        self.assertFalse(window.draw_box_tool_btn.isChecked())
        self.assertEqual(window.draw_box_tool_btn.text(), "선택지 추가")
        self.assertEqual(window.draw_comment_tool_btn.text(), "그리기 취소")

        window.draw_comment_tool_btn.click()
        self.assertEqual(window.edit_mode, MainCanvas.MODE_SELECT)
        self.assertFalse(window.draw_comment_tool_btn.isChecked())
        self.assertEqual(
            window.draw_comment_tool_btn.text(), "자유기입 영역 추가"
        )
        window.close()

    def test_successful_draw_returns_to_normal_selection(self):
        parent = SimpleNamespace(
            add_pending_box_from_stitched=Mock(),
            set_edit_mode=Mock(),
        )
        canvas = MainCanvas(parent)
        canvas.mode = MainCanvas.MODE_DRAW_BOX
        canvas.operation = "draw"
        canvas.start_pos = canvas.mapToScene(QPoint(10, 10))
        event = Mock()
        event.button.return_value = Qt.MouseButton.LeftButton
        event.pos.return_value = QPoint(60, 60)

        canvas.mouseReleaseEvent(event)

        parent.add_pending_box_from_stitched.assert_called_once()
        parent.set_edit_mode.assert_called_once_with(MainCanvas.MODE_SELECT)
        event.accept.assert_called_once()
        canvas.close()

    def test_colored_edit_buttons_keep_tooltip_text_readable(self):
        window = MainWindow()

        self.assertIn("QToolTip", window.styleSheet())
        self.assertIn("color: #263238", window.styleSheet())
        for button in (
            window.group_btn,
            window.value_map_btn,
            window.delete_selected_btn,
        ):
            with self.subTest(button=button.text()):
                self.assertTrue(button.toolTip())
                self.assertIn("QPushButton {", button.styleSheet())
        window.close()

    def test_manual_angle_is_hidden_and_applied_only_after_confirmation(self):
        window = MainWindow()
        window.file_paths = ["sample.pdf"]
        window.pages = [object()]
        window.preset.fine_angle = 0.2

        self.assertFalse(hasattr(window, "fine_angle_spin"))
        self.assertEqual(window.auto_deskew_btn.text(), "기울기 다시 맞추기")

        with (
            patch("src.ui.QInputDialog.getDouble", return_value=(0.4, False)),
            patch.object(window, "change_fine_angle") as apply_angle,
        ):
            window.open_manual_fine_angle_dialog()
        apply_angle.assert_not_called()

        with (
            patch("src.ui.QInputDialog.getDouble", return_value=(0.4, True)),
            patch.object(window, "change_fine_angle") as apply_angle,
        ):
            window.open_manual_fine_angle_dialog()
        apply_angle.assert_called_once_with(0.4)
        window.close()


if __name__ == "__main__":
    unittest.main()
