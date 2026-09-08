import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QWhatsThis
from src.models import Box, Field
from src.ui import MainWindow, ValueMappingDialog

APP = QApplication.instance() or QApplication([])


class HelpIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.intro = patch('src.help_ui.HelpController._show_intro')
        self.intro.start()
        self.addCleanup(self.intro.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {'LOCALAPPDATA': self.temp.name})
        self.env.start()
        self.window = MainWindow()
        self.window.show()
        APP.processEvents()

    def tearDown(self):
        QWhatsThis.leaveWhatsThisMode()
        QWhatsThis.hideText()
        self.window._preset_dirty = False
        self.window.close()
        self.env.stop()
        self.temp.cleanup()

    def test_help_button_prevents_analysis_or_canvas_edits(self):
        with patch('src.ui.QMessageBox.warning') as warning:
            QTest.mouseClick(self.window.item_help_btn, Qt.MouseButton.LeftButton)
            self.assertTrue(QWhatsThis.inWhatsThisMode())
            QTest.mouseClick(self.window.exec_btn, Qt.MouseButton.LeftButton)
            warning.assert_not_called()
            self.assertIsNone(self.window._analysis_thread)
        QWhatsThis.hideText()
        self.window.help_controller.enter_mode()
        QTest.mouseClick(self.window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(30, 30))
        self.assertIsNone(self.window.canvas.operation)
        self.assertEqual([], self.window.pending_boxes)

    def test_guide_sample_is_automatically_grouped_with_and_without_cache(self):
        from scripts.capture_manual import example_page
        from pathlib import Path
        from PyQt6.QtGui import QFontDatabase
        font = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / 'malgun.ttf'
        if not font.exists():
            self.skipTest('Guide capture requires the Windows Malgun Gothic font')
        self.assertGreaterEqual(QFontDatabase.addApplicationFont(str(font)), 0)
        page, sample_boxes = example_page()
        cached = {0: [(box.x, box.y, box.w, box.h) for box in sample_boxes]}
        for cache in (None, cached):
            with self.subTest(cached=cache is not None):
                self.window.file_paths = ['fictional-guide.pdf']
                self.window.pages = [page]
                self.window._update_page_size()
                with patch('src.ui.load_checkbox_cache', return_value=cache), patch('src.ui.save_checkbox_cache'):
                    self.window.auto_detect(prebuilt_templates={0: page}, mark_dirty=False)
                self.assertEqual(['Q1'], [field.name for field in self.window.preset.fields])
                self.assertEqual(3, len(self.window.preset.fields[0].boxes))
                self.assertEqual([], self.window.pending_boxes)
        self.assertIn('자동', self.window.load_pdf_btn.whatsThis())
        self.assertIn('누르지 않아도', self.window.group_btn.whatsThis())

    def test_mapping_help_does_not_toggle_or_save_checkboxes(self):
        field = Field('만족도', boxes=[Box(0, 10, 10, 20, 20)])
        dialog = ValueMappingDialog(self.window, [field], False)
        dialog.show()
        APP.processEvents()
        before = dialog.duplicate_check.isChecked()
        dialog.help_controller.enter_mode()
        QTest.mouseClick(dialog.duplicate_check, Qt.MouseButton.LeftButton)
        self.assertEqual(before, dialog.duplicate_check.isChecked())
        self.assertFalse(field.allow_duplicates)
        self.assertTrue(dialog.table.whatsThis())
        self.assertTrue(dialog.group_name_edit.whatsThis())
        dialog.reject()

    def test_average_guide_renames_question_without_entering_choice_names(self):
        fields = [Field(f'Q{number}', boxes=[Box(0, x * 40, 10, 20, 20) for x in range(5)])
                  for number in range(1, 6)]
        dialog = ValueMappingDialog(self.window, fields, True)
        self.assertIn('모두 비우고', dialog.average_hint.text())
        self.assertIn('문항 이름은 바꿔도', dialog.average_hint.text())
        for index in range(5):
            dialog.group_combo.setCurrentIndex(index)
            dialog.group_name_edit.setText(f'만족도 {index + 1}')
            self.assertTrue(dialog.average_check.isChecked())
            self.assertTrue(all(not dialog.table.item(row, 1).text() for row in range(5)))
        dialog.accept()
        self.assertTrue(all(field.show_average for field in fields))
        self.assertTrue(all(not any(field.value_map) for field in fields))
        self.assertEqual([f'만족도 {number}' for number in range(1, 6)], [field.name for field in fields])

    def test_region_help_keeps_existing_edit_menu_and_does_not_delete(self):
        box = Box(0, 10, 10, 20, 20)
        self.window.pending_boxes = [box]
        labels = []
        def choose_help(menu, *_):
            labels.extend(action.text() for action in menu.actions())
            return next(action for action in menu.actions() if action.text() == '이 영역 도움말')
        with patch('src.ui.QMenu.exec', new=choose_help), patch.object(self.window.help_controller, 'show_help') as show:
            self.window.show_box_context_menu(QPoint(10, 10), box)
            show.assert_called_once()
        self.assertIn('선택한 영역 삭제', labels)
        self.assertIn('선택한 영역을 새 문항으로 묶기', labels)
        self.assertEqual([box], self.window.pending_boxes)

    def test_every_file_menu_action_has_help_and_titles_have_no_step_numbers(self):
        self.assertEqual(self.window.help_btn.geometry().center().y(),
                         self.window.item_help_btn.geometry().center().y())
        self.assertGreater(self.window.item_help_btn.x(), self.window.help_btn.x())
        def check_menu(menu):
            for action in menu.actions():
                if action.isSeparator():
                    continue
                self.assertTrue(action.whatsThis(), action.text())
                if action.menu():
                    check_menu(action.menu())
        check_menu(self.window.file_menu_btn.menu())
        for button in (self.window.load_pdf_btn, self.window.value_map_btn,
                       self.window.exec_btn, self.window.open_results_btn):
            self.assertNotRegex(button.whatsThis(), r'<h3>\d+\.')

    def test_analysis_toolbar_button_order(self):
        buttons = [self.window.undo_btn, self.window.redo_btn,
                   self.window.exec_btn, self.window.open_results_btn]
        self.assertEqual(1, len({button.geometry().center().y() for button in buttons}))
        self.assertEqual(sorted(button.x() for button in buttons), [button.x() for button in buttons])
        self.assertLess(buttons[0].y(), self.window.help_btn.y())

    def test_duplicate_setting_is_only_in_choice_settings_dialog(self):
        settings = next(action.menu() for action in self.window.file_menu_btn.menu().actions()
                        if action.text() == '기울기')
        self.assertNotIn('중복 허용', [action.text() for action in settings.actions()])
        self.assertIn(self.window.manual_fine_angle_action, settings.actions())
        field = Field('복수 응답', boxes=[Box(0, 10, 10, 20, 20)])
        dialog = ValueMappingDialog(self.window, [field], False)
        self.assertIn('중복 허용', dialog.duplicate_check.text())
        dialog.duplicate_check.setChecked(True)
        dialog.accept()
        self.assertTrue(field.allow_duplicates)

    def test_per_question_direction_preserves_names_and_cancel(self):
        fields = [Field('Q1', [Box(0, x * 30, 0, 20, 20) for x in range(3)], ['A', 'B', 'C']),
                  Field('Q2', [Box(0, x * 30, 40, 20, 20) for x in range(3)])]
        dialog = ValueMappingDialog(self.window, fields, False)
        dialog.direction_combo.setCurrentIndex(1)
        self.assertEqual(['A', 'B', 'C'], [dialog.table.item(i, 1).text() for i in range(3)])
        self.assertIn('3 → 2 → 1', dialog.reverse_label.text())
        dialog.group_combo.setCurrentIndex(1)
        self.assertEqual(0, dialog.direction_combo.currentIndex())
        self.assertIn('1 → 2 → 3', dialog.reverse_label.text())
        dialog.reject()
        self.assertIsNone(fields[0].reverse_numbering)
        self.assertEqual(['A', 'B', 'C'], fields[0].value_map)

    def test_per_question_direction_save_and_undo(self):
        field = Field('Q1', [Box(0, x * 30, 0, 20, 20) for x in range(3)], ['A', 'B', 'C'])
        self.window.preset.fields = [field]
        before = self.window._capture_edit_state()
        dialog = ValueMappingDialog(self.window, [field], False)
        dialog.direction_combo.setCurrentIndex(1)
        dialog.accept()
        self.assertTrue(field.reverse_numbering)
        self.assertEqual(['C', 'B', 'A'], field.value_map)
        self.window._commit_edit(before, '문항 번호 방향')
        self.window.undo_edit()
        self.assertIsNone(self.window.preset.fields[0].reverse_numbering)
        self.assertEqual(['A', 'B', 'C'], self.window.preset.fields[0].value_map)
        self.window.redo_edit()
        self.assertTrue(self.window.preset.fields[0].reverse_numbering)

    def test_batch_copies_direction_and_unchecked_average_stays_unchecked(self):
        fields = [Field(f'Q{i}', [Box(0, x * 30, 0, 20, 20) for x in range(3)]) for i in range(3)]
        dialog = ValueMappingDialog(self.window, fields, False)
        dialog.average_check.setChecked(False)
        dialog.direction_combo.setCurrentIndex(1)
        self.assertFalse(dialog.average_check.isChecked())
        with patch('src.ui._BatchApplyDialog') as batch:
            batch.return_value.exec.return_value = dialog.DialogCode.Accepted
            batch.return_value.selected_indices = [1]
            dialog._batch_apply()
        dialog.accept()
        self.assertEqual([True, True, False], [field.reverse_numbering for field in fields])
        self.assertEqual([False, False, True], [field.show_average for field in fields])
        reopened = ValueMappingDialog(self.window, fields, True)
        self.assertFalse(reopened.average_check.isChecked())
        reopened.reject()

    def test_right_click_file_actions_does_not_trigger_or_toggle(self):
        menus = [entry[0] for entry in self.window._menu_help_entries
                 if entry[1] is self.window.rotation_actions[0]]
        page_menu = menus[0]
        file_menu = self.window.file_menu_btn.menu()
        cache_action = next(action for action in file_menu.actions() if action.text() == '캐시 삭제')
        for menu, action in ((page_menu, self.window.rotation_actions[0]), (file_menu, cache_action)):
            with self.subTest(action=action.text()), patch('src.help_ui.QWhatsThis.showText') as show:
                triggered = []
                action.triggered.connect(triggered.append)
                before = action.isChecked()
                menu.popup(self.window.mapToGlobal(QPoint(40, 40)))
                APP.processEvents()
                QTest.mouseClick(menu, Qt.MouseButton.RightButton, pos=menu.actionGeometry(action).center())
                APP.processEvents()
                self.assertTrue(show.called)
                self.assertEqual([], triggered)
                self.assertEqual(before, action.isChecked())
                action.triggered.disconnect(triggered.append)
                menu.close()


if __name__ == '__main__':
    unittest.main()
