import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QEvent, QPoint, Qt
from PyQt6.QtGui import QAction, QContextMenuEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QWidget,
)

from src.help_ui import HelpController


_APP = QApplication.instance() or QApplication([])


class ContextHelpTests(unittest.TestCase):
    def setUp(self):
        self.settings_dir = tempfile.TemporaryDirectory()
        self.old_local_app_data = os.environ.get("LOCALAPPDATA")
        os.environ["LOCALAPPDATA"] = self.settings_dir.name
        self.owner = QWidget()
        self.owner.resize(240, 120)
        self.owner.show()
        _APP.processEvents()
        self.controller = HelpController(self.owner)

    def tearDown(self):
        self.owner.close()
        self.owner.deleteLater()
        _APP.processEvents()
        if self.old_local_app_data is None:
            os.environ.pop("LOCALAPPDATA", None)
        else:
            os.environ["LOCALAPPDATA"] = self.old_local_app_data
        self.settings_dir.cleanup()

    def _context_event(self, widget):
        return QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse,
            QPoint(4, 4),
            widget.mapToGlobal(QPoint(4, 4)),
        )

    def test_register_escapes_plain_text_and_help_mode_does_not_click_button(self):
        button = QPushButton("run", self.owner)
        clicked = []
        button.clicked.connect(lambda: clicked.append(True))
        self.controller.register(button, "<Run>", "First & foremost\nsecond")

        self.assertIn("&lt;Run&gt;", button.whatsThis())
        self.assertIn("First &amp; foremost<br>second", button.whatsThis())
        with patch.object(self.controller, "_show_intro"):
            self.controller.enter_mode()
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        _APP.processEvents()
        self.assertEqual(clicked, [])
        from PyQt6.QtWidgets import QWhatsThis

        QWhatsThis.leaveWhatsThisMode()

    def test_menu_action_right_click_shows_help_without_triggering_or_toggling(self):
        menu = QMenu(self.owner)
        action = menu.addAction("Delete")
        action.setCheckable(True)
        triggered = []
        action.triggered.connect(triggered.append)
        self.controller.register_action(menu, action, "Delete", "Removes a box")
        menu.popup(self.owner.mapToGlobal(QPoint(10, 10)))
        _APP.processEvents()

        with patch("src.help_ui.QWhatsThis.showText") as show:
            QTest.mouseClick(
                menu,
                Qt.MouseButton.RightButton,
                pos=menu.actionGeometry(action).center(),
            )
            _APP.processEvents()
        self.assertEqual(show.call_count, 1)
        self.assertFalse(action.isChecked())
        self.assertEqual(triggered, [])
        self.assertIn("Removes a box", action.whatsThis())
        menu.close()

    def test_menu_action_help_popup_survives_right_button_release(self):
        menu = QMenu(self.owner)
        action = menu.addAction("Delete")
        triggered = []
        action.triggered.connect(triggered.append)
        self.controller.register_action(menu, action, "Delete", "Removes a box")
        menu.popup(self.owner.mapToGlobal(QPoint(10, 10)))
        _APP.processEvents()
        position = menu.actionGeometry(action).center()

        QTest.mousePress(menu, Qt.MouseButton.RightButton, pos=position)
        QTest.qWait(100)
        QTest.mouseRelease(menu, Qt.MouseButton.RightButton, pos=position)
        QTest.qWait(100)

        self.assertIsNotNone(QApplication.activePopupWidget())
        self.assertIsNot(QApplication.activePopupWidget(), menu)
        self.assertEqual(triggered, [])
        from PyQt6.QtWidgets import QWhatsThis

        QWhatsThis.hideText()
        menu.close()

    def test_menu_action_context_event_shows_registered_action_help(self):
        menu = QMenu(self.owner)
        action = menu.addAction("Cache")
        self.controller.register_action(menu, action, "Cache", "Clears cache")
        position = menu.actionGeometry(action).center()
        event = QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse, position, menu.mapToGlobal(position)
        )

        with patch("src.help_ui.QWhatsThis.showText") as show:
            self.assertTrue(self.controller.eventFilter(menu, event))
        self.assertIn("Clears cache", show.call_args.args[1])

    def test_intro_checked_setting_persists_for_a_new_controller(self):
        with (
            patch("src.help_ui.QMessageBox.exec"),
            patch("src.help_ui.QCheckBox.isChecked", return_value=True),
        ):
            self.controller._show_intro()

        later = HelpController(self.owner)
        self.assertTrue(later._should_skip_intro())
        with (
            patch.object(later, "_show_intro") as intro,
            patch("src.help_ui.QWhatsThis.enterWhatsThisMode"),
        ):
            later.enter_mode()
        intro.assert_not_called()

    def test_intro_unchecked_setting_is_shown_again(self):
        with (
            patch("src.help_ui.QMessageBox.exec"),
            patch("src.help_ui.QCheckBox.isChecked", return_value=False),
        ):
            self.controller._show_intro()

        later = HelpController(self.owner)
        self.assertFalse(later._should_skip_intro())
        with (
            patch.object(later, "_show_intro") as intro,
            patch("src.help_ui.QWhatsThis.enterWhatsThisMode"),
        ):
            later.enter_mode()
        intro.assert_called_once_with()

    def test_context_menu_shows_help_action_without_clicking_a_button(self):
        button = QPushButton("run", self.owner)
        clicked = []
        button.clicked.connect(lambda: clicked.append(True))
        self.controller.register(button, "Run", "Runs work")

        self.assertTrue(
            self.controller.eventFilter(button, self._context_event(button))
        )
        menu = next(iter(self.controller._open_menus))
        self.assertEqual(menu.actions()[0].text(), "이 항목 도움말")
        self.assertEqual(clicked, [])
        menu.close()
        self.assertNotIn(menu, self.controller._open_menus)

    def test_context_action_displays_registered_help(self):
        button = QPushButton("run", self.owner)
        self.controller.register(button, "Run", "Runs work")

        self.controller.eventFilter(button, self._context_event(button))
        menu = next(iter(self.controller._open_menus))
        with patch("src.help_ui.QWhatsThis.showText") as show:
            menu.actions()[0].trigger()
        show.assert_called_once_with(
            button.mapToGlobal(QPoint(4, 4)), button.whatsThis(), button
        )
        menu.close()

    def test_text_editor_keeps_standard_copy_paste_actions(self):
        editor = QLineEdit("copy this", self.owner)
        self.controller.register(editor, "Input", "Editable text")

        self.assertTrue(
            self.controller.eventFilter(editor, self._context_event(editor))
        )
        menu = next(iter(self.controller._open_menus))
        labels = [action.text().replace("&", "") for action in menu.actions()]
        self.assertTrue(any("Copy" in label or "복사" in label for label in labels))
        self.assertTrue(any("Paste" in label or "붙여넣기" in label for label in labels))
        self.assertEqual(menu.actions()[-1].text(), "이 항목 도움말")
        menu.close()

    def test_spinbox_child_line_edit_keeps_its_standard_menu(self):
        spinbox = QSpinBox(self.owner)
        self.controller.register(spinbox, "Count", "Choose a count")
        editor = spinbox.lineEdit()

        self.assertTrue(
            self.controller.eventFilter(editor, self._context_event(editor))
        )
        menu = next(iter(self.controller._open_menus))
        self.assertEqual(menu.actions()[-1].text(), "이 항목 도움말")
        menu.close()

    def test_disabled_widget_can_still_show_registered_help(self):
        checkbox = QCheckBox("choice", self.owner)
        checkbox.setEnabled(False)
        self.controller.register(checkbox, "Choice", "Unavailable right now")

        with patch("src.help_ui.QWhatsThis.showText") as show:
            self.controller.show_help(checkbox, QPoint(7, 8))
        show.assert_called_once_with(QPoint(7, 8), checkbox.whatsThis(), checkbox)

    def test_custom_menu_is_unaffected_even_when_help_is_registered(self):
        widget = QWidget(self.owner)
        widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        seen = []
        widget.customContextMenuRequested.connect(seen.append)
        self.controller.register(widget, "Canvas", "Existing editing menu")

        event = self._context_event(widget)
        self.assertFalse(self.controller.eventFilter(widget, event))
        self.assertEqual(seen, [])
        self.assertTrue(widget.whatsThis())

    def test_context_menu_false_keeps_whats_this_without_a_help_menu(self):
        button = QPushButton("canvas control", self.owner)
        self.controller.register(button, "Canvas", "Existing editing menu", context_menu=False)

        self.assertFalse(
            self.controller.eventFilter(button, self._context_event(button))
        )
        self.assertTrue(button.whatsThis())

    def test_ignored_owner_close_keeps_context_help_active(self):
        class IgnoreCloseWindow(QWidget):
            def closeEvent(self, event):
                event.ignore()

        ignored_owner = IgnoreCloseWindow()
        ignored_owner.show()
        controller = HelpController(ignored_owner)
        button = QPushButton("run", ignored_owner)
        controller.register(button, "Run", "Runs work")

        self.assertFalse(ignored_owner.close())
        _APP.processEvents()
        self.assertTrue(ignored_owner.isVisible())
        self.assertTrue(controller.eventFilter(button, self._context_event(button)))
        menu = next(iter(controller._open_menus))
        menu.close()
        ignored_owner.close()
        ignored_owner.deleteLater()
        _APP.processEvents()

    def test_hidden_owner_close_detaches_filters_without_waiting_for_hide(self):
        hidden_owner = QWidget()
        controller = HelpController(hidden_owner)
        button = QPushButton("run", hidden_owner)
        controller.register(button, "Run", "Runs work")

        self.assertTrue(hidden_owner.close())
        self.assertEqual(controller._entries, {})
        self.assertEqual(controller._watched_refs, {})
        hidden_owner.deleteLater()
        _APP.processEvents()


if __name__ == "__main__":
    unittest.main()
