"""Small, scoped contextual-help support for Qt widgets."""

from __future__ import annotations

from html import escape
import os
from pathlib import Path
from weakref import ref

from PyQt6 import sip
from PyQt6.QtCore import QEvent, QObject, QPoint, QSettings, QTimer, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QTextEdit,
    QWhatsThis,
    QWidget,
)


_HELP_ACTION_TEXT = "이 항목 도움말"


class HelpController(QObject):
    """Attach Qt's What&apos;s This help to widgets owned by one window.

    The controller deliberately filters only widgets explicitly registered with it.
    This keeps help-mode clicks and context menus local to the owning window.
    """

    def __init__(self, owner: QWidget):
        super().__init__(owner)
        # Qt owns this controller.  Keep only weak/id bookkeeping for the
        # widgets it watches so window teardown cannot retain dead wrappers.
        self._owner_ref = ref(owner)
        self._entries: dict[int, bool] = {}
        self._spin_line_edit_ids: set[int] = set()
        self._watched_refs: dict[int, ref[QWidget]] = {}
        self._open_menus: set[QMenu] = set()
        self._action_entries: dict[int, str] = {}
        self._action_menu_ids: set[int] = set()
        self._suppressed_action_context_id: int | None = None
        self._action_help_shown = False
        self._close_pending = False
        self._settings = self._create_settings()
        owner.installEventFilter(self)

    def register(
        self,
        widget: QWidget,
        title: str,
        body: str,
        context_menu: bool = True,
    ) -> None:
        """Register plain-text help for *widget*.

        ``context_menu=False`` leaves any menu behaviour entirely to the widget,
        while still enabling Qt's built-in What&apos;s This mode.
        """
        widget.setWhatsThis(self._format_help(title, body))
        self._watch(widget, context_menu)

        # A spin box delegates its text-editing context menu to its child line
        # edit.  Watching that child preserves copy/paste and exposes the same
        # help without requiring every caller to know this implementation detail.
        if isinstance(widget, QAbstractSpinBox) and widget.lineEdit() is not None:
            line_edit = widget.lineEdit()
            line_edit.setWhatsThis(widget.whatsThis())
            self._spin_line_edit_ids.add(id(line_edit))
            self._watch(line_edit, context_menu)

    def register_action(
        self, menu: QMenu, action: QAction, title: str, body: str
    ) -> None:
        """Register help for a menu action without changing its activation."""
        text = self._format_help(title, body)
        action.setWhatsThis(text)
        self._action_entries[id(action)] = text
        if id(menu) not in self._action_menu_ids:
            self._action_menu_ids.add(id(menu))
            self._watched_refs[id(menu)] = ref(menu)
            menu.installEventFilter(self)

    def enter_mode(self) -> None:
        """Let Qt select a registered widget without activating that widget."""
        if not self._should_skip_intro():
            self._show_intro()
        QWhatsThis.enterWhatsThisMode()

    def _show_intro(self) -> None:
        """Explain both help paths before entering Qt's What's This mode."""
        message = QMessageBox(self._owner_ref())
        message.setWindowTitle("항목 도움말")
        message.setText(
            "언제든 항목을 우클릭하면 설명을 볼 수 있습니다.\n"
            "파일 메뉴의 항목도 우클릭할 수 있습니다.\n\n"
            "이제 궁금한 항목을 왼쪽 클릭하면 설명만 표시됩니다. "
            "Esc를 누르면 도움말 모드를 끝냅니다."
        )
        message.setStandardButtons(QMessageBox.StandardButton.Ok)
        dont_show = QCheckBox("다시 보지 않기", message)
        message.setCheckBox(dont_show)
        message.exec()
        skip_intro = dont_show.isChecked()
        message.deleteLater()
        if skip_intro:
            self._set_skip_intro()

    @staticmethod
    def _create_settings() -> QSettings | None:
        try:
            local_app_data = os.environ.get("LOCALAPPDATA")
            if not local_app_data:
                return None
            settings_dir = Path(local_app_data) / "CheckFinder"
            settings_dir.mkdir(parents=True, exist_ok=True)
            return QSettings(
                str(settings_dir / "help.ini"), QSettings.Format.IniFormat
            )
        except (OSError, RuntimeError):
            return None

    def _should_skip_intro(self) -> bool:
        if self._settings is None:
            return False
        try:
            value = self._settings.value("help/skip_intro", False)
            return str(value).lower() in {"1", "true", "yes"}
        except RuntimeError:
            return False

    def _set_skip_intro(self) -> None:
        if self._settings is None:
            return
        try:
            self._settings.setValue("help/skip_intro", True)
            self._settings.sync()
        except RuntimeError:
            pass

    def show_help(self, widget: QWidget, global_pos: QPoint | None = None) -> None:
        """Show a widget's registered help at ``global_pos``."""
        if sip.isdeleted(widget) or id(widget) not in self._entries:
            return
        if global_pos is None:
            global_pos = widget.mapToGlobal(widget.rect().center())
        QWhatsThis.showText(global_pos, widget.whatsThis(), widget)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self._owner_ref() and event.type() == QEvent.Type.Close:
            owner = self._owner_ref()
            # A hidden widget has no subsequent Hide event.  This is the normal
            # path for short-lived/test windows, so detach now rather than
            # leaving native event filters installed until interpreter exit.
            if owner is not None and not sip.isdeleted(owner) and not owner.isVisible():
                self._detach()
                return super().eventFilter(watched, event)
            # This runs before QWidget.closeEvent(), which may ignore the
            # request (for example while analysis is running).  A successful
            # close emits Hide immediately afterwards; only then is teardown
            # safe.  The queued check clears a cancelled close request.
            self._close_pending = True
            QTimer.singleShot(0, self._clear_cancelled_close)
            return super().eventFilter(watched, event)
        if (
            isinstance(watched, QMenu)
            and id(watched) in self._action_menu_ids
            and self._handle_action_help_event(watched, event)
        ):
            return True
        if (
            watched is self._owner_ref()
            and event.type() == QEvent.Type.Hide
            and self._close_pending
        ):
            self._detach()
            return super().eventFilter(watched, event)
        if (
            not isinstance(watched, QWidget)
            or id(watched) not in self._entries
            or not self._belongs_to_owner(watched)
        ):
            return super().eventFilter(watched, event)
        if event.type() != QEvent.Type.ContextMenu or not self._entries[id(watched)]:
            return super().eventFilter(watched, event)

        # Qt itself owns What's This mode.  In particular, do not consume mouse
        # events: Qt then handles the help click before a button can activate.
        if QWhatsThis.inWhatsThisMode():
            return super().eventFilter(watched, event)
        # A custom/actions menu belongs to the widget.  Never replace it with a
        # help-only menu; callers can use context_menu=False for an explicit
        # What's This-only registration.
        if (
            watched.contextMenuPolicy() != Qt.ContextMenuPolicy.DefaultContextMenu
            and not self._is_standard_text_editor(watched)
        ):
            return super().eventFilter(watched, event)

        if self._is_standard_text_editor(watched):
            self._show_text_editor_menu(watched, event)
        else:
            self._show_help_menu(watched, event.globalPos())
        event.accept()
        return True

    def _handle_action_help_event(self, menu: QMenu, event: QEvent) -> bool:
        event_type = event.type()
        if event_type not in {
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
            QEvent.Type.ContextMenu,
        }:
            return False
        if event_type != QEvent.Type.ContextMenu and (
            event.button() != Qt.MouseButton.RightButton
        ):
            return False
        action = menu.actionAt(event.pos())
        if action is None or id(action) not in self._action_entries:
            return False

        action_id = id(action)
        if event_type == QEvent.Type.ContextMenu and (
            self._suppressed_action_context_id == action_id
        ):
            if not self._action_help_shown:
                self._show_action_help(menu, action, menu.mapToGlobal(event.pos()))
                self._action_help_shown = True
            event.accept()
            return True
        if event_type == QEvent.Type.MouseButtonPress:
            self._suppressed_action_context_id = action_id
            self._action_help_shown = False
            event.accept()
            return True
        if event_type == QEvent.Type.MouseButtonRelease:
            self._show_action_help(menu, action, menu.mapToGlobal(event.pos()))
            self._action_help_shown = True
            QTimer.singleShot(0, self._clear_action_context_suppression)
            event.accept()
            return True

        self._show_action_help(menu, action, menu.mapToGlobal(event.pos()))
        self._suppressed_action_context_id = action_id
        self._action_help_shown = True
        QTimer.singleShot(0, self._clear_action_context_suppression)
        event.accept()
        return True

    def _show_action_help(
        self, menu: QMenu, action: QAction, global_pos: QPoint
    ) -> None:
        QWhatsThis.showText(global_pos, self._action_entries[id(action)], menu)

    def _clear_action_context_suppression(self) -> None:
        self._suppressed_action_context_id = None
        self._action_help_shown = False

    @staticmethod
    def _format_help(title: str, body: str) -> str:
        paragraphs = [
            f"<p>{escape(part).replace(chr(10), '<br>')}</p>"
            for part in body.replace("\r\n", "\n").replace("\r", "\n").split("\n\n")
            if part
        ]
        return f"<h3>{escape(title)}</h3>" + "".join(paragraphs)

    def _is_standard_text_editor(self, widget: QWidget) -> bool:
        if type(widget) not in {QLineEdit, QTextEdit, QPlainTextEdit}:
            return False
        return (
            widget.contextMenuPolicy() == Qt.ContextMenuPolicy.DefaultContextMenu
            or id(widget) in self._spin_line_edit_ids
        )

    def _watch(self, widget: QWidget, context_menu: bool) -> None:
        self._entries[id(widget)] = context_menu
        self._watched_refs[id(widget)] = ref(widget)
        widget.installEventFilter(self)

    def _belongs_to_owner(self, widget: QWidget) -> bool:
        owner = self._owner_ref()
        return owner is not None and not sip.isdeleted(owner) and not sip.isdeleted(widget) and (
            widget is owner or owner.isAncestorOf(widget)
        )

    def _show_text_editor_menu(self, widget: QWidget, event: QEvent) -> None:
        # All three classes expose createStandardContextMenu().  Calling it keeps
        # Qt's normal undo/copy/paste actions, then appends our one extra action.
        menu: QMenu = widget.createStandardContextMenu()  # type: ignore[attr-defined]
        self._add_help_action(menu, widget, event.globalPos())
        self._popup_menu(menu, event.globalPos())

    def _show_help_menu(self, widget: QWidget, global_pos: QPoint) -> None:
        menu = QMenu(widget)
        self._add_help_action(menu, widget, global_pos)
        self._popup_menu(menu, global_pos)

    def _add_help_action(
        self, menu: QMenu, widget: QWidget, global_pos: QPoint
    ) -> None:
        action = menu.addAction(_HELP_ACTION_TEXT)
        action.triggered.connect(
            lambda _checked=False, target=widget, pos=QPoint(global_pos): self.show_help(
                target, pos
            )
        )

    def _popup_menu(self, menu: QMenu, global_pos: QPoint) -> None:
        self._open_menus.add(menu)
        menu.aboutToHide.connect(lambda target=menu: self._dispose_menu(target))
        menu.popup(global_pos)

    def _dispose_menu(self, menu: QMenu) -> None:
        self._open_menus.discard(menu)
        menu.deleteLater()

    def _detach(self) -> None:
        """Detach before the owner tears down its child widgets.

        Qt can otherwise delete a watched child while a Python event-filter
        wrapper is still active during application shutdown on Windows.
        """
        # A Close event has a queued cancellation check.  Mark it resolved
        # before touching Qt objects so that late queued calls cannot revisit a
        # hidden/deleted owner during interpreter shutdown.
        self._close_pending = False
        owner = self._owner_ref()
        if owner is not None and not sip.isdeleted(owner):
            owner.removeEventFilter(self)
        for widget_ref in self._watched_refs.values():
            widget = widget_ref()
            if widget is not None and not sip.isdeleted(widget):
                widget.removeEventFilter(self)
        self._watched_refs.clear()
        self._entries.clear()
        self._spin_line_edit_ids.clear()
        self._action_entries.clear()
        self._action_menu_ids.clear()
        self._suppressed_action_context_id = None
        self._action_help_shown = False

    def _clear_cancelled_close(self) -> None:
        if not self._close_pending:
            return
        owner = self._owner_ref()
        if owner is None or sip.isdeleted(owner):
            return
        if owner.isVisible():
            self._close_pending = False
        else:
            self._detach()
