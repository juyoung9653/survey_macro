import copy
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt6.QtCore import (
    QObject,
    QRectF,
    QThread,
    QTimer,
    Qt,
    QUrl,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QDesktopServices,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .models import Box, Field, TemplatePreset, validate_field_names
from .help_ui import HelpController
from .processor import (
    _runtime_directory,
    generate_ui_templates,
    generate_ui_templates_multi,
    remap_preset_to_detected_layout,
    run_analysis,
    _is_complete_result_run,
    _result_run_sort_key,
)
from .progress import ProgressTiming, format_duration
from .vision import (
    ImageAligner,
    PageOrientationError,
    apply_rotation,
    auto_detect_checkboxes,
    clear_all_cache,
    estimate_deskew_angle,
    load_checkbox_cache,
    load_pdf_pages,
    save_checkbox_cache,
)

ROTATION_LABELS = ["원본 0°", "좌측 90°", "우측 90°", "180°"]
ROTATION_CODES = [
    -1,
    cv2.ROTATE_90_COUNTERCLOCKWISE,
    cv2.ROTATE_90_CLOCKWISE,
    cv2.ROTATE_180,
]
ROTATION_MAP = {idx: code for idx, code in enumerate(ROTATION_CODES)}


class MainCanvas(QGraphicsView):
    """모든 페이지가 이어 붙여진 단일 캔버스"""

    MODE_SELECT = "select"
    MODE_DRAW_BOX = "draw_box"
    MODE_DRAW_COMMENT = "draw_comment"

    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 줌(확대/축소) 시 마우스 커서 위치를 중심으로 하도록 설정
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.mode = self.MODE_SELECT
        self.operation = None
        self.start_pos = None
        self.temp_rect = None
        self.select_rect = None
        self.preview_rects = []
        self.drag_boxes = []
        self.resize_box = None
        self.resize_handle = None

    def set_image(self, cv_img):
        if cv_img.ndim == 2:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
        h, w, c = cv_img.shape
        # numpy 배열의 메모리가 QImage에 제대로 유지되도록 복사본 사용
        bytes_per_line = w * c
        qimg = QImage(
            cv_img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888
        ).copy()
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(qimg))
        self.scene.setSceneRect(0, 0, w, h)
        self.temp_rect = None
        self.select_rect = None
        self.preview_rects = []

    def set_mode(self, mode: str):
        if mode not in {
            self.MODE_SELECT,
            self.MODE_DRAW_BOX,
            self.MODE_DRAW_COMMENT,
        }:
            mode = self.MODE_SELECT
        self._finish_interaction()
        self.mode = mode
        self.setCursor(
            Qt.CursorShape.CrossCursor
            if mode != self.MODE_SELECT
            else Qt.CursorShape.ArrowCursor
        )
        self.setFocus()

    def _remove_scene_item(self, item):
        if item is None:
            return
        try:
            if item.scene() is self.scene:
                self.scene.removeItem(item)
        except RuntimeError:
            pass

    def _finish_interaction(self):
        self._remove_scene_item(self.temp_rect)
        self._remove_scene_item(self.select_rect)
        for item in self.preview_rects:
            self._remove_scene_item(item)
        self.operation = None
        self.start_pos = None
        self.temp_rect = None
        self.select_rect = None
        self.preview_rects = []
        self.drag_boxes = []
        self.resize_box = None
        self.resize_handle = None

    @staticmethod
    def _drag_rect(start_pos, end_pos) -> QRectF:
        return QRectF(start_pos, end_pos).normalized()

    def _preview_pen(self) -> QPen:
        color = (
            Qt.GlobalColor.magenta
            if self.mode == self.MODE_DRAW_COMMENT
            else Qt.GlobalColor.blue
        )
        return QPen(color, 2, Qt.PenStyle.DashLine)

    def _start_move_preview(self):
        self.drag_boxes = list(self.parent_window.selected_boxes)
        pen = QPen(Qt.GlobalColor.darkYellow, 2, Qt.PenStyle.DashLine)
        self.preview_rects = [
            self.scene.addRect(self.parent_window.get_stitched_rect(box), pen)
            for box in self.drag_boxes
        ]

    def _start_resize_preview(self, box):
        self.resize_box = box
        pen = QPen(Qt.GlobalColor.darkYellow, 2, Qt.PenStyle.DashLine)
        self.preview_rects = [
            self.scene.addRect(self.parent_window.get_stitched_rect(box), pen)
        ]

    # --- Ctrl + 마우스 휠 (확대/축소) ---
    def wheelEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            zoom_in = 1.15
            zoom_out = 1 / zoom_in
            if event.angleDelta().y() > 0:
                self.scale(zoom_in, zoom_in)
            else:
                self.scale(zoom_out, zoom_out)
            event.accept()
        else:
            # Ctrl을 누르지 않았을 때는 일반 스크롤 동작
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        self.start_pos = self.mapToScene(event.pos())
        if self.mode in {self.MODE_DRAW_BOX, self.MODE_DRAW_COMMENT}:
            self.operation = "draw"
            self.temp_rect = self.scene.addRect(
                QRectF(self.start_pos, self.start_pos), self._preview_pen()
            )
            event.accept()
            return

        scale = max(0.1, abs(self.transform().m11()))
        handle_hit = self.parent_window.resize_handle_at_stitched(
            self.start_pos.x(), self.start_pos.y(), 9.0 / scale
        )
        if handle_hit is not None:
            self.operation = "resize"
            self.resize_box, self.resize_handle = handle_hit
            self._start_resize_preview(self.resize_box)
            event.accept()
            return

        clicked = self.parent_window.box_at_stitched(
            self.start_pos.x(), self.start_pos.y()
        )
        modifiers = event.modifiers()
        shift_pressed = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        ctrl_pressed = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        if clicked is not None:
            if shift_pressed:
                remains_selected = self.parent_window.select_box_range(
                    clicked, additive=ctrl_pressed
                )
            else:
                remains_selected = self.parent_window.select_box(
                    clicked, additive=ctrl_pressed
                )
            if remains_selected:
                self.operation = "move"
                self._start_move_preview()
            event.accept()
            return

        if not ctrl_pressed:
            self.parent_window.clear_box_selection()
        self.operation = "select"
        self.select_rect = self.scene.addRect(
            QRectF(self.start_pos, self.start_pos),
            QPen(Qt.GlobalColor.red, 1, Qt.PenStyle.DashLine),
        )
        event.accept()

    def mouseMoveEvent(self, event):
        if self.start_pos is None:
            super().mouseMoveEvent(event)
            return
        cur_pos = self.mapToScene(event.pos())
        if self.operation == "draw" and self.temp_rect:
            self.temp_rect.setRect(self._drag_rect(self.start_pos, cur_pos))
        elif self.operation == "select" and self.select_rect:
            self.select_rect.setRect(self._drag_rect(self.start_pos, cur_pos))
        elif self.operation == "move" and self.preview_rects:
            dx = cur_pos.x() - self.start_pos.x()
            dy = cur_pos.y() - self.start_pos.y()
            dx, dy = self.parent_window.bounded_move_delta(
                self.drag_boxes, dx, dy
            )
            for box, item in zip(self.drag_boxes, self.preview_rects):
                rect = self.parent_window.get_stitched_rect(box)
                rect.translate(dx, dy)
                item.setRect(rect)
        elif self.operation == "resize" and self.preview_rects:
            dx = cur_pos.x() - self.start_pos.x()
            dy = cur_pos.y() - self.start_pos.y()
            rect = self.parent_window.preview_resized_stitched_rect(
                self.resize_box, self.resize_handle, dx, dy
            )
            self.preview_rects[0].setRect(rect)
        else:
            super().mouseMoveEvent(event)
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self.start_pos is None:
            super().mouseReleaseEvent(event)
            return

        end_pos = self.mapToScene(event.pos())
        rect = self._drag_rect(self.start_pos, end_pos)
        operation = self.operation
        mode = self.mode
        drag_boxes = list(self.drag_boxes)
        resize_box = self.resize_box
        resize_handle = self.resize_handle
        dx = end_pos.x() - self.start_pos.x()
        dy = end_pos.y() - self.start_pos.y()
        self._finish_interaction()

        if operation == "draw" and rect.width() > 5 and rect.height() > 5:
            if mode == self.MODE_DRAW_COMMENT:
                self.parent_window.add_comment_box_from_stitched(
                    rect.x(), rect.y(), rect.width(), rect.height()
                )
            else:
                self.parent_window.add_pending_box_from_stitched(
                    rect.x(), rect.y(), rect.width(), rect.height()
                )
            self.parent_window.set_edit_mode(self.MODE_SELECT)
        elif operation == "select":
            ctrl_pressed = bool(
                event.modifiers() & Qt.KeyboardModifier.ControlModifier
            )
            self.parent_window.handle_selection_from_stitched(
                rect.x(), rect.y(), rect.width(), rect.height(), ctrl_pressed
            )
        elif operation == "move" and drag_boxes:
            dx, dy = self.parent_window.bounded_move_delta(drag_boxes, dx, dy)
            self.parent_window.move_selected_boxes(dx, dy)
        elif operation == "resize" and resize_box is not None:
            self.parent_window.resize_box(
                resize_box, resize_handle, dx, dy
            )
        event.accept()

    def contextMenuEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        clicked = self.parent_window.box_at_stitched(
            scene_pos.x(), scene_pos.y()
        )
        if clicked is None:
            self.parent_window.help_controller.show_help(self, event.globalPos())
            event.accept()
            return
        if not any(box is clicked for box in self.parent_window.selected_boxes):
            self.parent_window.select_box(clicked, additive=False)
        self.parent_window.show_box_context_menu(
            self.mapToGlobal(event.pos()), clicked
        )
        event.accept()

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Undo):
            self.parent_window.undo_edit()
            event.accept()
            return
        if event.matches(QKeySequence.StandardKey.Redo):
            self.parent_window.redo_edit()
            event.accept()
            return
        if event.key() in {Qt.Key.Key_Delete, Qt.Key.Key_Backspace}:
            self.parent_window.delete_selected_boxes()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            self._finish_interaction()
            self.parent_window.set_edit_mode(self.MODE_SELECT)
            event.accept()
            return
        movement = {
            Qt.Key.Key_Left: (-1, 0),
            Qt.Key.Key_Right: (1, 0),
            Qt.Key.Key_Up: (0, -1),
            Qt.Key.Key_Down: (0, 1),
        }.get(event.key())
        if movement and self.parent_window.selected_boxes:
            step = 10 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
            self.parent_window.move_selected_boxes(
                movement[0] * step, movement[1] * step
            )
            event.accept()
            return
        super().keyPressEvent(event)


class _BatchApplyDialog(QDialog):
    def __init__(self, parent, src_name: str, candidates: list[tuple[int, str]]):
        super().__init__(parent)
        self.selected_indices: list[int] = []
        self.setWindowTitle("일괄 적용 대상 선택")
        self.setMinimumWidth(350)

        layout = QVBoxLayout(self)
        self.help_controller = HelpController(self)
        layout.addWidget(QLabel(f"현재 문항: {src_name}"))
        layout.addWidget(QLabel("선택지 개수가 같은 문항 (일괄 적용 대상):"))

        self.checkboxes: list[tuple[QCheckBox, int]] = []
        for idx, name in candidates:
            cb = QCheckBox(name)
            cb.setChecked(True)
            layout.addWidget(cb)
            self.checkboxes.append((cb, idx))
            self.help_controller.register(cb, "이 문항에도 적용", "체크한 문항에 현재 선택지 이름과 설정을 복사합니다.\n선택지 개수가 같아도 질문의 뜻이 다른 문항은 체크를 해제하세요.")

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok_btn = button_box.button(QDialogButtonBox.StandardButton.Ok)
        cancel_btn = button_box.button(QDialogButtonBox.StandardButton.Cancel)
        if ok_btn:
            ok_btn.setText("적용")
        if cancel_btn:
            cancel_btn.setText("취소")
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.help_controller.register(ok_btn, "적용", "체크한 문항에 설정을 복사합니다.\n선택지 설정 창의 확인을 눌러야 최종 반영됩니다.")
        self.help_controller.register(cancel_btn, "취소", "일괄 적용을 하지 않고 이전 설정 창으로 돌아갑니다.")

    def _on_accept(self):
        self.selected_indices = [idx for cb, idx in self.checkboxes if cb.isChecked()]
        if not self.selected_indices:
            QMessageBox.warning(self, "알림", "선택된 문항이 없습니다.")
            return
        self.accept()


class ValueMappingDialog(QDialog):
    def __init__(self, parent, fields: list[Field], reverse_numbering: bool):
        super().__init__(parent)
        self.fields = fields
        self.reverse_numbering = reverse_numbering
        self.current_group_index = 0
        self.working_maps = [list(f.value_map) for f in fields]
        self.working_names = [f.name for f in fields]
        self.working_allow_duplicates = [f.allow_duplicates for f in fields]
        self.working_show_average = [
            f.show_average if f.reverse_numbering is not None else
            f.show_average or not any(f.value_map) for f in fields
        ]
        self.working_reverse_numbering = [f.effective_reverse_numbering(reverse_numbering) for f in fields]
        self.row_index_order = []

        self.setWindowTitle("문항 및 선택지 설정")
        self.setMinimumSize(480, 460)

        layout = QVBoxLayout(self)

        group_layout = QHBoxLayout()
        group_layout.addWidget(QLabel("문항 선택"))
        self.group_combo = QComboBox()
        self.group_combo.addItems([f.name for f in fields])
        group_layout.addWidget(self.group_combo)
        layout.addLayout(group_layout)

        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("문항 이름"))
        self.group_name_edit = QLineEdit()
        name_layout.addWidget(self.group_name_edit)
        layout.addLayout(name_layout)

        direction_layout = QHBoxLayout()
        direction_layout.addWidget(QLabel("선택지 번호 방향"))
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["정순 (1 → 마지막 번호)", "역순 (마지막 번호 → 1)"])
        direction_layout.addWidget(self.direction_combo)
        layout.addLayout(direction_layout)
        self.reverse_label = QLabel()
        self.reverse_label.setWordWrap(True)
        layout.addWidget(self.reverse_label)

        check_layout = QHBoxLayout()
        self.duplicate_check = QCheckBox("중복 허용 (다중 선택 가능)")
        check_layout.addWidget(self.duplicate_check)
        self.average_check = QCheckBox("평균 보기")
        check_layout.addWidget(self.average_check)
        layout.addLayout(check_layout)

        self.average_hint = QLabel(
            "평균을 낼 문항은 아래 선택지 이름을 모두 비우고 ‘평균 보기’를 체크하세요.\n"
            "문항 이름은 바꿔도 됩니다. 선택지 이름을 입력하면 평균 보기가 해제됩니다."
        )
        self.average_hint.setWordWrap(True)
        layout.addWidget(self.average_hint)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["번호", "선택지 이름"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.table.cellChanged.connect(self._on_cell_changed)
        layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        self.batch_apply_btn = QPushButton("일괄 적용")
        self.batch_apply_btn.clicked.connect(self._batch_apply)
        btn_layout.addWidget(self.batch_apply_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok_btn = button_box.button(QDialogButtonBox.StandardButton.Ok)
        cancel_btn = button_box.button(QDialogButtonBox.StandardButton.Cancel)
        if ok_btn:
            ok_btn.setText("확인")
        if cancel_btn:
            cancel_btn.setText("취소")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.group_combo.currentIndexChanged.connect(self._on_group_changed)
        self.direction_combo.currentIndexChanged.connect(self._on_direction_changed)

        if fields:
            self._load_group(0)

        self.help_controller = HelpController(self)
        for widget, title, text in (
            (self.group_combo, "문항 선택", "설정할 문항을 고릅니다.\n다른 문항으로 이동해도 이 창 안의 수정 내용은 유지됩니다. 마지막에 확인을 누르세요."),
            (self.group_name_edit, "문항 이름", "엑셀에서 알아볼 수 있는 질문 이름을 입력합니다. 예: 이용 만족도.\n다른 문항과 이름이 겹치지 않게 하세요."),
            (self.direction_combo, "선택지 번호 방향", "선택한 문항만 정순 또는 역순으로 바꿉니다. 아래 점수 순서를 종이의 보기와 대조하세요.\n입력한 선택지 이름은 원래 칸에 대응하도록 자동 재배치됩니다. 다른 문항에도 적용하려면 일괄 적용을 사용하세요."),
            (self.reverse_label, "점수 순서", "보기 순서에 대응하는 점수입니다. 가로는 왼쪽부터, 여러 줄은 위에서 아래로 읽습니다.\n부정형 질문을 자동으로 판별해 역채점하는 기능은 아닙니다."),
            (self.duplicate_check, "중복 허용", "여러 보기를 고를 수 있는 질문일 때 체크합니다.\n한 가지만 고르는 질문이라면 해제하세요."),
            (self.average_check, "평균 보기", "평균을 낼 문항은 오른쪽 선택지 이름을 모두 비우고 이 항목을 체크하세요. 왼쪽 번호가 점수로 사용됩니다.\n문항 이름은 바꿔도 됩니다. 선택지 이름에 글자나 숫자를 입력하면 평균 보기가 해제됩니다. 점수 방향을 종이와 대조하세요."),
            (self.average_hint, "평균용 문항과 이름 표시용 문항", "평균용 문항: 선택지 이름은 빈 칸으로 두고 평균 보기를 체크합니다.\n이름 표시용 문항: 오른쪽에 이름을 입력하고 평균 보기는 끕니다. 문항 이름 변경은 어느 경우든 가능합니다."),
            (self.table, "선택지 이름", "평균을 낼 문항은 오른쪽 칸을 비워 두세요. ‘매우 불만족’이나 숫자를 다시 입력하지 않습니다. 왼쪽 번호가 점수입니다.\n평균 없이 이름으로 표시할 문항만 오른쪽에 값을 입력하세요. 예: 인터넷, 지인 소개, 홍보물. 입력하면 평균 보기가 해제됩니다."),
            (self.batch_apply_btn, "일괄 적용", "선택지 이름, 번호 방향, 중복 허용, 평균 보기 설정을 함께 복사합니다. 문항 이름은 복사하지 않습니다.\n같은 보기 구성을 사용하는 문항만 대상으로 선택하세요."),
            (ok_btn, "확인", "이 창에서 바꾼 문항 이름과 선택지 설정을 반영합니다.\n다음에도 쓰려면 메인 화면에서 프리셋 저장을 누르세요."),
            (cancel_btn, "취소", "이 창에서 수정한 내용을 반영하지 않고 닫습니다."),
        ):
            self.help_controller.register(widget, title, text)
        self.item_help_btn = QPushButton("? 항목 도움말")
        self.item_help_btn.clicked.connect(self.help_controller.enter_mode)
        self.item_help_btn.setToolTip("누른 뒤 궁금한 항목을 클릭하세요. Esc로 취소합니다.")
        btn_layout.addWidget(self.item_help_btn)
        self.help_controller.register(self.item_help_btn, "항목 도움말", "이 버튼을 누른 뒤 궁금한 항목을 클릭하면 설명이 나옵니다.\n항목을 우클릭해도 설명을 볼 수 있습니다. Esc로 취소합니다.")

    def _load_group(self, index: int):
        if index < 0 or index >= len(self.fields):
            return

        field = self.fields[index]
        box_count = len(field.boxes)
        self.table.setRowCount(box_count)

        self.group_name_edit.setText(self.working_names[index])

        dup = (
            self.working_allow_duplicates[index]
            if index < len(self.working_allow_duplicates)
            else False
        )
        self.duplicate_check.setChecked(bool(dup))

        avg = (
            self.working_show_average[index]
            if index < len(self.working_show_average)
            else False
        )

        values = self.working_maps[index] if index < len(self.working_maps) else []
        self.row_index_order = list(range(box_count))
        if self.working_reverse_numbering[index]:
            self.row_index_order = list(reversed(self.row_index_order))
        self.direction_combo.blockSignals(True)
        self.direction_combo.setCurrentIndex(int(self.working_reverse_numbering[index]))
        self.direction_combo.setEnabled(not field.is_comment and box_count > 1)
        self.direction_combo.blockSignals(False)
        self.reverse_label.setText(
            "자유기입 문항에는 점수 번호를 적용하지 않습니다." if field.is_comment else
            "점수 순서: " + " → ".join(str(number + 1) for number in self.row_index_order) + "점"
        )

        self.table.blockSignals(True)
        for row_idx, map_idx in enumerate(self.row_index_order):
            num_item = QTableWidgetItem(str(map_idx + 1))
            num_item.setFlags(num_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row_idx, 0, num_item)

            value = values[map_idx] if map_idx < len(values) else ""
            self.table.setItem(row_idx, 1, QTableWidgetItem(value))
        self.table.blockSignals(False)

        self.average_check.setChecked(bool(avg))

    def _on_direction_changed(self, index: int):
        if not self.fields:
            return
        current = self.current_group_index
        reverse = bool(index)
        if self.fields[current].is_comment or self.working_reverse_numbering[current] == reverse:
            return
        self._save_current_group()
        self.working_maps[current] = list(reversed(self.working_maps[current]))
        self.working_reverse_numbering[current] = reverse
        self._load_group(current)

    def _on_cell_changed(self, row: int, col: int):
        if col != 1:
            return
        # 값이 하나라도 입력되면 평균 보기 해제
        has_value = any(
            (self.table.item(r, 1) and self.table.item(r, 1).text().strip())
            for r in range(self.table.rowCount())
        )
        if has_value:
            self.average_check.setChecked(False)

    def _save_current_group(self):
        if not self.fields:
            return
        idx = self.current_group_index
        if idx < 0 or idx >= len(self.fields):
            return

        row_count = self.table.rowCount()
        values = [""] * row_count
        for row_idx in range(row_count):
            item = self.table.item(row_idx, 1)
            value = item.text().strip() if item else ""
            map_idx = (
                self.row_index_order[row_idx]
                if row_idx < len(self.row_index_order)
                else row_idx
            )
            if map_idx < len(values):
                values[map_idx] = value

        name = self.group_name_edit.text().strip()
        if not name:
            name = self.fields[idx].name

        if idx >= len(self.working_maps):
            self.working_maps.extend([[]] * (idx - len(self.working_maps) + 1))
        self.working_maps[idx] = values

        if idx >= len(self.working_names):
            self.working_names.extend([""] * (idx - len(self.working_names) + 1))
        self.working_names[idx] = name
        self.group_combo.setItemText(idx, name)

        if idx >= len(self.working_allow_duplicates):
            self.working_allow_duplicates.extend(
                [False] * (idx - len(self.working_allow_duplicates) + 1)
            )
        self.working_allow_duplicates[idx] = self.duplicate_check.isChecked()

        if idx >= len(self.working_show_average):
            self.working_show_average.extend(
                [False] * (idx - len(self.working_show_average) + 1)
            )
        self.working_show_average[idx] = self.average_check.isChecked()

    def _on_group_changed(self, index: int):
        self._save_current_group()
        self.current_group_index = index
        self._load_group(index)

    def _batch_apply(self):
        self._save_current_group()
        src_idx = self.current_group_index
        src_box_count = len(self.fields[src_idx].boxes)
        src_values = self.working_maps[src_idx]

        candidates = []
        for i, f in enumerate(self.fields):
            if i == src_idx:
                continue
            if len(f.boxes) == src_box_count and f.is_comment == self.fields[src_idx].is_comment:
                candidates.append((i, f.name))

        if not candidates:
            QMessageBox.information(
                self, "알림", "선택지 개수가 같은 문항이 없습니다."
            )
            return

        dialog = _BatchApplyDialog(self, self.fields[src_idx].name, candidates)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            for idx in dialog.selected_indices:
                self.working_maps[idx] = list(src_values)
                self.working_reverse_numbering[idx] = self.working_reverse_numbering[src_idx]
                if src_idx < len(self.working_allow_duplicates):
                    self.working_allow_duplicates[idx] = self.working_allow_duplicates[
                        src_idx
                    ]
                if src_idx < len(self.working_show_average):
                    self.working_show_average[idx] = self.working_show_average[src_idx]
            self._load_group(self.current_group_index)

    def accept(self):
        self._save_current_group()
        try:
            cleaned_names = validate_field_names(self.working_names)
        except ValueError as exc:
            QMessageBox.warning(self, "문항 이름 확인", str(exc))
            return
        for idx, field in enumerate(self.fields):
            box_count = len(field.boxes)
            values = self.working_maps[idx] if idx < len(self.working_maps) else []
            if len(values) < box_count:
                values = values + [""] * (box_count - len(values))
            elif len(values) > box_count:
                values = values[:box_count]
            field.value_map = values

            field.name = cleaned_names[idx]

            dup = (
                self.working_allow_duplicates[idx]
                if idx < len(self.working_allow_duplicates)
                else False
            )
            field.allow_duplicates = bool(dup)

            avg = (
                self.working_show_average[idx]
                if idx < len(self.working_show_average)
                else False
            )
            field.show_average = bool(avg)
            if not field.is_comment:
                field.reverse_numbering = self.working_reverse_numbering[idx]
        super().accept()


class _AnalysisWorker(QObject):
    progress = pyqtSignal(float, str)
    finished = pyqtSignal(bool, str)

    def __init__(
        self,
        file_paths: list[str],
        template_pages: list[np.ndarray],
        preset: TemplatePreset,
        template_pages_preprocessed: bool = False,
        single_sample_template_pages: list[np.ndarray] | None = None,
    ):
        super().__init__()
        self.file_paths = file_paths
        self.template_pages = template_pages
        self.preset = preset
        self.template_pages_preprocessed = template_pages_preprocessed
        self.single_sample_template_pages = single_sample_template_pages or []

    @pyqtSlot()
    def run(self):
        try:
            success = run_analysis(
                self.file_paths,
                self.template_pages,
                self.preset,
                progress_cb=self.progress.emit,
                template_pages_preprocessed=self.template_pages_preprocessed,
                single_sample_template_pages=self.single_sample_template_pages,
            )
            self.finished.emit(bool(success), "")
        except Exception as exc:
            self.finished.emit(False, str(exc))


class MainWindow(QMainWindow):
    ROT_CYCLE = ROTATION_CODES

    def __init__(self):
        super().__init__()
        self.setWindowTitle("설문지 자동 분석기")
        self.resize(1600, 900)

        self.preset = TemplatePreset()
        self.pages = []
        self._pages_are_canonical = False
        self._analysis_reference_pages = []
        self._single_sample_template_pages = []
        self._inferred_display_templates = {}
        self._analysis_validation_error = ""
        self.file_paths = []

        self.preset_dir = Path(os.getenv("LOCALAPPDATA")) / "CheckFinder" / "presets"
        self.preset_dir.mkdir(parents=True, exist_ok=True)
        self.current_preset_name = None
        self._preset_dirty = False

        self.is_a_view = False  # False = B안(1열 세로연결), True = A안(2열 세로연결)
        self.rot_idx = 0
        self.page_H = 0
        self.page_W = 0

        self.pending_boxes = []
        self.selected_boxes = []
        self._selection_anchor: Box | None = None
        self.edit_mode = MainCanvas.MODE_SELECT
        self._undo_history = []
        self._redo_history = []
        self._edit_history_limit = 50

        self._analysis_thread: QThread | None = None
        self._analysis_worker: _AnalysisWorker | None = None
        self._analysis_progress: QProgressDialog | None = None
        self._analysis_result: tuple[bool, str] | None = None
        self._analysis_timing: ProgressTiming | None = None
        self._analysis_last_progress = 0.0
        self._analysis_last_message = "분석 준비 중..."
        self._analysis_elapsed_text = ""
        self._analysis_timer = QTimer(self)
        self._analysis_timer.setInterval(1000)
        self._analysis_timer.timeout.connect(self._refresh_analysis_progress)

        self._init_ui()

    def _init_menu(self):
        self.file_menu_btn = QToolButton()
        self.file_menu_btn.setText("파일")
        self.file_menu_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        file_menu = QMenu(self)
        self.file_menu_btn.setMenu(file_menu)

        load_action = file_menu.addAction("PDF 불러오기")
        load_action.triggered.connect(self.load_pdf)

        page_menu = file_menu.addMenu("페이지 설정")
        rotation_menu = page_menu.addMenu("회전")

        self.rotation_group = QActionGroup(self)
        self.rotation_group.setExclusive(True)
        self.rotation_actions = []

        for idx, label in enumerate(ROTATION_LABELS):
            action = rotation_menu.addAction(label)
            action.setCheckable(True)
            self.rotation_group.addAction(action)
            action.triggered.connect(lambda checked, i=idx: self.change_rotation(i))
            self.rotation_actions.append(action)

        # Legacy API only; users now change direction in the per-field dialog.
        self.reverse_number_action = QAction("번호 역순: ON", self)
        self.reverse_number_action.setCheckable(True)
        self.reverse_number_action.toggled.connect(self.toggle_reverse_numbering)

        self.view_toggle_action = page_menu.addAction("")
        self.view_toggle_action.triggered.connect(self.toggle_view)

        preset_menu = file_menu.addMenu("프리셋")
        save_action = preset_menu.addAction("저장")
        save_action.triggered.connect(self.save_preset)
        save_as_action = preset_menu.addAction("다른 이름으로 저장")
        save_as_action.triggered.connect(self.save_preset_as)
        load_preset_action = preset_menu.addAction("불러오기")
        load_preset_action.triggered.connect(self.load_preset_dialog)
        delete_preset_action = preset_menu.addAction("삭제")
        delete_preset_action.triggered.connect(self.delete_preset)

        settings_menu = file_menu.addMenu("기울기")
        self.manual_fine_angle_action = settings_menu.addAction(
            "수동 기울기 조정..."
        )
        self.manual_fine_angle_action.setToolTip(
            "자동 보정 후에도 기울어진 경우에만 전체 각도를 직접 조정합니다."
        )
        self.manual_fine_angle_action.triggered.connect(
            self.open_manual_fine_angle_dialog
        )

        cache_action = file_menu.addAction("캐시 삭제")
        cache_action.triggered.connect(self.clear_cache)

        self._menu_help_entries = [
            (file_menu, load_action, "PDF 불러오기", "같은 양식의 PDF를 선택합니다. 체크칸을 찾고 가로줄 기준으로 Q1, Q2… 문항을 자동으로 묶습니다.\n완료 후 문항이 실제 질문과 맞는지 확인하세요."),
            (file_menu, page_menu.menuAction(), "페이지 설정", "페이지 회전과 화면 배치를 설정합니다.\n선택지 번호 방향은 선택지 이름 설정 창에서 문항별로 변경하세요."),
            (page_menu, rotation_menu.menuAction(), "회전", "설문지가 옆으로 눕거나 뒤집혀 있으면 페이지 방향을 선택합니다.\n작은 기울기는 기울기 → 수동 기울기 조정을 사용하세요."),
            (page_menu, self.view_toggle_action, "화면 배치", "설문지 페이지를 세로 한 열 또는 두 열로 배치해 봅니다.\n응답 내용이나 문항 구성을 바꾸는 기능은 아닙니다."),
            (file_menu, preset_menu.menuAction(), "프리셋", "문항 이름, 선택지 설정과 영역 위치를 저장하고 다시 사용합니다.\n다음에도 같은 설문 양식일 때 불러오세요."),
            (preset_menu, save_action, "프리셋 저장", "현재 문항과 영역 설정을 저장합니다. 이미 이름이 있는 프리셋은 해당 설정을 갱신합니다.\n결과 엑셀 저장과는 다릅니다."),
            (preset_menu, save_as_action, "다른 이름으로 저장", "현재 설정을 다른 프리셋 이름으로 저장합니다. 기존 설정을 남기면서 새 버전을 만들 때 사용하세요."),
            (preset_menu, load_preset_action, "프리셋 불러오기", "PDF를 먼저 연 뒤 같은 양식에 저장한 설정을 불러옵니다.\n불러온 후 문항과 영역 위치가 맞는지 확인하세요."),
            (preset_menu, delete_preset_action, "프리셋 삭제", "저장된 프리셋을 선택해 삭제합니다. 다시 쓸 설정인지 먼저 확인하세요.\n원본 PDF를 삭제하는 기능은 아닙니다."),
            (file_menu, settings_menu.menuAction(), "기울기", "작은 기울기를 직접 보정하는 설정을 엽니다.\n자동 보정 후에도 기울어졌을 때만 사용하세요."),
            (settings_menu, self.manual_fine_angle_action, "수동 기울기 조정", "자동 보정 뒤에도 종이가 기울어졌을 때 각도를 직접 조정합니다.\n적용하면 체크칸을 다시 탐지하므로 문항과 영역을 다시 확인하세요."),
            (file_menu, cache_action, "캐시 삭제", "저장된 중간 처리 자료를 지우고 현재 PDF를 다시 탐색합니다.\n원본 PDF나 결과 엑셀을 지우는 기능은 아닙니다. 다시 탐지된 문항 구성을 확인하세요."),
        ]
        self._menu_help_entries.extend(
            (rotation_menu, action, f"회전: {label}", "페이지를 선택한 방향으로 회전하고 체크칸을 다시 탐지합니다.\n적용 뒤 문항과 영역 위치를 다시 확인하세요.")
            for action, label in zip(self.rotation_actions, ROTATION_LABELS)
        )

        self._sync_rotation_actions()
        self._sync_reverse_numbering_state()
        self._sync_view_toggle_text()

    def _init_ui(self):
        self._init_menu()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.setStyleSheet(
            "QToolTip { color: #263238; background-color: #FFFDE7; "
            "border: 1px solid #90A4AE; padding: 5px; }"
        )
        main_layout = QVBoxLayout(main_widget)

        # 상단 툴바 버튼
        btn_layout = QHBoxLayout()

        self.draw_box_tool_btn = QPushButton("선택지 추가")
        self.draw_box_tool_btn.setCheckable(True)
        self.draw_box_tool_btn.setToolTip(
            "새 선택지 영역을 드래그합니다. 다시 누르면 그리기를 취소합니다."
        )
        self.draw_comment_tool_btn = QPushButton("자유기입 영역 추가")
        self.draw_comment_tool_btn.setCheckable(True)
        self.draw_comment_tool_btn.setToolTip(
            "글이나 숫자를 적는 영역을 드래그합니다. 다시 누르면 그리기를 취소합니다."
        )
        mode_buttons = {
            self.draw_box_tool_btn: MainCanvas.MODE_DRAW_BOX,
            self.draw_comment_tool_btn: MainCanvas.MODE_DRAW_COMMENT,
        }
        for button, mode in mode_buttons.items():
            button.toggled.connect(
                lambda checked, selected_mode=mode: self.set_edit_mode(
                    selected_mode if checked else MainCanvas.MODE_SELECT
                )
            )

        self.group_btn = QPushButton("문항으로 묶기")
        self.group_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; }"
        )
        self.group_btn.setToolTip(
            "선택한 박스들을 하나의 문항으로 묶습니다."
        )
        self.group_btn.clicked.connect(self.group_boxes)

        self.value_map_btn = QPushButton("선택지 이름 설정")
        self.value_map_btn.setStyleSheet(
            "QPushButton { background-color: #7E57C2; color: white; "
            "font-weight: bold; }"
        )
        self.value_map_btn.setToolTip("문항 이름과 각 선택지의 결과값을 설정합니다.")
        self.value_map_btn.clicked.connect(self.open_value_mapping)

        self.comment_field_btn = QPushButton("자유기입 전환")
        self.comment_field_btn.setToolTip(
            "기존 문항을 자유기입 문항으로 지정하거나 해제합니다."
        )
        self.comment_field_btn.clicked.connect(self.assign_comment_field)

        self.delete_selected_btn = QPushButton("선택 삭제")
        self.delete_selected_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; }"
        )
        self.delete_selected_btn.setToolTip(
            "선택한 박스를 삭제합니다. Ctrl+Z로 되돌릴 수 있습니다."
        )
        self.delete_selected_btn.clicked.connect(self.delete_selected_boxes)

        self.undo_btn = QPushButton("↶ 실행 취소")
        self.undo_btn.setToolTip("마지막 편집을 되돌립니다. (Ctrl+Z)")
        self.undo_btn.clicked.connect(self.undo_edit)
        self.redo_btn = QPushButton("↷ 다시 실행")
        self.redo_btn.setToolTip("되돌린 편집을 다시 적용합니다. (Ctrl+Y)")
        self.redo_btn.clicked.connect(self.redo_edit)
        self.undo_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)
        self.undo_shortcut = QShortcut(
            QKeySequence(QKeySequence.StandardKey.Undo), self
        )
        self.undo_shortcut.activated.connect(self.undo_edit)
        self.redo_shortcut = QShortcut(
            QKeySequence(QKeySequence.StandardKey.Redo), self
        )
        self.redo_shortcut.activated.connect(self.redo_edit)

        self.exec_btn = QPushButton("▶ 분석 실행")
        self.exec_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; }"
        )
        self.exec_btn.clicked.connect(self.execute_analysis)

        if hasattr(self, "file_menu_btn"):
            self.file_menu_btn.setFixedHeight(self.group_btn.sizeHint().height())
            btn_layout.addWidget(self.file_menu_btn)

        self.load_pdf_btn = QPushButton("PDF 불러오기")
        self.load_pdf_btn.setToolTip(
            "분석할 PDF를 선택합니다. 여러 파일도 한 번에 선택할 수 있습니다."
        )
        self.load_pdf_btn.clicked.connect(self.load_pdf)
        btn_layout.addWidget(self.load_pdf_btn)

        self.load_preset_btn = QPushButton("프리셋 불러오기")
        self.load_preset_btn.setToolTip(
            "PDF를 연 뒤 저장해 둔 문항과 선택지 설정을 적용합니다."
        )
        self.load_preset_btn.clicked.connect(self.load_preset_dialog)
        btn_layout.addWidget(self.load_preset_btn)

        self.save_preset_btn = QPushButton("프리셋 저장")
        self.save_preset_btn.setToolTip(
            "현재 문항·선택지·영역 설정을 저장합니다."
        )
        self.save_preset_btn.clicked.connect(self.save_preset)
        btn_layout.addWidget(self.save_preset_btn)

        self.document_status_label = QLabel()
        self.document_status_label.setStyleSheet(
            "padding: 5px 8px; color: #1A237E; background: #E8EAF6;"
        )
        btn_layout.addWidget(self.document_status_label)

        self.auto_deskew_btn = QPushButton("기울기 다시 맞추기")
        self.auto_deskew_btn.setToolTip(
            "PDF를 열 때 자동으로 맞춘 기울기를 다시 계산합니다."
        )
        self.auto_deskew_btn.clicked.connect(self.auto_deskew_pages)
        btn_layout.addWidget(self.auto_deskew_btn)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.undo_btn)
        btn_layout.addWidget(self.redo_btn)
        btn_layout.addWidget(self.exec_btn)

        main_layout.addLayout(btn_layout)

        edit_layout = QHBoxLayout()
        edit_layout.addWidget(QLabel("편집 도구:"))
        edit_layout.addWidget(self.draw_box_tool_btn)
        edit_layout.addWidget(self.draw_comment_tool_btn)
        edit_layout.addSpacing(8)
        edit_layout.addWidget(self.group_btn)
        edit_layout.addWidget(self.value_map_btn)
        edit_layout.addWidget(self.comment_field_btn)
        edit_layout.addWidget(self.delete_selected_btn)
        edit_layout.addStretch(1)
        self.open_results_btn = QPushButton("결과 폴더")
        self.open_results_btn.setToolTip("가장 최근에 완료된 분석의 결과 폴더를 엽니다.")
        self.open_results_btn.clicked.connect(self.open_results_folder)
        btn_layout.addWidget(self.open_results_btn)
        self.help_btn = QPushButton("설명서")
        self.help_btn.setToolTip("실제 화면을 보며 PDF부터 결과 확인까지 따라합니다.")
        self.help_btn.clicked.connect(self.open_help)
        edit_layout.addWidget(self.help_btn)
        main_layout.addLayout(edit_layout)

        guide_layout = QHBoxLayout()
        self.edit_status_label = QLabel()
        self.edit_status_label.setStyleSheet(
            "padding: 5px 8px; color: #263238; background: #ECEFF1;"
        )
        self.edit_legend_label = QLabel(
            "초록: 선택형  |  보라: 자유기입  |  파랑: 미분류  |  노랑: 선택됨"
        )
        self.edit_legend_label.setStyleSheet("color: #546E7A; padding: 5px;")
        guide_layout.addWidget(self.edit_status_label, 1)
        guide_layout.addWidget(self.edit_legend_label)
        main_layout.addLayout(guide_layout)

        # 단일 거대 캔버스 배치
        self.canvas = MainCanvas(self)
        main_layout.addWidget(self.canvas)
        self.set_edit_mode(MainCanvas.MODE_SELECT)
        self._refresh_document_status()
        self._init_context_help(edit_layout)

    def _init_context_help(self, help_layout):
        self.help_controller = HelpController(self)
        entries = (
            (self.file_menu_btn, "파일 메뉴", "PDF 불러오기, 페이지 방향, 프리셋과 캐시 설정이 있습니다.\n선택지 번호 방향은 선택지 이름 설정 창에서 문항별로 변경하세요."),
            (self.load_pdf_btn, "PDF 불러오기", "PDF를 열면 체크칸을 찾고 같은 가로줄끼리 Q1, Q2… 문항으로 자동으로 묶습니다.\n초록색 문항이 실제 질문과 맞는지 확인하세요. 같은 양식의 파일은 여러 개 골라도 됩니다."),
            (self.load_preset_btn, "같은 양식의 설정 불러오기", "PDF를 먼저 연 뒤 저장해 둔 문항·선택지 설정을 불러옵니다.\n질문이나 위치가 바뀐 설문에는 예전 설정을 그대로 쓰지 마세요."),
            (self.save_preset_btn, "프리셋 저장", "PDF 파일이 아니라 설문 양식의 설정을 저장합니다. 질문과 체크칸 배치가 같으면 파일 이름·응답자·체크한 답이 달라도 재사용할 수 있습니다.\n예: 교육 만족도 양식. 다음 파일은 PDF를 연 뒤 프리셋을 불러오세요. 분석 결과 엑셀을 저장하는 버튼은 아닙니다."),
            (self.document_status_label, "불러온 문서와 설정", "현재 PDF와 적용된 프리셋 상태를 보여줍니다.\n분석 전에 원하는 파일과 설정인지 확인하세요."),
            (self.auto_deskew_btn, "기울기 다시 맞추기", "PDF를 열 때 자동으로 맞춘 기울기를 다시 계산합니다.\n체크칸과 영역이 비스듬히 어긋날 때 사용하세요. 보정 뒤 영역 위치를 다시 확인하세요."),
            (self.draw_box_tool_btn, "빠진 선택지 추가", "버튼을 누르고 종이의 체크칸 테두리를 드래그합니다.\n추가한 파란 영역은 같은 질문의 다른 선택지와 함께 문항으로 묶으세요. Esc로 그리기를 취소합니다."),
            (self.draw_comment_tool_btn, "자유기입 영역 추가", "글이나 숫자를 적는 답변 칸을 드래그해 지정합니다.\n직접 쓴 글을 문자로 바꾸는 기능은 아닙니다. 내용은 검수 PDF에서 확인하세요."),
            (self.group_btn, "자동 문항이 틀렸을 때만 다시 묶기", "PDF를 열면 가로줄 기준으로 문항이 자동 생성됩니다. 맞게 묶였다면 이 버튼은 누르지 않아도 됩니다.\n수정할 때는 한 질문의 칸을 모두 Ctrl+클릭으로 고른 뒤 누르세요. 선택한 칸만 기존 묶음에서 새 문항으로 이동합니다."),
            (self.value_map_btn, "선택지 이름 설정", "먼저 평균을 낼 문항인지 확인하세요. 평균용 문항은 선택지 이름을 비워 두고 평균 보기를 체크합니다.\n문항 이름은 바꿔도 됩니다. 평균 없이 이름으로 표시할 문항만 선택지 이름을 입력하세요."),
            (self.comment_field_btn, "자유기입 전환", "선택한 기존 문항을 글이나 숫자를 적는 영역으로 바꾸거나 되돌립니다.\n보라색이 자유기입 영역입니다. 새 영역은 자유기입 영역 추가로 만드세요."),
            (self.delete_selected_btn, "선택 삭제", "현재 선택한 영역만 삭제합니다. 원본 PDF는 지우지 않습니다.\n잘못 지웠으면 Ctrl+Z로 되돌리세요."),
            (self.undo_btn, "실행 취소", "방금 한 영역 편집을 되돌립니다. 단축키는 Ctrl+Z입니다.\n되돌릴 편집이 없으면 비활성화됩니다."),
            (self.redo_btn, "다시 실행", "실행 취소한 편집을 다시 적용합니다. 단축키는 Ctrl+Y입니다."),
            (self.exec_btn, "분석 실행", "설정한 문항을 기준으로 PDF를 분석하고 엑셀과 검수용 파일을 만듭니다.\n파일과 문항 설정을 먼저 확인하세요. 완료 후 결과를 확인해야 작업이 끝납니다."),
            (self.open_results_btn, "결과 확인", "가장 최근에 완료된 분석의 설문결과_날짜.시간 폴더를 엽니다.\n엑셀의 결과·검수필요 시트를 보고, 확인이 필요한 응답은 검수 PDF와 대조하세요."),
            (self.help_btn, "설명서", "실제 화면의 스크린샷을 보며 PDF 불러오기부터 결과 확인까지 따라합니다.\n지금 궁금한 버튼 하나만 알고 싶으면 우클릭하세요."),
            (self.edit_status_label, "현재 편집 상태", "선택한 영역과 현재 편집 모드를 보여줍니다.\n원하는 동작이 안 되면 Esc를 눌러 그리기를 취소한 뒤 다시 선택하세요."),
            (self.edit_legend_label, "영역 색상", "초록은 문항으로 묶은 선택형, 보라는 자유기입, 파랑은 아직 묶지 않은 영역입니다.\n노랑은 현재 선택한 영역입니다."),
        )
        for widget, title, text in entries:
            self.help_controller.register(widget, title, text)
        for menu, action, title, text in self._menu_help_entries:
            self.help_controller.register_action(menu, action, title, text)
        self.help_controller.register(self.canvas, "자동 문항 확인", "초록색 Q1, Q2…가 실제 질문별 보기와 맞는지 확인하세요. 가로줄 기준 자동 묶음이 맞으면 그대로 진행합니다.\n잘못 묶인 곳만 같은 질문의 칸을 골라 문항으로 묶으세요. 영역 우클릭으로 편집·도움말을 열 수 있습니다. 원본 PDF는 변경하지 않습니다.", context_menu=False)
        self.help_controller.register(self.canvas.viewport(), "설문지 화면", "선택지를 클릭하거나 드래그해 선택합니다. 선택한 영역은 이동·크기 조절할 수 있습니다.\n초록: 선택형 / 보라: 자유기입 / 파랑: 미분류 / 노랑: 선택됨", context_menu=False)
        self.item_help_btn = QPushButton("? 항목 도움말")
        self.item_help_btn.setToolTip("누른 뒤 궁금한 버튼이나 체크박스를 클릭하세요. Esc로 취소합니다.")
        self.item_help_btn.clicked.connect(self.help_controller.enter_mode)
        self.help_controller.register(self.item_help_btn, "항목 도움말", "누른 뒤 궁금한 항목을 클릭하세요. 설명만 표시하며 그 항목을 실행하지 않습니다.\n우클릭으로도 설명을 볼 수 있습니다. Esc로 취소합니다.")
        self.statusBar().showMessage("궁금한 항목은 우클릭 · 처음이라면 ‘설명서’")
        help_layout.insertWidget(help_layout.indexOf(self.help_btn) + 1, self.item_help_btn)

    def _refresh_document_status(self):
        if not hasattr(self, "document_status_label"):
            return
        if not self.file_paths:
            document_text = "PDF 없음"
        else:
            document_text = (
                f"PDF {len(self.file_paths)}개 · {self.preset.page_count}쪽/부"
            )
        if self.current_preset_name:
            preset_text = self.current_preset_name
        elif getattr(self, "_preset_dirty", False):
            preset_text = "미저장 설정"
        else:
            preset_text = "미적용"
        if getattr(self, "_preset_dirty", False):
            preset_text += " (수정됨)"
        display_text = ""
        validation_error = getattr(self, "_analysis_validation_error", "")
        if getattr(self, "_inferred_display_templates", {}):
            display_text = " · 화면: 자동 생성 빈 양식"
        elif validation_error:
            display_text = " · 분석 전 확인 필요"
        self.document_status_label.setText(
            f"{document_text} · 프리셋: {preset_text}{display_text}"
        )
        self.document_status_label.setToolTip(validation_error)

    def _set_preset_dirty(self, dirty: bool = True):
        self._preset_dirty = bool(dirty)
        refresh = getattr(self, "_refresh_document_status", None)
        if callable(refresh):
            refresh()

    def _confirm_save_or_discard_changes(self, action_text: str) -> bool:
        if not getattr(self, "_preset_dirty", False):
            return True
        choice = QMessageBox.warning(
            self,
            "저장하지 않은 변경 내용",
            f"{action_text}\n\n현재 편집 내용을 저장할까요?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if choice == QMessageBox.StandardButton.Save:
            return bool(self.save_preset())
        return choice == QMessageBox.StandardButton.Discard

    @staticmethod
    def _manual_index_path() -> Path | None:
        candidates = [_runtime_directory() / "설명서.html"]
        bundle_directory = getattr(sys, "_MEIPASS", None)
        if bundle_directory:
            candidates.append(Path(bundle_directory) / "설명서.html")
        return next((path for path in candidates if path.is_file()), None)

    def open_help(self):
        manual_path = self._manual_index_path()
        if manual_path is None:
            QMessageBox.warning(
                self,
                "도움말 없음",
                "설명서.html을 찾을 수 없습니다. 프로그램을 다시 설치해주세요.",
            )
            return False
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(manual_path.resolve()))):
            QMessageBox.warning(
                self,
                "도움말 열기 실패",
                "기본 웹 브라우저에서 설명서를 열지 못했습니다.",
            )
            return False
        return True

    def open_results_folder(self):
        result_folder = _runtime_directory() / "결과"
        try:
            runs = []
            for folder in result_folder.iterdir() if result_folder.is_dir() else []:
                if _is_complete_result_run(folder):
                    try:
                        runs.append((_result_run_sort_key(folder), folder))
                    except ValueError:
                        continue
            latest = max(runs, key=lambda item: item[0])[1] if runs else None
        except OSError as exc:
            QMessageBox.critical(
                self, "결과 폴더 오류", f"분석 결과 폴더를 확인할 수 없습니다.\n\n{exc}"
            )
            return False
        if latest is None:
            QMessageBox.information(
                self, "분석 결과 없음", "완료된 분석 결과가 없습니다. 먼저 분석을 실행해주세요."
            )
            return False
        if not QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(latest.resolve()))
        ):
            QMessageBox.warning(
                self, "결과 폴더 열기 실패", "결과 폴더를 열지 못했습니다."
            )
            return False
        return True

    def _sync_reverse_numbering_state(self):
        is_on = bool(self.preset.reverse_numbering)
        if hasattr(self, "reverse_number_action"):
            self.reverse_number_action.blockSignals(True)
            self.reverse_number_action.setChecked(is_on)
            self.reverse_number_action.setText(
                "번호 역순: ON" if is_on else "번호 역순: OFF"
            )
            self.reverse_number_action.blockSignals(False)

    def _sync_view_toggle_text(self):
        text = "세로 보기" if self.is_a_view else "모아보기"
        if hasattr(self, "view_toggle_action"):
            self.view_toggle_action.setText(text)

    def _reset_state_for_new_pdf(self):
        self.pending_boxes.clear()
        self.selected_boxes.clear()
        self._selection_anchor = None
        self.preset.fields.clear()
        self.preset.page_fine_angles.clear()
        self.current_preset_name = None
        self._pages_are_canonical = False
        self._analysis_reference_pages = []
        self._single_sample_template_pages = []
        self._inferred_display_templates = {}
        self._analysis_validation_error = ""
        self.is_a_view = False
        self._preset_dirty = False
        MainWindow._clear_edit_history(self)
        self._refresh_document_status()

    def _capture_document_state(self) -> dict:
        return {
            "file_paths": list(getattr(self, "file_paths", [])),
            "pages": list(getattr(self, "pages", [])),
            "pages_are_canonical": bool(
                getattr(self, "_pages_are_canonical", False)
            ),
            "analysis_reference_pages": list(
                getattr(self, "_analysis_reference_pages", [])
            ),
            "single_sample_template_pages": list(
                getattr(self, "_single_sample_template_pages", [])
            ),
            "inferred_display_templates": dict(
                getattr(self, "_inferred_display_templates", {})
            ),
            "analysis_validation_error": getattr(
                self, "_analysis_validation_error", ""
            ),
            "preset": copy.deepcopy(getattr(self, "preset", TemplatePreset())),
            "pending_boxes": copy.deepcopy(getattr(self, "pending_boxes", [])),
            "is_a_view": bool(getattr(self, "is_a_view", False)),
            "rot_idx": int(getattr(self, "rot_idx", 0)),
            "current_preset_name": getattr(self, "current_preset_name", None),
            "preset_dirty": bool(getattr(self, "_preset_dirty", False)),
            "undo_history": copy.deepcopy(getattr(self, "_undo_history", [])),
            "redo_history": copy.deepcopy(getattr(self, "_redo_history", [])),
        }

    def _restore_document_state(self, snapshot: dict):
        self.file_paths = snapshot["file_paths"]
        self.pages = snapshot["pages"]
        self._pages_are_canonical = snapshot["pages_are_canonical"]
        self._analysis_reference_pages = snapshot["analysis_reference_pages"]
        self._single_sample_template_pages = list(
            snapshot.get("single_sample_template_pages", [])
        )
        self._inferred_display_templates = dict(
            snapshot.get("inferred_display_templates", {})
        )
        self._analysis_validation_error = snapshot.get(
            "analysis_validation_error", ""
        )
        self.preset = snapshot["preset"]
        self.pending_boxes = snapshot["pending_boxes"]
        self.selected_boxes = []
        self.is_a_view = snapshot["is_a_view"]
        self.rot_idx = snapshot["rot_idx"]
        self.current_preset_name = snapshot["current_preset_name"]
        self._preset_dirty = snapshot["preset_dirty"]
        self._undo_history = snapshot["undo_history"]
        self._redo_history = snapshot["redo_history"]
        for method_name in (
            "_sync_rotation_actions",
            "_sync_fine_angle_spin",
            "_sync_reverse_numbering_state",
            "_update_page_size",
            "_update_history_buttons",
            "update_canvas",
            "_refresh_document_status",
        ):
            method = getattr(self, method_name, None)
            if callable(method):
                method()

    def _capture_edit_state(self):
        return copy.deepcopy((self.preset.fields, self.pending_boxes))

    @staticmethod
    def _box_geometry_key(box: Box) -> tuple[int, int, int, int, int]:
        return (box.page_idx, box.x, box.y, box.w, box.h)

    @staticmethod
    def _boxes_share_region(first: Box, second: Box) -> bool:
        if first.page_idx != second.page_idx:
            return False
        width_ratio = max(first.w, second.w) / max(1, min(first.w, second.w))
        height_ratio = max(first.h, second.h) / max(1, min(first.h, second.h))
        if width_ratio > 1.8 or height_ratio > 1.8:
            return False
        left = max(first.x, second.x)
        top = max(first.y, second.y)
        right = min(first.x + first.w, second.x + second.w)
        bottom = min(first.y + first.h, second.y + second.h)
        intersection = max(0, right - left) * max(0, bottom - top)
        smaller_area = min(first.w * first.h, second.w * second.h)
        if smaller_area > 0 and intersection / smaller_area >= 0.5:
            return True
        first_center = (first.x + first.w / 2, first.y + first.h / 2)
        second_center = (second.x + second.w / 2, second.y + second.h / 2)
        tolerance = max(4.0, min(first.w, first.h, second.w, second.h) * 0.45)
        return (
            abs(first_center[0] - second_center[0]) <= tolerance
            and abs(first_center[1] - second_center[1]) <= tolerance
        )

    @staticmethod
    def _box_region_match_cost(first: Box, second: Box) -> float:
        first_center = (first.x + first.w / 2, first.y + first.h / 2)
        second_center = (second.x + second.w / 2, second.y + second.h / 2)
        scale = max(4.0, min(first.w, first.h, second.w, second.h))
        center_cost = (
            abs(first_center[0] - second_center[0])
            + abs(first_center[1] - second_center[1])
        ) / scale
        size_cost = abs(first.w - second.w) / max(first.w, second.w, 1)
        size_cost += abs(first.h - second.h) / max(first.h, second.h, 1)
        return center_cost + size_cost

    @staticmethod
    def _merge_detected_box_geometry(
        preset: TemplatePreset,
        pending_boxes: list[Box],
        detected_boxes_by_page: dict[int, list[Box]],
        matched_keys: set[tuple[int, int, int, int, int]],
        supplied_keys: set[tuple[int, int, int, int, int]],
    ) -> int:
        """Reuse nearby current geometry and discard unmatched old detections."""
        current_boxes = [
            box
            for boxes in detected_boxes_by_page.values()
            for box in boxes
            if MainWindow._box_geometry_key(box) in supplied_keys
        ]
        matchable_boxes = [
            box
            for field in preset.fields
            if not field.is_comment
            for box in field.boxes
        ] + list(pending_boxes)
        exact_output_ids = {
            id(box)
            for box in matchable_boxes
            if MainWindow._box_geometry_key(box) in matched_keys
        }
        reused_count = sum(
            MainWindow._box_geometry_key(box) in matched_keys
            for box in current_boxes
        )

        current_candidates = [
            (index, box)
            for index, box in enumerate(current_boxes)
            if MainWindow._box_geometry_key(box) not in matched_keys
        ]
        output_candidates = [
            (index, box)
            for index, box in enumerate(matchable_boxes)
            if id(box) not in exact_output_ids
        ]
        candidate_pairs = []
        for current_index, current_box in current_candidates:
            for output_index, output_box in output_candidates:
                if not MainWindow._boxes_share_region(current_box, output_box):
                    continue
                candidate_pairs.append(
                    (
                        MainWindow._box_region_match_cost(
                            current_box, output_box
                        ),
                        current_index,
                        output_index,
                        current_box,
                        output_box,
                    )
                )

        assigned_current: set[int] = set()
        assigned_output: set[int] = set()
        for (
            _cost,
            current_index,
            output_index,
            current_box,
            output_box,
        ) in sorted(candidate_pairs, key=lambda item: item[:3]):
            if (
                current_index in assigned_current
                or output_index in assigned_output
            ):
                continue
            output_box.x = current_box.x
            output_box.y = current_box.y
            output_box.w = current_box.w
            output_box.h = current_box.h
            assigned_current.add(current_index)
            assigned_output.add(output_index)

        reused_count += len(assigned_current)
        return reused_count

    def _restore_edit_state(self, snapshot):
        fields, pending_boxes = copy.deepcopy(snapshot)
        self.preset.fields = fields
        self.pending_boxes = pending_boxes
        self.selected_boxes.clear()
        self._selection_anchor = None
        self.update_canvas()

    def _update_history_buttons(self):
        if hasattr(self, "undo_btn"):
            self.undo_btn.setEnabled(bool(self._undo_history))
        if hasattr(self, "redo_btn"):
            self.redo_btn.setEnabled(bool(self._redo_history))

    def _clear_edit_history(self):
        if hasattr(self, "_undo_history"):
            self._undo_history.clear()
        if hasattr(self, "_redo_history"):
            self._redo_history.clear()
        MainWindow._update_history_buttons(self)

    def _commit_edit(self, previous_state, description: str) -> bool:
        current_state = self._capture_edit_state()
        if current_state == previous_state:
            return False
        self._undo_history.append((previous_state, description))
        if len(self._undo_history) > self._edit_history_limit:
            self._undo_history.pop(0)
        self._redo_history.clear()
        self._update_history_buttons()
        MainWindow._set_preset_dirty(self, True)
        if hasattr(self, "statusBar"):
            self.statusBar().showMessage(
                f"{description} · Ctrl+Z로 되돌릴 수 있습니다.", 5000
            )
        return True

    def undo_edit(self):
        if not self._undo_history:
            return
        previous_state, description = self._undo_history.pop()
        self._redo_history.append((self._capture_edit_state(), description))
        self._restore_edit_state(previous_state)
        self._update_history_buttons()
        MainWindow._set_preset_dirty(self, True)
        if hasattr(self, "statusBar"):
            self.statusBar().showMessage(f"실행 취소: {description}", 4000)

    def redo_edit(self):
        if not self._redo_history:
            return
        next_state, description = self._redo_history.pop()
        self._undo_history.append((self._capture_edit_state(), description))
        self._restore_edit_state(next_state)
        self._update_history_buttons()
        MainWindow._set_preset_dirty(self, True)
        if hasattr(self, "statusBar"):
            self.statusBar().showMessage(f"다시 실행: {description}", 4000)

    def set_edit_mode(self, mode: str):
        valid_modes = {
            MainCanvas.MODE_SELECT,
            MainCanvas.MODE_DRAW_BOX,
            MainCanvas.MODE_DRAW_COMMENT,
        }
        self.edit_mode = mode if mode in valid_modes else MainCanvas.MODE_SELECT
        if hasattr(self, "canvas"):
            self.canvas.set_mode(self.edit_mode)
        mode_buttons = {
            MainCanvas.MODE_DRAW_BOX: getattr(self, "draw_box_tool_btn", None),
            MainCanvas.MODE_DRAW_COMMENT: getattr(
                self, "draw_comment_tool_btn", None
            ),
        }
        default_text = {
            MainCanvas.MODE_DRAW_BOX: "선택지 추가",
            MainCanvas.MODE_DRAW_COMMENT: "자유기입 영역 추가",
        }
        default_tooltip = {
            MainCanvas.MODE_DRAW_BOX: (
                "새 선택지 영역을 드래그합니다. 다시 누르면 그리기를 취소합니다."
            ),
            MainCanvas.MODE_DRAW_COMMENT: (
                "글이나 숫자를 적는 영역을 드래그합니다. "
                "다시 누르면 그리기를 취소합니다."
            ),
        }
        for button_mode, button in mode_buttons.items():
            if button is None:
                continue
            active = self.edit_mode == button_mode
            signals_were_blocked = button.blockSignals(True)
            button.setChecked(active)
            button.setText("그리기 취소" if active else default_text[button_mode])
            button.setToolTip(
                "그리기를 취소하고 박스 선택으로 돌아갑니다."
                if active
                else default_tooltip[button_mode]
            )
            button.blockSignals(signals_were_blocked)
        self._refresh_edit_status()

    def _refresh_edit_status(self):
        if not hasattr(self, "edit_status_label"):
            return
        selected_count = len(self.selected_boxes)
        if not self.pages:
            text = "먼저 파일 > PDF 불러오기를 선택하세요."
        elif self.edit_mode == MainCanvas.MODE_DRAW_BOX:
            text = "선택지 추가: 영역을 드래그하세요. 버튼을 다시 누르면 취소됩니다."
        elif self.edit_mode == MainCanvas.MODE_DRAW_COMMENT:
            text = (
                "자유기입 영역 추가: 글을 적는 영역을 드래그하세요. "
                "버튼을 다시 누르면 취소됩니다."
            )
        elif selected_count == 1:
            text = (
                "1개 선택됨: 드래그로 이동 · 모서리로 크기 조절 "
                "· Ctrl 개별 선택 · Shift 범위 선택"
            )
        elif selected_count > 1:
            text = (
                f"{selected_count}개 선택됨: 드래그로 함께 이동 · "
                "Ctrl 개별 추가·해제 · Shift 범위 선택"
            )
        else:
            text = (
                "박스 클릭으로 선택 · Ctrl+클릭 개별 · "
                "Shift+클릭 가로·세로·사각 범위"
            )
        self.edit_status_label.setText(text)

    @staticmethod
    def _group_boxes_by_row(boxes: list[Box]) -> list[list[Box]]:
        if not boxes:
            return []

        boxes.sort(key=lambda b: b.y)
        rows: list[list[Box]] = []
        current_row: list[Box] = []

        for b in boxes:
            if not current_row:
                current_row.append(b)
                continue

            last_b = current_row[-1]
            y_tolerance = max(b.h, last_b.h) * 0.5
            if abs(b.y - last_b.y) <= y_tolerance:
                current_row.append(b)
            else:
                rows.append(current_row)
                current_row = [b]

        if current_row:
            rows.append(current_row)

        return rows

    def _boxes_in_reading_order(self, boxes: list[Box]) -> list[Box]:
        by_page: dict[int, list[Box]] = {}
        for box in boxes:
            by_page.setdefault(box.page_idx, []).append(box)

        ordered = []
        for page_idx in sorted(by_page):
            for row in self._group_boxes_by_row(by_page[page_idx]):
                ordered.extend(sorted(row, key=lambda box: box.x))
        return ordered

    def _all_boxes(self) -> list[Box]:
        boxes = list(self.pending_boxes)
        for field in self.preset.fields:
            boxes.extend(field.boxes)
        return boxes

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        if not hasattr(self, "_font_cache"):
            self._font_cache = {}

        cache = self._font_cache
        if size in cache:
            return cache[size]

        font = None
        windows_dir = Path(os.environ.get("WINDIR", "C:/Windows"))
        font_candidates = [
            windows_dir / "Fonts" / "malgun.ttf",
            windows_dir / "Fonts" / "malgunbd.ttf",
        ]

        for font_path in font_candidates:
            if font_path.exists():
                font = ImageFont.truetype(str(font_path), size)
                break

        if font is None:
            font = ImageFont.load_default()

        cache[size] = font
        return font

    def _draw_texts(
        self,
        img: np.ndarray,
        entries: list[tuple[str, int, int, tuple[int, int, int]]],
    ) -> np.ndarray:
        if not entries:
            return img

        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(img_pil)
            font = self._get_font(18)

            for text, x, y, color in entries:
                draw.text((x, y), text, font=font, fill=(color[2], color[1], color[0]))

            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception:
            for text, x, y, color in entries:
                cv2.putText(
                    img,
                    text,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
            return img

    def toggle_reverse_numbering(self, checked: bool):
        checked = bool(checked)
        changed = self.preset.reverse_numbering != checked
        if changed:
            MainWindow._reverse_choice_names(self.preset.fields)
            # Edit snapshots do not store the numbering direction. Rebase their
            # maps too, so a later undo/redo still names the same physical boxes.
            for history in (self._undo_history, self._redo_history):
                for (fields, _pending), _description in history:
                    MainWindow._reverse_choice_names(fields)
        self.preset.reverse_numbering = checked
        self._sync_reverse_numbering_state()
        if changed:
            MainWindow._set_preset_dirty(self, True)

    @staticmethod
    def _reverse_choice_names(fields: list[Field]) -> None:
        for field in fields:
            count = len(field.boxes)
            if field.is_comment or field.reverse_numbering is not None or count < 2 or not any(field.value_map):
                continue
            # Maps are keyed by displayed number, not physical box index.
            # Pad missing entries before reversing; preserve unrelated trailing
            # entries from older presets instead of silently deleting data.
            values = (field.value_map[:count] + [""] * count)[:count]
            field.value_map = list(reversed(values)) + field.value_map[count:]

    def _show_progress_dialog(self, title: str, label: str) -> QProgressDialog:
        dialog = QProgressDialog(label, None, 0, 100, self)
        dialog.setWindowTitle(title)
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        dialog.setCancelButton(None)
        dialog.setAutoClose(True)
        dialog.setAutoReset(True)
        dialog.setMinimumDuration(0)
        dialog.show()
        QApplication.processEvents()
        return dialog

    @staticmethod
    def _wrap_progress(base: int, span: int, default_message: str, callback):
        def _cb(value: int, message: str = ""):
            mapped = base + int(value / 100 * span)
            callback(mapped, message or default_message)

        return _cb

    @staticmethod
    def _make_progress_cb(dialog: QProgressDialog):
        def _cb(value: int, message: str = ""):
            dialog.setValue(value)
            if message:
                dialog.setLabelText(message)
            QApplication.processEvents()

        return _cb

    def open_value_mapping(self):
        if not self.preset.fields:
            QMessageBox.information(self, "알림", "선택지 이름을 설정할 문항이 없습니다.")
            return

        previous_state = self._capture_edit_state()
        dialog = ValueMappingDialog(
            self, self.preset.fields, self.preset.reverse_numbering
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._commit_edit(previous_state, "문항과 선택지 이름 변경")
            self.update_canvas()

    def assign_comment_field(self):
        if not self.preset.fields:
            QMessageBox.information(
                self, "알림", "자유기입으로 전환할 문항이 없습니다."
            )
            return

        names = [f.name for f in self.preset.fields]
        name, ok = QInputDialog.getItem(
            self, "자유기입 전환", "문항 선택:", names, 0, False
        )
        if not ok or not name:
            return

        target = next((f for f in self.preset.fields if f.name == name), None)
        if not target:
            return

        if target.is_comment:
            reply = QMessageBox.question(
                self,
                "자유기입 지정 해제",
                "이미 자유기입 문항입니다. 지정을 해제할까요?",
            )
            if reply == QMessageBox.StandardButton.Yes:
                previous_state = self._capture_edit_state()
                target.is_comment = False
                self._commit_edit(previous_state, "자유기입 지정 해제")
                QMessageBox.information(
                    self, "완료", "자유기입 지정이 해제되었습니다."
                )
        else:
            previous_state = self._capture_edit_state()
            target.is_comment = True
            self._commit_edit(previous_state, "자유기입으로 지정")
            QMessageBox.information(self, "완료", "자유기입으로 지정되었습니다.")
        self.update_canvas()

    def _sanitize_config_name(self, name: str) -> str:
        invalid_chars = '<>:"/\\|?*'
        cleaned = "".join("_" if ch in invalid_chars else ch for ch in name).strip()
        return cleaned

    def _serialize_config(self) -> dict:
        return {
            "page_count": self.preset.page_count,
            "fine_angle": self.preset.fine_angle,
            "page_fine_angles": list(self.preset.page_fine_angles),
            "rot_code": self.preset.rot_code,
            "reverse_numbering": self.preset.reverse_numbering,
            "template_dilate_pct": self.preset.template_dilate_pct,
            "is_a_view": self.is_a_view,
            "fields": [f.to_dict() for f in self.preset.fields],
            "pending_boxes": [b.to_dict() for b in self.pending_boxes],
        }

    def _filter_boxes_outside_page_count(self):
        if self.preset.page_count <= 0:
            return
        self.pending_boxes = [
            b for b in self.pending_boxes if b.page_idx < self.preset.page_count
        ]
        for field in self.preset.fields:
            field.boxes = [
                b for b in field.boxes if b.page_idx < self.preset.page_count
            ]
        self.preset.fields = [f for f in self.preset.fields if f.boxes]

    def _sync_fine_angle_spin(self):
        if hasattr(self, "fine_angle_spin"):
            self.fine_angle_spin.blockSignals(True)
            self.fine_angle_spin.setValue(self.preset.fine_angle)
            self.fine_angle_spin.blockSignals(False)

    def _sync_rotation_index(self):
        if self.preset.rot_code in self.ROT_CYCLE:
            self.rot_idx = self.ROT_CYCLE.index(self.preset.rot_code)
        else:
            self.rot_idx = 0
            self.preset.rot_code = self.ROT_CYCLE[self.rot_idx]
        self._sync_rotation_actions()

    def _sync_rotation_actions(self):
        if hasattr(self, "rotation_actions"):
            for idx, action in enumerate(self.rotation_actions):
                action.blockSignals(True)
                action.setChecked(idx == self.rot_idx)
                action.blockSignals(False)

    def _configured_template_page(self, page: np.ndarray, page_idx: int) -> np.ndarray:
        """Return a page in the coordinate system used by boxes and aligners."""
        if getattr(self, "_pages_are_canonical", False):
            return page
        return apply_rotation(
            page,
            self.preset.rot_code,
            self.preset.fine_angle_for_page(page_idx),
        )

    @staticmethod
    def _build_single_sample_template_pages(
        saved_templates: list[np.ndarray],
        current_templates: dict[int, np.ndarray],
        page_transforms: dict[int, np.ndarray],
        page_count: int,
    ) -> list[np.ndarray]:
        """Warp clean preset pages into the current checkbox coordinate system."""
        if len(saved_templates) < page_count:
            return []

        transformed = []
        for page_idx in range(page_count):
            source = saved_templates[page_idx]
            target = current_templates.get(page_idx)
            matrix = page_transforms.get(page_idx)
            if (
                not isinstance(source, np.ndarray)
                or source.size == 0
                or not isinstance(target, np.ndarray)
                or target.size == 0
                or matrix is None
            ):
                return []
            matrix = np.asarray(matrix, dtype=np.float64)
            if matrix.shape != (2, 3):
                return []
            target_h, target_w = target.shape[:2]
            border_value = (255, 255, 255) if source.ndim == 3 else 255
            transformed.append(
                cv2.warpAffine(
                    source,
                    matrix,
                    (target_w, target_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=border_value,
                )
            )
        return transformed

    def _clear_inferred_display_templates(self, error_message: str = ""):
        """Clear the display-only template without changing analysis pages."""
        self._inferred_display_templates = {}
        self._analysis_validation_error = str(error_message or "")
        refresh = getattr(self, "_refresh_document_status", None)
        if callable(refresh):
            refresh()
        status_bar = getattr(self, "statusBar", None)
        if callable(status_bar):
            if self._analysis_validation_error:
                status_bar().showMessage(self._analysis_validation_error, 15000)
            else:
                status_bar().clearMessage()

    def _set_inferred_display_templates(self, templates) -> bool:
        """Use complete, coordinate-compatible inferred pages as canvas bases."""
        page_count = int(getattr(self.preset, "page_count", 0))
        pages = list(getattr(self, "pages", []))
        if page_count <= 0 or len(pages) < page_count:
            MainWindow._clear_inferred_display_templates(
                self,
                "기준 페이지가 부족해 안전한 분석 좌표를 확인할 수 없습니다.",
            )
            return False

        candidates = templates if isinstance(templates, dict) else {}
        validated: dict[int, np.ndarray] = {}
        for page_idx in range(page_count):
            template = candidates.get(page_idx)
            if (
                not isinstance(template, np.ndarray)
                or template.size == 0
                or template.ndim not in (2, 3)
                or (template.ndim == 3 and template.shape[2] != 3)
            ):
                MainWindow._clear_inferred_display_templates(
                    self,
                    f"{page_idx + 1}쪽의 자동 생성 빈 양식이 없어 분석할 수 없습니다.",
                )
                return False

            configured_page = MainWindow._configured_template_page(
                self, pages[page_idx], page_idx
            )
            if template.shape[:2] != configured_page.shape[:2]:
                expected_h, expected_w = configured_page.shape[:2]
                actual_h, actual_w = template.shape[:2]
                MainWindow._clear_inferred_display_templates(
                    self,
                    f"{page_idx + 1}쪽 자동 생성 빈 양식 크기"
                    f"({actual_w}×{actual_h})가 기준 페이지 크기"
                    f"({expected_w}×{expected_h})와 달라 분석할 수 없습니다.",
                )
                return False
            validated[page_idx] = template

        self._inferred_display_templates = validated
        self._analysis_validation_error = ""
        refresh = getattr(self, "_refresh_document_status", None)
        if callable(refresh):
            refresh()
        status_bar = getattr(self, "statusBar", None)
        if callable(status_bar):
            status_bar().clearMessage()
        return True

    def _canvas_base_page(self, page: np.ndarray, page_idx: int) -> np.ndarray:
        template = getattr(self, "_inferred_display_templates", {}).get(page_idx)
        if isinstance(template, np.ndarray) and template.size > 0:
            return template
        return MainWindow._configured_template_page(self, page, page_idx)

    def _reload_raw_template_pages(self, progress_cb=None) -> bool:
        """Restore source pages before changing a transform baked into a preset."""
        if not self.file_paths or self.preset.page_count <= 0:
            return False
        raw_pages = load_pdf_pages(
            self.file_paths[0],
            progress_cb=progress_cb,
            page_indices=list(range(self.preset.page_count)),
        )
        if not raw_pages:
            return False
        self.pages = raw_pages
        self._pages_are_canonical = False
        self._analysis_reference_pages = []
        self._single_sample_template_pages = []
        MainWindow._clear_inferred_display_templates(self)
        if len(raw_pages) < self.preset.page_count:
            self.preset.page_count = len(raw_pages)
            self._filter_boxes_outside_page_count()
        self._update_page_size()
        return True

    def _update_page_size(self):
        if not self.pages:
            self.page_H = 0
            self.page_W = 0
            return
        sample = self._configured_template_page(self.pages[0], 0)
        self.page_H, self.page_W = sample.shape[:2]

        # 가로 길이(W)가 세로 길이(H)보다 크면(가로 모드) 2페이지 보기를 강제로 끕니다.
        if self.page_W > self.page_H:
            self.is_a_view = False

        self._sync_view_toggle_text()

    def _apply_loaded_preset(
        self, data: dict, preset_name: str = "", progress_cb=None
    ):
        def report(value: int, message: str = ""):
            if progress_cb:
                progress_cb(value, message)

        previous_preset = getattr(self, "preset", None)
        previous_pages = list(getattr(self, "pages", []))
        previous_pages_are_canonical = getattr(
            self, "_pages_are_canonical", False
        )
        previous_analysis_reference_pages = list(
            getattr(self, "_analysis_reference_pages", [])
        )
        previous_single_sample_template_pages = list(
            getattr(self, "_single_sample_template_pages", [])
        )
        previous_inferred_display_templates = dict(
            getattr(self, "_inferred_display_templates", {})
        )
        previous_analysis_validation_error = getattr(
            self, "_analysis_validation_error", ""
        )
        previous_pending_boxes = list(getattr(self, "pending_boxes", []))
        previous_selected_boxes = list(getattr(self, "selected_boxes", []))
        previous_selection_anchor = getattr(self, "_selection_anchor", None)
        previous_is_a_view = getattr(self, "is_a_view", False)
        page_count = max(1, int(data.get("page_count", 1)))
        fine_angle = float(data.get("fine_angle", 0.0))
        rot_code = int(data.get("rot_code", -1))
        can_reuse_current_detection = bool(
            self.file_paths
            and previous_preset is not None
            and not previous_pages_are_canonical
            and previous_preset.page_count == page_count
            and previous_preset.rot_code == rot_code
            and abs(previous_preset.fine_angle - fine_angle) < 0.001
            and len(previous_preset.page_fine_angles) >= page_count
        )
        raw_page_angles = data.get("page_fine_angles", [])
        if not isinstance(raw_page_angles, list):
            raw_page_angles = []
        page_fine_angles = []
        for value in raw_page_angles[:page_count]:
            try:
                page_fine_angles.append(float(value))
            except (TypeError, ValueError):
                page_fine_angles.append(0.0)
        if can_reuse_current_detection:
            page_fine_angles = list(previous_preset.page_fine_angles[:page_count])

        detected_boxes_by_page: dict[int, list[Box]] | None = None
        if can_reuse_current_detection:
            detected_boxes_by_page = {}
            for field in previous_preset.fields:
                if field.is_comment:
                    continue
                for box in field.boxes:
                    detected_boxes_by_page.setdefault(box.page_idx, []).append(
                        copy.copy(box)
                    )
            if not any(detected_boxes_by_page.values()):
                detected_boxes_by_page = None
        self.preset = TemplatePreset(
            page_count=page_count,
            fine_angle=fine_angle,
            page_fine_angles=page_fine_angles,
            rot_code=rot_code,
            reverse_numbering=bool(data.get("reverse_numbering", True)),
            template_dilate_pct=float(data.get("template_dilate_pct", 0.3)),
            fields=[],
        )
        self.preset.fields = [Field.from_dict(f) for f in data.get("fields", [])]
        self.pending_boxes = [Box.from_dict(b) for b in data.get("pending_boxes", [])]
        self.is_a_view = bool(data.get("is_a_view", False))
        self.selected_boxes.clear()
        self._selection_anchor = None
        self._analysis_reference_pages = []
        self._single_sample_template_pages = []
        self._inferred_display_templates = {}
        self._analysis_validation_error = ""
        self._sync_rotation_index()
        self._sync_fine_angle_spin()
        self._sync_reverse_numbering_state()

        saved_templates = self._load_template_images(preset_name) if preset_name else []
        alignment_message = ""

        if self.file_paths:
            existing_pages = getattr(self, "pages", [])
            if (
                existing_pages
                and not getattr(self, "_pages_are_canonical", False)
                and len(existing_pages) >= self.preset.page_count
            ):
                raw_pages = list(existing_pages[: self.preset.page_count])
                report(20, "현재 PDF 페이지 준비 완료")
            else:
                raw_pages = load_pdf_pages(
                    self.file_paths[0],
                    progress_cb=self._wrap_progress(
                        0, 20, "현재 PDF 페이지 준비 중...", report
                    ),
                    page_indices=list(range(self.preset.page_count)),
                )
            if len(raw_pages) < self.preset.page_count:
                self.preset.page_count = len(raw_pages)
            self.pages = raw_pages
            self._pages_are_canonical = False
            self._filter_boxes_outside_page_count()

            if len(self.preset.page_fine_angles) < self.preset.page_count:
                self._estimate_page_fine_angles(
                    progress_cb=self._wrap_progress(
                        20, 10, "자동 수평 맞춤 중...", report
                    )
                )
            else:
                report(30, "저장된 페이지 각도 적용")

            current_templates = {}
            if self.preset.page_count > 0:
                template_progress = self._wrap_progress(
                    30, 50, "현재 체크박스 템플릿 생성 중...", report
                )
                # Preset coordinates and the canvas both use the first PDF as
                # their alignment reference. A median spanning several files
                # can blur frames when scanners use slightly different content
                # scales, so derive preset anchors from that same first PDF.
                current_templates = generate_ui_templates(
                    self.file_paths[0],
                    self.preset.page_count,
                    self.preset.rot_code,
                    self.preset.fine_angle,
                    progress_cb=template_progress,
                    page_fine_angles=self.preset.page_fine_angles,
                )

            source_templates = {
                page_idx: template
                for page_idx, template in enumerate(saved_templates)
                if page_idx < self.preset.page_count
            }
            report(82, "탐지된 체크박스 기준으로 프리셋 정렬 중...")
            remap = remap_preset_to_detected_layout(
                self.preset,
                current_templates,
                source_templates=source_templates,
                auxiliary_boxes=self.pending_boxes,
                detected_boxes_by_page=detected_boxes_by_page,
            )
            if detected_boxes_by_page and not remap.accepted:
                # The broad first-pass detector can miss small or faint boxes.
                # Retry with the preset-sized detector before deciding that the
                # form is incompatible or falling back to canonical alignment.
                preset_redetection = remap_preset_to_detected_layout(
                    self.preset,
                    current_templates,
                    source_templates=source_templates,
                    auxiliary_boxes=self.pending_boxes,
                )
                if preset_redetection.accepted or not getattr(
                    remap, "compatible", True
                ):
                    remap = preset_redetection
            if not getattr(remap, "compatible", True):
                # A clearly different form must not be warped into the saved
                # template. Restore the complete previous editing state so a
                # failed preset choice cannot leave unusable coordinates behind.
                self.preset = previous_preset or TemplatePreset()
                self.pages = previous_pages
                self._pages_are_canonical = previous_pages_are_canonical
                self._analysis_reference_pages = previous_analysis_reference_pages
                self._single_sample_template_pages = (
                    previous_single_sample_template_pages
                )
                self._inferred_display_templates = (
                    previous_inferred_display_templates
                )
                self._analysis_validation_error = (
                    previous_analysis_validation_error
                )
                self.pending_boxes = previous_pending_boxes
                self.selected_boxes = previous_selected_boxes
                self._selection_anchor = previous_selection_anchor
                self.is_a_view = previous_is_a_view
                self._sync_rotation_index()
                self._sync_fine_angle_spin()
                self._sync_reverse_numbering_state()
                self._update_page_size()
                self.update_canvas()
                raise ValueError(
                    "현재 PDF와 일치하지 않는 프리셋이라 불러올 수 없습니다."
                )
            if remap.accepted:
                self.preset = remap.config
                self.pending_boxes = list(remap.auxiliary_boxes)
                matched_keys = set(getattr(remap, "matched_box_keys", ()))
                supplied_keys = set(getattr(remap, "supplied_box_keys", ()))
                reused_current_count = 0
                if detected_boxes_by_page and supplied_keys:
                    reused_current_count = MainWindow._merge_detected_box_geometry(
                        self.preset,
                        self.pending_boxes,
                        detected_boxes_by_page,
                        matched_keys,
                        supplied_keys,
                    )

                self.pages = raw_pages
                self._pages_are_canonical = False
                self._analysis_reference_pages = [
                    current_templates[i] for i in range(self.preset.page_count)
                ]
                self._single_sample_template_pages = (
                    MainWindow._build_single_sample_template_pages(
                        saved_templates,
                        current_templates,
                        getattr(remap, "page_transforms", {}),
                        self.preset.page_count,
                    )
                )
                MainWindow._set_inferred_display_templates(
                    self, current_templates
                )
                alignment_message = (
                    "자동 탐지 위치에 프리셋 설정 적용 완료 "
                    f"({remap.matched_boxes}/{remap.expected_boxes})"
                )
                if reused_current_count:
                    alignment_message += (
                        f" · 기존 박스 {reused_current_count}개 위치 유지"
                    )
                report(95, alignment_message)
            elif (
                raw_pages
                and saved_templates
                and len(saved_templates) >= self.preset.page_count
            ):
                # 탐지가 불완전하면 좌표를 덮어쓰지 않고 기존 정렬로 안전하게 복구합니다.
                aligned = []
                for i in range(self.preset.page_count):
                    aligner = ImageAligner(saved_templates[i])
                    img = apply_rotation(
                        raw_pages[i],
                        self.preset.rot_code,
                        self.preset.fine_angle_for_page(i),
                    )
                    aligned.append(aligner.align(img))
                    report(
                        82 + int((i + 1) / self.preset.page_count * 13),
                        "기존 템플릿 정렬로 복구 중...",
                    )
                self.pages = aligned
                self._pages_are_canonical = True
                self._analysis_reference_pages = list(
                    saved_templates[: self.preset.page_count]
                )
                self._single_sample_template_pages = list(
                    saved_templates[: self.preset.page_count]
                )
                MainWindow._clear_inferred_display_templates(self)
                alignment_message = "체크박스 매칭 불완전: 기존 템플릿 정렬 사용"
            else:
                self.pages = raw_pages
                self._pages_are_canonical = False
                self._analysis_reference_pages = []
                self._single_sample_template_pages = []
                MainWindow._clear_inferred_display_templates(
                    self,
                    "현재 PDF와 프리셋의 체크박스 배치를 충분히 확인하지 못해 "
                    "분석할 수 없습니다.",
                )
                alignment_message = "체크박스 매칭 불완전: 저장 좌표 유지"

            self._update_page_size()
        elif saved_templates:
            # PDF 없이 프리셋만 로드한 경우: 저장된 템플릿을 표시
            self.pages = saved_templates[: self.preset.page_count]
            self._pages_are_canonical = True
            self._analysis_reference_pages = list(self.pages)
            self._single_sample_template_pages = list(self.pages)
            MainWindow._clear_inferred_display_templates(self)
            self._update_page_size()
            alignment_message = "저장된 프리셋 템플릿 표시"

        MainWindow._clear_edit_history(self)
        self.update_canvas()
        self._sync_view_toggle_text()
        report(100, alignment_message or "프리셋 불러오기 완료")
        if alignment_message and hasattr(self, "statusBar"):
            self.statusBar().showMessage(alignment_message, 10000)

    def save_preset(self) -> bool:
        if self.current_preset_name:
            return self._save_preset_to_name(self.current_preset_name)
        return self.save_preset_as()

    def save_preset_as(self) -> bool:
        name, ok = QInputDialog.getText(
            self, "프리셋 저장", "프리셋 이름을 입력하세요:"
        )
        if not ok:
            return False
        name = self._sanitize_config_name(name)
        if not name:
            QMessageBox.warning(self, "경고", "유효한 프리셋 이름을 입력하세요.")
            return False
        path = self.preset_dir / f"{name}.json"
        if path.exists():
            reply = QMessageBox.question(
                self,
                "프리셋 덮어쓰기",
                f"'{name}' 프리셋이 이미 있습니다. 덮어쓸까요?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return False
        return self._save_preset_to_name(name)

    def _save_template_images(self, name: str):
        if not self.pages:
            return
        for i, page in enumerate(self.pages):
            img = self._configured_template_page(page, i)
            success, buf = cv2.imencode(".png", img)
            if success:
                tpl_path = self.preset_dir / f"{name}_tpl_p{i}.png"
                tpl_path.write_bytes(buf.tobytes())

    def _load_template_images(self, name: str) -> list:
        pages = []
        i = 0
        while True:
            tpl_path = self.preset_dir / f"{name}_tpl_p{i}.png"
            if not tpl_path.exists():
                break
            img = cv2.imdecode(
                np.frombuffer(tpl_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR
            )
            if img is not None:
                pages.append(img)
            i += 1
        return pages

    def _save_preset_to_name(self, name: str) -> bool:
        try:
            validate_field_names(field.name for field in self.preset.fields)
        except ValueError as exc:
            QMessageBox.warning(self, "문항 이름 확인", str(exc))
            return False
        data = self._serialize_config()
        path = self.preset_dir / f"{name}.json"
        try:
            with open(path, "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
            self._save_template_images(name)
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"프리셋 저장 실패: {exc}")
            return False
        self.current_preset_name = name
        MainWindow._set_preset_dirty(self, False)
        if hasattr(self, "statusBar"):
            self.statusBar().showMessage(f"프리셋 '{name}' 저장 완료", 5000)
        QMessageBox.information(
            self,
            "프리셋 저장 완료",
            f"프리셋 '{name}'을(를) 저장했습니다.",
        )
        return True

    def _list_config_names(self) -> list[str]:
        return sorted(path.stem for path in self.preset_dir.glob("*.json"))

    def _load_preset_by_name(self, name: str) -> bool:
        path = self.preset_dir / f"{name}.json"
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"프리셋 불러오기 실패: {exc}")
            return False
        try:
            validate_field_names(
                Field.from_dict(field_data).name
                for field_data in data.get("fields", [])
            )
        except (TypeError, ValueError) as exc:
            QMessageBox.critical(
                self, "프리셋 문항 이름 오류", f"이 프리셋은 불러올 수 없습니다.\n\n{exc}"
            )
            return False
        if not MainWindow._confirm_save_or_discard_changes(
            self,
            "다른 프리셋을 불러오면 현재 편집 내용이 바뀝니다."
        ):
            return False

        snapshot = MainWindow._capture_document_state(self)
        progress = None
        progress_cb = None
        if self.file_paths:
            progress = self._show_progress_dialog(
                "프리셋 불러오기", "현재 문서에 프리셋 맞추는 중..."
            )
            progress_cb = self._make_progress_cb(progress)
        try:
            self._apply_loaded_preset(
                data,
                preset_name=name,
                progress_cb=progress_cb,
            )
        except Exception as exc:
            MainWindow._restore_document_state(self, snapshot)
            QMessageBox.critical(self, "오류", f"프리셋 적용 실패: {exc}")
            return False
        finally:
            if progress is not None:
                progress.close()
        self.current_preset_name = name
        MainWindow._set_preset_dirty(self, False)
        return True

    def delete_preset(self):
        names = self._list_config_names()
        if not names:
            QMessageBox.information(self, "알림", "저장된 프리셋이 없습니다.")
            return

        name, ok = QInputDialog.getItem(
            self, "프리셋 삭제", "삭제할 프리셋 선택:", names, 0, False
        )
        if not ok or not name:
            return

        reply = QMessageBox.question(
            self,
            "확인",
            f"프리셋 '{name}'을(를) 삭제하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # JSON 파일 삭제
        json_path = self.preset_dir / f"{name}.json"
        try:
            json_path.unlink(missing_ok=True)
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"프리셋 삭제 실패: {exc}")
            return

        # 템플릿 이미지 파일 삭제
        i = 0
        while True:
            tpl_path = self.preset_dir / f"{name}_tpl_p{i}.png"
            if not tpl_path.exists():
                break
            tpl_path.unlink()
            i += 1

        if self.current_preset_name == name:
            self.current_preset_name = None
            MainWindow._set_preset_dirty(self, True)

        self._refresh_document_status()

        QMessageBox.information(self, "완료", f"프리셋 '{name}'이(가) 삭제되었습니다.")

    def load_preset_dialog(self):
        names = self._list_config_names()
        if not names:
            QMessageBox.information(self, "알림", "저장된 프리셋이 없습니다.")
            return

        name, ok = QInputDialog.getItem(
            self, "프리셋 불러오기", "프리셋 선택:", names, 0, False
        )
        if not ok or not name:
            return
        self._load_preset_by_name(name)

    def load_pdf(self) -> bool:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "PDF 선택", "", "PDF Files (*.pdf)"
        )
        if not paths:
            return False

        page_count, ok = QInputDialog.getInt(
            self,
            "PDF 불러오기",
            "설문지 한 부는 몇 페이지인가요? (앞·뒤면이면 2)",
            max(1, self.preset.page_count),
            1,
            10,
        )
        if not ok:
            return False
        if not self._confirm_save_or_discard_changes(
            "새 PDF를 불러오면 현재 편집 내용이 바뀝니다."
        ):
            return False

        snapshot = self._capture_document_state()
        progress = None
        try:
            progress = self._show_progress_dialog("PDF 로드", "PDF 로딩 중...")
            progress_cb = self._make_progress_cb(progress)

            loaded_pages = load_pdf_pages(
                paths[0],
                progress_cb=self._wrap_progress(
                    0, 35, "PDF 로딩 중...", progress_cb
                ),
                page_indices=list(range(page_count)),
            )
            if len(loaded_pages) < page_count:
                raise ValueError(
                    f"첫 PDF에서 {page_count}쪽을 읽어야 하지만 "
                    f"{len(loaded_pages)}쪽만 읽었습니다."
                )

            self.file_paths = list(paths)
            self.preset.page_count = page_count
            self.pages = loaded_pages
            self._pages_are_canonical = False
            self._reset_state_for_new_pdf()
            self._estimate_page_fine_angles(
                progress_cb=self._wrap_progress(
                    35, 5, "자동 수평 맞춤 중...", progress_cb
                )
            )
            self._update_page_size()
            self.update_canvas()
            self._sync_view_toggle_text()

            # 여러 PDF의 첫 설문을 합쳐 자동 탐지 기준을 더 안정적으로 만듭니다.
            multi_templates = None
            if len(self.file_paths) > 1:
                multi_templates = generate_ui_templates_multi(
                    self.file_paths,
                    page_count,
                    self.preset.rot_code,
                    self.preset.fine_angle,
                    progress_cb=self._wrap_progress(
                        40, 30, "템플릿 병합 중...", progress_cb
                    ),
                    page_fine_angles=self.preset.page_fine_angles,
                )

            self.auto_detect(
                progress_cb=self._wrap_progress(
                    70, 30, "체크박스 탐지 중...", progress_cb
                ),
                prebuilt_templates=multi_templates,
                mark_dirty=False,
            )
            progress_cb(100, "PDF 로드 완료")
        except Exception as exc:
            self._restore_document_state(snapshot)
            guidance = (
                "페이지의 방향이나 위치를 기준 양식에 맞추지 못했습니다. "
                "함께 선택한 PDF의 문항 배치와 페이지 방향을 확인해주세요."
                if isinstance(exc, PageOrientationError)
                else "PDF를 불러오지 못했습니다. 파일이 손상되었거나 암호가 "
                "설정됐는지 확인해주세요."
            )
            QMessageBox.critical(
                self,
                "PDF 불러오기 실패",
                f"{guidance}\n\n세부 내용: {exc}",
            )
            return False
        finally:
            if progress is not None:
                progress.close()

        QMessageBox.information(self, "완료", "PDF가 성공적으로 로드되었습니다.")
        return True

    def _estimate_page_fine_angles(self, progress_cb=None) -> list[float]:
        page_count = min(self.preset.page_count, len(self.pages))
        previous = list(self.preset.page_fine_angles)
        angles = [
            previous[index] if index < len(previous) else 0.0
            for index in range(page_count)
        ]
        if page_count <= 0:
            self.preset.page_fine_angles = []
            return []

        for page_idx, page in enumerate(self.pages[:page_count]):
            base = apply_rotation(
                page, self.preset.rot_code, self.preset.fine_angle
            )
            estimated = estimate_deskew_angle(base)
            if estimated is not None:
                angles[page_idx] = float(estimated)
            if progress_cb:
                progress_cb(
                    int((page_idx + 1) / page_count * 100),
                    f"자동 수평 맞춤 중... ({page_idx + 1}/{page_count})",
                )

        self.preset.page_fine_angles = angles
        return angles

    def _page_angle_summary(self) -> str:
        if not self.preset.page_fine_angles:
            return "보정 없음"
        return ", ".join(
            f"{index + 1}쪽 {angle:+.1f}°"
            for index, angle in enumerate(self.preset.page_fine_angles)
        )

    def auto_deskew_pages(self):
        if not self.pages or not self.file_paths:
            QMessageBox.information(self, "알림", "먼저 PDF를 불러와주세요.")
            return

        def estimate(progress_cb):
            self._estimate_page_fine_angles(progress_cb)
            self._update_page_size()

        if self._redetect_checkboxes_with_progress(
            "기울기 다시 맞추기", before_detect=estimate
        ):
            self.statusBar().showMessage(
                f"기울기 다시 맞추기 완료: {self._page_angle_summary()}",
                10000,
            )

    def open_manual_fine_angle_dialog(self):
        if not self.pages or not self.file_paths:
            QMessageBox.information(self, "알림", "먼저 PDF를 불러와주세요.")
            return

        angle, ok = QInputDialog.getDouble(
            self,
            "수동 기울기 조정",
            "자동 보정 후에도 기울어진 경우에만 각도를 입력하세요.",
            value=self.preset.fine_angle,
            min=-10.0,
            max=10.0,
            decimals=1,
            step=0.1,
        )
        if not ok or abs(float(angle) - self.preset.fine_angle) <= 0.0001:
            return
        self.change_fine_angle(float(angle))

    def _redetect_checkboxes_with_progress(
        self, title: str, before_detect=None
    ) -> bool:
        """Run a full checkbox re-detection with the same progress UI as loading."""
        if not self.pages or not self.file_paths:
            return False

        progress = self._show_progress_dialog(title, "체크박스 재탐색 준비 중...")
        progress_cb = self._make_progress_cb(progress)
        try:
            needs_raw_pages = getattr(self, "_pages_are_canonical", False)
            preparation_steps = int(needs_raw_pages) + int(before_detect is not None)
            preparation_end = 10 if preparation_steps else 0
            completed_steps = 0

            if needs_raw_pages:
                step_start = int(
                    preparation_end * completed_steps / preparation_steps
                )
                step_end = int(
                    preparation_end * (completed_steps + 1) / preparation_steps
                )
                self._reload_raw_template_pages(
                    self._wrap_progress(
                        step_start,
                        step_end - step_start,
                        "원본 페이지 준비 중...",
                        progress_cb,
                    )
                )
                completed_steps += 1

            if before_detect is not None:
                step_start = int(
                    preparation_end * completed_steps / preparation_steps
                )
                step_end = int(
                    preparation_end * (completed_steps + 1) / preparation_steps
                )
                before_detect(
                    self._wrap_progress(
                        step_start,
                        step_end - step_start,
                        "재탐색 준비 중...",
                        progress_cb,
                    )
                )
                detect_progress = self._wrap_progress(
                    preparation_end,
                    100 - preparation_end,
                    "체크박스 재탐색 중...",
                    progress_cb,
                )
            else:
                detect_progress = (
                    self._wrap_progress(
                        preparation_end,
                        100 - preparation_end,
                        "체크박스 재탐색 중...",
                        progress_cb,
                    )
                    if preparation_end
                    else progress_cb
                )
            self.auto_detect(progress_cb=detect_progress)
            progress_cb(100, "체크박스 재탐색 완료")
            return True
        finally:
            progress.close()

    def clear_cache(self):
        """체크박스 탐지 캐시를 모두 삭제하고 재탐지를 수행합니다."""
        reply = QMessageBox.question(
            self,
            "캐시 삭제",
            "탐지 캐시를 삭제하고 체크박스를 다시 탐지할까요?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        if self.pages and self.file_paths:
            def delete_cache(progress_cb, clear_cache_fn=clear_all_cache):
                progress_cb(0, "캐시 삭제 중...")
                clear_cache_fn()
                progress_cb(100, "캐시 삭제 완료")

            self._redetect_checkboxes_with_progress(
                "캐시 삭제 후 재탐색", before_detect=delete_cache
            )
        else:
            clear_all_cache()
            QMessageBox.information(self, "완료", "캐시가 삭제되었습니다.")

    def change_fine_angle(self, angle: float):
        """Apply a confirmed manual correction and redetect the boxes once."""
        changed = abs(self.preset.fine_angle - angle) > 0.0001
        self.preset.fine_angle = angle
        if changed:
            MainWindow._set_preset_dirty(self, True)
        if not self.pages:
            return
        self._update_page_size()
        self._redetect_checkboxes_with_progress("미세 회전 적용")

    def change_rotation(self, index: int):
        """메뉴에서 회전 방향을 선택하면 동작합니다."""
        # 중복 실행 방지
        if self.rot_idx == index:
            return

        self.rot_idx = index

        # 변경된 콤보박스 항목 순서에 맞게 OpenCV 회전 코드 매핑
        # 0: 원본 0°, 1: 좌측 90°, 2: 우측 90°, 3: 180°
        self.preset.rot_code = ROTATION_MAP.get(index, -1)
        self.preset.page_fine_angles = [
            0.0 for _ in range(min(self.preset.page_count, len(self.pages)))
        ]
        self._sync_rotation_actions()
        MainWindow._set_preset_dirty(self, True)

        if not self.pages:
            return

        def estimate(progress_cb):
            self._estimate_page_fine_angles(progress_cb)
            self._update_page_size()

        self._redetect_checkboxes_with_progress(
            "회전 적용", before_detect=estimate
        )

    def toggle_view(self):
        if self.preset.page_count > 1:
            # 가로보기(용지 방향이 가로)일 때는 2열 보기를 제한
            if self.page_W > self.page_H and not self.is_a_view:
                QMessageBox.information(
                    self,
                    "알림",
                    "가로 모드(너비가 넓음)에서는 2페이지 나란히 보기를 지원하지 않습니다.",
                )
                return

            self.is_a_view = not self.is_a_view
            self.update_canvas()
            self._sync_view_toggle_text()
            MainWindow._set_preset_dirty(self, True)

    def update_canvas(self):
        if not self.pages:
            self._refresh_edit_status()
            return

        drawn_pages = []
        for i, page in enumerate(self.pages):
            img = self._canvas_base_page(page, i)
            canvas_img = img.copy()
            if canvas_img.ndim == 2:
                canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_GRAY2BGR)

            if canvas_img.shape[:2] != (self.page_H, self.page_W):
                canvas_img = cv2.resize(canvas_img, (self.page_W, self.page_H))

            text_entries = []
            for field in self.preset.fields:
                for b in field.boxes:
                    if b.page_idx == i:
                        selected = self._box_is_selected(b)
                        if selected:
                            color = (0, 210, 255)
                        elif field.is_comment:
                            color = (180, 60, 180)
                        else:
                            color = (0, 165, 70)
                        thick = 4 if selected else 2
                        cv2.rectangle(
                            canvas_img, (b.x, b.y), (b.x + b.w, b.y + b.h), color, thick
                        )
                        label = field.name
                        if field.is_comment and not label.startswith("자유기입"):
                            label = f"[자유기입] {label}"
                        text_entries.append((label, b.x, max(0, b.y - 20), color))
                        if selected and len(self.selected_boxes) == 1:
                            self._draw_resize_handles(canvas_img, b)

            if text_entries:
                canvas_img = self._draw_texts(canvas_img, text_entries)

            for b in self.pending_boxes:
                if b.page_idx == i:
                    selected = self._box_is_selected(b)
                    color = (0, 210, 255) if selected else (230, 120, 20)
                    thick = 4 if selected else 2
                    cv2.rectangle(
                        canvas_img, (b.x, b.y), (b.x + b.w, b.y + b.h), color, thick
                    )
                    if selected and len(self.selected_boxes) == 1:
                        self._draw_resize_handles(canvas_img, b)

            drawn_pages.append(canvas_img)

        if self.is_a_view:
            rows = []
            for i in range(0, len(drawn_pages), 2):
                p1 = drawn_pages[i]
                if i + 1 < len(drawn_pages):
                    p2 = drawn_pages[i + 1]
                else:
                    p2 = np.ones_like(p1) * 255
                rows.append(np.hstack((p1, p2)))
            stitched = np.vstack(rows)
        else:
            stitched = np.vstack(drawn_pages)

        self.canvas.set_image(stitched)
        self._refresh_edit_status()

    @staticmethod
    def _draw_resize_handles(image: np.ndarray, box: Box):
        radius = 5
        for x, y in (
            (box.x, box.y),
            (box.x + box.w, box.y),
            (box.x, box.y + box.h),
            (box.x + box.w, box.y + box.h),
        ):
            cv2.rectangle(
                image,
                (x - radius, y - radius),
                (x + radius, y + radius),
                (0, 210, 255),
                -1,
            )
            cv2.rectangle(
                image,
                (x - radius, y - radius),
                (x + radius, y + radius),
                (70, 70, 70),
                1,
            )

    def _box_is_selected(self, target: Box) -> bool:
        return any(box is target for box in self.selected_boxes)

    def _box_from_stitched_rect(self, st_x, st_y, width, height) -> Box | None:
        if self.page_W <= 0 or self.page_H <= 0 or width <= 0 or height <= 0:
            return None
        st_x = int(round(st_x))
        st_y = int(round(st_y))
        width = int(round(width))
        height = int(round(height))
        if st_x < 0 or st_y < 0:
            return None

        if self.is_a_view:
            column = st_x // self.page_W
            row = st_y // self.page_H
            if column not in (0, 1):
                return None
            page_idx = row * 2 + column
            local_x = st_x - column * self.page_W
            local_y = st_y - row * self.page_H
        else:
            page_idx = st_y // self.page_H
            local_x = st_x
            local_y = st_y - page_idx * self.page_H

        if not (0 <= page_idx < self.preset.page_count):
            return None
        right = min(self.page_W, local_x + width)
        bottom = min(self.page_H, local_y + height)
        local_x = max(0, min(self.page_W - 1, local_x))
        local_y = max(0, min(self.page_H - 1, local_y))
        if right - local_x <= 5 or bottom - local_y <= 5:
            return None
        return Box(
            page_idx,
            local_x,
            local_y,
            right - local_x,
            bottom - local_y,
        )

    def add_pending_box_from_stitched(self, st_x, st_y, w, h):
        box = self._box_from_stitched_rect(st_x, st_y, w, h)
        if box is None:
            return
        previous_state = self._capture_edit_state()
        self.pending_boxes.append(box)
        self.selected_boxes = [box]
        self._commit_edit(previous_state, "선택지 추가")
        self.update_canvas()

    def _unique_field_name(self, base: str) -> str:
        existing = {field.name for field in self.preset.fields}
        if base not in existing:
            return base
        number = 2
        while f"{base} {number}" in existing:
            number += 1
        return f"{base} {number}"

    def add_comment_box_from_stitched(self, st_x, st_y, w, h):
        box = self._box_from_stitched_rect(st_x, st_y, w, h)
        if box is None:
            return
        previous_state = self._capture_edit_state()
        field = Field(
            name=self._unique_field_name("자유기입"),
            boxes=[box],
            is_comment=True,
        )
        self.preset.fields.append(field)
        self.selected_boxes = [box]
        self._commit_edit(previous_state, "자유기입 영역 추가")
        self.update_canvas()

    def get_stitched_rect(self, box: Box) -> QRectF:
        if self.is_a_view:
            row = box.page_idx // 2
            col = box.page_idx % 2
            return QRectF(
                col * self.page_W + box.x, row * self.page_H + box.y, box.w, box.h
            )
        else:
            row = box.page_idx
            return QRectF(box.x, row * self.page_H + box.y, box.w, box.h)

    def box_at_stitched(self, x, y) -> Box | None:
        matches = [
            box
            for box in self._all_boxes()
            if self.get_stitched_rect(box).contains(x, y)
        ]
        return min(matches, key=lambda box: box.w * box.h) if matches else None

    def select_box(self, box: Box, additive: bool = False) -> bool:
        already_selected = self._box_is_selected(box)
        if additive and already_selected:
            self.selected_boxes = [
                selected for selected in self.selected_boxes if selected is not box
            ]
            remains_selected = False
            if getattr(self, "_selection_anchor", None) is box:
                self._selection_anchor = (
                    self.selected_boxes[-1] if self.selected_boxes else None
                )
        elif already_selected:
            remains_selected = True
            self._selection_anchor = box
        else:
            if not additive:
                self.selected_boxes.clear()
            self.selected_boxes.append(box)
            self._selection_anchor = box
            remains_selected = True
        self.update_canvas()
        return remains_selected

    def _boxes_in_selection_range(self, anchor: Box, target: Box) -> list[Box]:
        if anchor.page_idx != target.page_idx:
            return [target]
        left = min(anchor.x, target.x)
        top = min(anchor.y, target.y)
        right = max(anchor.x + anchor.w, target.x + target.w)
        bottom = max(anchor.y + anchor.h, target.y + target.h)
        return [
            box
            for box in self._all_boxes()
            if box.page_idx == anchor.page_idx
            and left <= box.x + box.w / 2 <= right
            and top <= box.y + box.h / 2 <= bottom
        ]

    def select_box_range(self, box: Box, additive: bool = False) -> bool:
        all_boxes = self._all_boxes()
        anchor = getattr(self, "_selection_anchor", None)
        anchor_is_valid = anchor is not None and any(
            anchor is candidate for candidate in all_boxes
        )
        if not anchor_is_valid or not self._box_is_selected(anchor):
            anchor = next(
                (
                    selected
                    for selected in reversed(self.selected_boxes)
                    if any(selected is candidate for candidate in all_boxes)
                ),
                box,
            )
            self._selection_anchor = anchor

        range_boxes = self._boxes_in_selection_range(anchor, box)
        if not additive:
            self.selected_boxes.clear()
        for candidate in range_boxes:
            if not self._box_is_selected(candidate):
                self.selected_boxes.append(candidate)
        self.update_canvas()
        return self._box_is_selected(box)

    def clear_box_selection(self):
        if not self.selected_boxes:
            self._selection_anchor = None
            return
        self.selected_boxes.clear()
        self._selection_anchor = None
        self.update_canvas()

    def resize_handle_at_stitched(self, x, y, tolerance: float):
        if len(self.selected_boxes) != 1:
            return None
        box = self.selected_boxes[0]
        rect = self.get_stitched_rect(box)
        handles = {
            "nw": (rect.left(), rect.top()),
            "ne": (rect.right(), rect.top()),
            "sw": (rect.left(), rect.bottom()),
            "se": (rect.right(), rect.bottom()),
        }
        for name, (handle_x, handle_y) in handles.items():
            if abs(x - handle_x) <= tolerance and abs(y - handle_y) <= tolerance:
                return box, name
        return None

    def bounded_move_delta(self, boxes: list[Box], dx, dy) -> tuple[int, int]:
        if not boxes:
            return 0, 0
        dx = int(round(dx))
        dy = int(round(dy))
        min_dx = max(-box.x for box in boxes)
        max_dx = min(self.page_W - box.x - box.w for box in boxes)
        min_dy = max(-box.y for box in boxes)
        max_dy = min(self.page_H - box.y - box.h for box in boxes)
        return (
            max(min_dx, min(max_dx, dx)),
            max(min_dy, min(max_dy, dy)),
        )

    def move_selected_boxes(self, dx, dy):
        if not self.selected_boxes:
            return
        dx, dy = self.bounded_move_delta(self.selected_boxes, dx, dy)
        if dx == 0 and dy == 0:
            return
        previous_state = self._capture_edit_state()
        for box in self.selected_boxes:
            box.x += dx
            box.y += dy
        self._commit_edit(previous_state, "선택 영역 이동")
        self.update_canvas()

    def _resized_geometry(self, box: Box, handle: str, dx, dy):
        min_size = 8
        dx = int(round(dx))
        dy = int(round(dy))
        left = box.x
        top = box.y
        right = box.x + box.w
        bottom = box.y + box.h
        if "w" in handle:
            left = max(0, min(right - min_size, left + dx))
        if "e" in handle:
            right = min(self.page_W, max(left + min_size, right + dx))
        if "n" in handle:
            top = max(0, min(bottom - min_size, top + dy))
        if "s" in handle:
            bottom = min(self.page_H, max(top + min_size, bottom + dy))
        return left, top, right - left, bottom - top

    def preview_resized_stitched_rect(self, box: Box, handle: str, dx, dy) -> QRectF:
        x, y, width, height = self._resized_geometry(box, handle, dx, dy)
        return self.get_stitched_rect(
            Box(box.page_idx, x, y, width, height)
        )

    def resize_box(self, box: Box, handle: str, dx, dy):
        if box is None or handle not in {"nw", "ne", "sw", "se"}:
            return
        geometry = self._resized_geometry(box, handle, dx, dy)
        if geometry == (box.x, box.y, box.w, box.h):
            return
        previous_state = self._capture_edit_state()
        box.x, box.y, box.w, box.h = geometry
        self._commit_edit(previous_state, "선택 영역 크기 변경")
        self.update_canvas()

    def handle_selection_from_stitched(self, x, y, w, h, additive):
        sel_rect = QRectF(x, y, w, h)

        if w < 5 and h < 5:
            clicked = self.box_at_stitched(x, y)
            if clicked is not None:
                self.select_box(clicked, additive=additive)
                return
            if not additive:
                self.selected_boxes.clear()
                self._selection_anchor = None
        else:
            if not additive:
                self.selected_boxes.clear()
            intersecting = []
            for b in self._all_boxes():
                b_rect = self.get_stitched_rect(b)
                if sel_rect.intersects(b_rect):
                    intersecting.append(b)
                    if not self._box_is_selected(b):
                        self.selected_boxes.append(b)
            if intersecting:
                self._selection_anchor = self._boxes_in_reading_order(
                    intersecting
                )[0]
            elif not additive:
                self._selection_anchor = None

        self.update_canvas()

    def _field_for_box(self, target: Box) -> Field | None:
        for field in self.preset.fields:
            if any(box is target for box in field.boxes):
                return field
        return None

    @staticmethod
    def _remove_box_ids_from_field(
        field: Field, removed_ids: set[int], default_reverse: bool = False
    ):
        old_count = len(field.boxes)
        reverse = field.effective_reverse_numbering(default_reverse)
        physical_values = []
        kept_boxes = []
        for index, box in enumerate(field.boxes, start=1):
            if id(box) in removed_ids:
                continue
            map_index = old_count - index if reverse else index - 1
            physical_values.append(
                field.value_map[map_index] if map_index < len(field.value_map) else ""
            )
            kept_boxes.append(box)
        if field.is_comment:
            field.boxes = kept_boxes
            return
        kept_values = [""] * len(kept_boxes)
        for index, value in enumerate(physical_values, start=1):
            map_index = len(kept_boxes) - index if reverse else index - 1
            kept_values[map_index] = value
        field.boxes = kept_boxes
        field.value_map = kept_values + field.value_map[old_count:]

    def delete_selected_boxes(self, confirm: bool = True):
        if not self.selected_boxes:
            return
        selected_ids = {id(box) for box in self.selected_boxes}
        removes_question = any(
            field.boxes
            and all(id(box) in selected_ids for box in field.boxes)
            for field in self.preset.fields
        )
        if confirm and (len(selected_ids) > 1 or removes_question):
            detail = (
                "선택한 박스를 삭제하면 문항 전체도 함께 삭제됩니다."
                if removes_question
                else f"선택한 박스 {len(selected_ids)}개를 삭제할까요?"
            )
            reply = QMessageBox.question(
                self,
                "선택 삭제",
                f"{detail}\n삭제 후 Ctrl+Z로 되돌릴 수 있습니다.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        previous_state = self._capture_edit_state()
        self.pending_boxes = [
            box for box in self.pending_boxes if id(box) not in selected_ids
        ]
        for field in self.preset.fields:
            self._remove_box_ids_from_field(
                field, selected_ids, self.preset.reverse_numbering
            )
        self.preset.fields = [field for field in self.preset.fields if field.boxes]
        self.selected_boxes.clear()
        self._commit_edit(previous_state, f"선택 영역 {len(selected_ids)}개 삭제")
        self.update_canvas()

    def delete_question(self, field: Field, confirm: bool = True):
        if not any(current is field for current in self.preset.fields):
            return
        if confirm:
            reply = QMessageBox.question(
                self,
                "문항 전체 삭제",
                f"'{field.name}' 문항과 선택지 {len(field.boxes)}개를 삭제할까요?\n"
                "삭제 후 Ctrl+Z로 되돌릴 수 있습니다.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        previous_state = self._capture_edit_state()
        removed_ids = {id(box) for box in field.boxes}
        self.preset.fields = [
            current for current in self.preset.fields if current is not field
        ]
        self.selected_boxes = [
            box for box in self.selected_boxes if id(box) not in removed_ids
        ]
        self._commit_edit(previous_state, f"'{field.name}' 문항 삭제")
        self.update_canvas()

    def toggle_comment_field(self, field: Field):
        if not any(current is field for current in self.preset.fields):
            return
        previous_state = self._capture_edit_state()
        field.is_comment = not field.is_comment
        description = (
            "자유기입으로 지정" if field.is_comment else "자유기입 지정 해제"
        )
        self._commit_edit(previous_state, description)
        self.update_canvas()

    def show_box_context_menu(self, global_pos, clicked: Box):
        menu = QMenu(self)
        delete_selected_action = menu.addAction("선택한 영역 삭제")
        group_action = menu.addAction("선택한 영역을 새 문항으로 묶기")
        field = self._field_for_box(clicked)
        delete_question_action = None
        comment_action = None
        if field is not None:
            menu.addSeparator()
            comment_action = menu.addAction(
                "자유기입 지정 해제"
                if field.is_comment
                else "자유기입으로 지정"
            )
            delete_question_action = menu.addAction(f"'{field.name}' 문항 전체 삭제")
        menu.addSeparator()
        help_action = menu.addAction("이 영역 도움말")
        chosen = menu.exec(global_pos)
        if chosen is delete_selected_action:
            self.delete_selected_boxes()
        elif chosen is group_action:
            self.group_boxes()
        elif comment_action is not None and chosen is comment_action:
            self.toggle_comment_field(field)
        elif delete_question_action is not None and chosen is delete_question_action:
            self.delete_question(field)
        elif chosen is help_action:
            self.help_controller.show_help(self.canvas, global_pos)

    def group_boxes(self):
        if not self.selected_boxes:
            QMessageBox.warning(
                self,
                "알림",
                "선택된 영역이 없습니다.\n선택 모드에서 박스를 클릭하거나 드래그해주세요.",
            )
            return

        name, ok = QInputDialog.getText(
            self,
            "문항 만들기",
            "문항 이름을 입력하세요",
        )
        name = name.strip() if ok else ""
        if not name:
            return
        selected = self._boxes_in_reading_order(self.selected_boxes)
        selected_ids = {id(box) for box in selected}
        surviving_names = [
            field.name
            for field in self.preset.fields
            if any(id(box) not in selected_ids for box in field.boxes)
        ]
        try:
            validate_field_names([*surviving_names, name])
        except ValueError as exc:
            QMessageBox.warning(self, "문항 이름 확인", str(exc))
            return
        previous_state = self._capture_edit_state()
        mapped_values = {}
        for field in self.preset.fields:
            count = len(field.boxes)
            reverse = field.effective_reverse_numbering(
                self.preset.reverse_numbering
            )
            for index, box in enumerate(field.boxes, start=1):
                if id(box) in selected_ids:
                    map_index = count - index if reverse else index - 1
                    mapped_values[id(box)] = (
                        field.value_map[map_index]
                        if map_index < len(field.value_map)
                        else ""
                    )
        self.pending_boxes = [
            box for box in self.pending_boxes if id(box) not in selected_ids
        ]
        for field in self.preset.fields:
            self._remove_box_ids_from_field(
                field, selected_ids, self.preset.reverse_numbering
            )
        self.preset.fields = [field for field in self.preset.fields if field.boxes]
        new_reverse = bool(self.preset.reverse_numbering)
        new_values = [""] * len(selected)
        for index, box in enumerate(selected, start=1):
            map_index = len(selected) - index if new_reverse else index - 1
            new_values[map_index] = mapped_values.get(id(box), "")
        new_field = Field(
            name=name,
            boxes=selected,
            value_map=new_values,
            reverse_numbering=new_reverse,
        )
        self.preset.fields.append(new_field)
        self.selected_boxes = list(new_field.boxes)
        self._commit_edit(previous_state, f"'{name}' 문항 만들기")
        self.update_canvas()

    def auto_detect(
        self, progress_cb=None, prebuilt_templates=None, mark_dirty: bool = True
    ):
        """
        체크박스를 자동으로 탐지하고, 수평으로 같은 라인에 있는 항목을
        Q1, Q2, Q3 등의 문항(Field)으로 자동 할당합니다.

        prebuilt_templates: load_pdf에서 이미 생성한 병합 템플릿 (중복 생성 방지)
        """
        if not self.pages or not self.file_paths:
            return

        def report(value: int, message: str = ""):
            if progress_cb:
                progress_cb(value, message)

        self.selected_boxes.clear()
        self.pending_boxes.clear()
        self.preset.fields.clear()
        self._single_sample_template_pages = []

        def template_progress(value: int, message: str = ""):
            mapped = int(value * 0.7)
            report(mapped, message or "템플릿 생성 중...")

        if prebuilt_templates is not None:
            templates = prebuilt_templates
            report(0, "병합 템플릿 사용")
        else:
            report(0, "템플릿 생성 중...")
            if len(self.file_paths) > 1:
                templates = generate_ui_templates_multi(
                    self.file_paths,
                    self.preset.page_count,
                    self.preset.rot_code,
                    self.preset.fine_angle,
                    progress_cb=template_progress,
                    page_fine_angles=self.preset.page_fine_angles,
                )
            else:
                templates = generate_ui_templates(
                    self.file_paths[0],
                    self.preset.page_count,
                    self.preset.rot_code,
                    self.preset.fine_angle,
                    progress_cb=template_progress,
                    page_fine_angles=self.preset.page_fine_angles,
                )
        MainWindow._set_inferred_display_templates(self, templates)

        # ── 캐시 확인 ──
        cached = load_checkbox_cache(
            self.file_paths,
            self.preset.page_count,
            self.preset.rot_code,
            self.preset.fine_angle,
            self.preset.page_fine_angles,
        )
        if cached is not None:
            report(70, "캐시된 체크박스 불러오는 중...")
            question_number = 1
            for page_idx in sorted(cached.keys()):
                boxes = [Box(page_idx, *b) for b in cached[page_idx]]
                rows = self._group_boxes_by_row(boxes)
                for row in rows:
                    row.sort(key=lambda b: b.x)
                    field_name = f"Q{question_number}"
                    self.preset.fields.append(Field(name=field_name, boxes=row))
                    question_number += 1
            report(100, "체크박스 탐지 완료 (캐시)")
            MainWindow._clear_edit_history(self)
            self.update_canvas()
            if mark_dirty:
                MainWindow._set_preset_dirty(self, True)
            return

        question_number = 1

        total_pages = len(self.pages)
        report(70, "체크박스 탐지 중...")

        detected_cache: dict[int, list[tuple[int, int, int, int]]] = {}

        for i, page in enumerate(self.pages):
            if templates and i in templates:
                img = templates[i]
            else:
                img = self._configured_template_page(page, i)

            detected = auto_detect_checkboxes(img)
            detected_cache[i] = detected

            # Box 객체로 변환
            boxes = [Box(i, b[0], b[1], b[2], b[3]) for b in detected]

            rows = self._group_boxes_by_row(boxes)

            # 2. 각 줄을 왼쪽부터 오른쪽으로 정렬한 뒤 Field(문항)로 할당
            for row in rows:
                row.sort(key=lambda b: b.x)

                field_name = f"Q{question_number}"
                new_field = Field(name=field_name, boxes=row)
                self.preset.fields.append(new_field)
                question_number += 1

            if total_pages > 0:
                progress_value = 70 + int((i + 1) / total_pages * 30)
                report(progress_value, f"체크박스 탐지 중... ({i + 1}/{total_pages})")

        # ── 캐시 저장 ──
        save_checkbox_cache(
            self.file_paths,
            self.preset.page_count,
            self.preset.rot_code,
            self.preset.fine_angle,
            detected_cache,
            self.preset.page_fine_angles,
        )

        report(100, "체크박스 탐지 완료")
        MainWindow._clear_edit_history(self)
        self.update_canvas()
        if mark_dirty:
            MainWindow._set_preset_dirty(self, True)

    def _analysis_input_pages(self) -> tuple[list[np.ndarray], bool]:
        """Return stable analysis references separately from display pages."""
        saved_references = list(
            getattr(self, "_analysis_reference_pages", [])
        )
        use_saved_references = (
            len(saved_references) >= self.preset.page_count
            and self.preset.page_count > 0
        )
        if use_saved_references:
            return saved_references[: self.preset.page_count], True
        return list(self.pages), bool(self._pages_are_canonical)

    def _analysis_input_validation_message(
        self, analysis_pages: list[np.ndarray]
    ) -> str:
        existing_error = getattr(self, "_analysis_validation_error", "")
        if existing_error:
            return existing_error
        page_count = int(getattr(self.preset, "page_count", 0))
        if page_count <= 0 or len(analysis_pages) < page_count:
            return "분석에 필요한 페이지별 기준 템플릿이 모두 준비되지 않았습니다."
        for page_idx, page in enumerate(analysis_pages[:page_count]):
            if not isinstance(page, np.ndarray) or page.size == 0:
                return f"{page_idx + 1}쪽 기준 템플릿이 비어 있어 분석할 수 없습니다."
        return ""

    def execute_analysis(self):
        if not self.file_paths or not self.preset.fields:
            QMessageBox.warning(self, "경고", "파일이나 생성된 템플릿 항목이 없습니다.")
            return
        try:
            validate_field_names(field.name for field in self.preset.fields)
        except ValueError as exc:
            QMessageBox.warning(self, "문항 이름 확인", str(exc))
            return
        if self._analysis_thread is not None and self._analysis_thread.isRunning():
            QMessageBox.information(self, "알림", "이미 분석이 진행 중입니다.")
            return

        analysis_pages, pages_preprocessed = self._analysis_input_pages()
        validation_message = self._analysis_input_validation_message(analysis_pages)
        if validation_message:
            QMessageBox.warning(
                self,
                "분석 실행 불가",
                f"{validation_message}\n\nPDF와 프리셋을 확인한 뒤 다시 탐지해주세요.",
            )
            return

        self._analysis_progress = self._show_progress_dialog(
            "분석", "분석 준비 중..."
        )
        self._analysis_progress.setMinimumWidth(680)
        self._analysis_timing = ProgressTiming()
        self._analysis_last_progress = 0.0
        self._analysis_last_message = "분석 준비 중..."
        self._analysis_elapsed_text = ""
        self._refresh_analysis_progress()
        self._analysis_timer.start()
        self._analysis_result = None
        self.exec_btn.setEnabled(False)

        thread = QThread(self)
        worker = _AnalysisWorker(
            list(self.file_paths),
            analysis_pages,
            copy.deepcopy(self.preset),
            template_pages_preprocessed=pages_preprocessed,
            single_sample_template_pages=list(
                getattr(self, "_single_sample_template_pages", [])
            ),
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._update_analysis_progress)
        worker.finished.connect(self._store_analysis_result)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._finish_analysis)
        thread.finished.connect(thread.deleteLater)

        self._analysis_thread = thread
        self._analysis_worker = worker
        thread.start()

    @pyqtSlot(float, str)
    def _update_analysis_progress(self, value: float, message: str):
        if self._analysis_progress is None:
            return
        self._analysis_last_progress = max(0.0, min(100.0, float(value)))
        if message:
            self._analysis_last_message = message
        self._refresh_analysis_progress()

    @pyqtSlot()
    def _refresh_analysis_progress(self):
        if self._analysis_progress is None or self._analysis_timing is None:
            return
        self._analysis_progress.setValue(int(self._analysis_last_progress))
        self._analysis_progress.setLabelText(
            self._analysis_timing.label(
                self._analysis_last_progress, self._analysis_last_message
            )
        )

    @pyqtSlot(bool, str)
    def _store_analysis_result(self, success: bool, error_message: str):
        self._analysis_result = (success, error_message)
        self._analysis_timer.stop()
        if self._analysis_timing is not None:
            self._analysis_elapsed_text = format_duration(
                self._analysis_timing.elapsed_seconds()
            )
        if self._analysis_progress is not None:
            if success:
                self._analysis_last_progress = 100.0
                self._analysis_last_message = "완료"
                self._refresh_analysis_progress()
            self._analysis_progress.close()

    @pyqtSlot()
    def _finish_analysis(self):
        success, error_message = self._analysis_result or (
            False,
            "분석 작업이 예기치 않게 종료되었습니다.",
        )

        if self._analysis_progress is not None:
            self._analysis_progress.close()
        self._analysis_timer.stop()
        self._analysis_progress = None
        self._analysis_timing = None
        self._analysis_worker = None
        self._analysis_thread = None
        self._analysis_result = None
        self.exec_btn.setEnabled(True)

        if success:
            elapsed = self._analysis_elapsed_text or "00:00"
            QMessageBox.information(
                self,
                "완료",
                "분석이 완료되었습니다.\n\n"
                "결과는 실행 파일 옆 '결과' 폴더에 저장되었습니다.\n\n"
                f"총 소요 시간: {elapsed}",
            )
        elif error_message:
            elapsed_suffix = (
                f"\n\n소요 시간: {self._analysis_elapsed_text}"
                if self._analysis_elapsed_text
                else ""
            )
            QMessageBox.critical(
                self,
                "오류",
                f"분석 중 문제가 발생했습니다.\n\n{error_message}{elapsed_suffix}",
            )
        else:
            elapsed_suffix = (
                f"\n\n소요 시간: {self._analysis_elapsed_text}"
                if self._analysis_elapsed_text
                else ""
            )
            QMessageBox.critical(
                self, "오류", f"분석 중 문제가 발생했습니다.{elapsed_suffix}"
            )
        self._analysis_elapsed_text = ""

    def closeEvent(self, a0: QCloseEvent):
        if self._analysis_thread is not None and self._analysis_thread.isRunning():
            QMessageBox.information(
                self,
                "분석 진행 중",
                "분석이 끝난 뒤 프로그램을 종료해주세요.",
            )
            a0.ignore()
            return
        if not self._confirm_save_or_discard_changes(
            "프로그램을 종료하면 저장하지 않은 편집 내용이 사라집니다."
        ):
            a0.ignore()
            return
        super().closeEvent(a0)
