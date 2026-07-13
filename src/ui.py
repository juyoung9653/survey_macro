import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QActionGroup, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
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

from .models import Box, Field, TemplatePreset
from .processor import generate_ui_templates, generate_ui_templates_multi, run_analysis
from .vision import (
    ImageAligner,
    apply_rotation,
    auto_detect_checkboxes,
    clear_all_cache,
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

    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 줌(확대/축소) 시 마우스 커서 위치를 중심으로 하도록 설정
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self.drawing = False  # 좌클릭 (박스 그리기)
        self.selecting = False  # 우클릭 (박스 선택하기)

        self.start_pos = None
        self.temp_rect = None  # 그리기용 파란색 임시 박스
        self.select_rect = None  # 범위선택용 빨간 점선 박스

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

    # --- 마우스 클릭 및 드래그 (좌클릭: 그리기, 우클릭: 선택) ---
    def mousePressEvent(self, event):
        self.start_pos = self.mapToScene(event.pos())
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.temp_rect = self.scene.addRect(
                QRectF(self.start_pos, self.start_pos), QPen(Qt.GlobalColor.blue, 2)
            )
        elif event.button() == Qt.MouseButton.RightButton:
            self.selecting = True
            self.select_rect = self.scene.addRect(
                QRectF(self.start_pos, self.start_pos),
                QPen(Qt.GlobalColor.red, 1, Qt.PenStyle.DashLine),
            )
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        cur_pos = self.mapToScene(event.pos())
        if self.drawing and self.temp_rect:
            self.temp_rect.setRect(QRectF(self.start_pos, cur_pos).normalized())
        elif self.selecting and self.select_rect:
            self.select_rect.setRect(QRectF(self.start_pos, cur_pos).normalized())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        end_pos = self.mapToScene(event.pos())
        x = int(min(self.start_pos.x(), end_pos.x()))
        y = int(min(self.start_pos.y(), end_pos.y()))
        w = int(abs(self.start_pos.x() - end_pos.x()))
        h = int(abs(self.start_pos.y() - end_pos.y()))

        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False
            if self.temp_rect:
                self.scene.removeItem(self.temp_rect)
            if w > 5 and h > 5:
                # 스티치된(이어붙여진) 전체 좌표를 전달
                self.parent_window.add_pending_box_from_stitched(x, y, w, h)
            else:
                shift_pressed = bool(
                    event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                )
                self.parent_window.handle_selection_from_stitched(
                    x, y, w, h, shift_pressed
                )

        elif event.button() == Qt.MouseButton.RightButton and self.selecting:
            self.selecting = False
            if self.select_rect:
                self.scene.removeItem(self.select_rect)
            shift_pressed = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.parent_window.handle_selection_from_stitched(x, y, w, h, shift_pressed)
        else:
            super().mouseReleaseEvent(event)


class _BatchApplyDialog(QDialog):
    def __init__(self, parent, src_name: str, candidates: list[tuple[int, str]]):
        super().__init__(parent)
        self.selected_indices: list[int] = []
        self.setWindowTitle("일괄 적용 대상 선택")
        self.setMinimumWidth(350)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"현재 그룹: {src_name}"))
        layout.addWidget(QLabel("번호 개수가 같은 그룹 (일괄 적용 대상):"))

        self.checkboxes: list[tuple[QCheckBox, int]] = []
        for idx, name in candidates:
            cb = QCheckBox(name)
            cb.setChecked(True)
            layout.addWidget(cb)
            self.checkboxes.append((cb, idx))

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

    def _on_accept(self):
        self.selected_indices = [idx for cb, idx in self.checkboxes if cb.isChecked()]
        if not self.selected_indices:
            QMessageBox.warning(self, "알림", "선택된 그룹이 없습니다.")
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
        self.working_show_average = [f.show_average for f in fields]
        self.row_index_order = []

        self.setWindowTitle("값 매핑")
        self.setMinimumSize(480, 460)

        layout = QVBoxLayout(self)

        group_layout = QHBoxLayout()
        group_layout.addWidget(QLabel("그룹 선택"))
        self.group_combo = QComboBox()
        self.group_combo.addItems([f.name for f in fields])
        group_layout.addWidget(self.group_combo)
        layout.addLayout(group_layout)

        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("그룹명"))
        self.group_name_edit = QLineEdit()
        name_layout.addWidget(self.group_name_edit)
        layout.addLayout(name_layout)

        reverse_text = "ON" if reverse_numbering else "OFF"
        self.reverse_label = QLabel(f"현재 번호 역순: {reverse_text}")
        layout.addWidget(self.reverse_label)

        check_layout = QHBoxLayout()
        self.duplicate_check = QCheckBox("중복 허용 (다중 선택 가능)")
        check_layout.addWidget(self.duplicate_check)
        self.average_check = QCheckBox("평균 보기")
        check_layout.addWidget(self.average_check)
        layout.addLayout(check_layout)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["번호", "값"])
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

        if fields:
            self._load_group(0)

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
        if self.reverse_numbering:
            self.row_index_order = list(reversed(self.row_index_order))

        self.table.blockSignals(True)
        for row_idx, map_idx in enumerate(self.row_index_order):
            num_item = QTableWidgetItem(str(map_idx + 1))
            num_item.setFlags(num_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row_idx, 0, num_item)

            value = values[map_idx] if map_idx < len(values) else ""
            self.table.setItem(row_idx, 1, QTableWidgetItem(value))
        self.table.blockSignals(False)

        # 값이 하나도 없으면 기본적으로 평균 보기 체크
        all_empty = all(
            not (self.table.item(r, 1) and self.table.item(r, 1).text().strip())
            for r in range(box_count)
        )
        if all_empty:
            avg = True
        self.average_check.setChecked(bool(avg))

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
            if len(f.boxes) == src_box_count:
                candidates.append((i, f.name))

        if not candidates:
            QMessageBox.information(self, "알림", "번호 개수가 같은 그룹이 없습니다.")
            return

        dialog = _BatchApplyDialog(self, self.fields[src_idx].name, candidates)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            for idx in dialog.selected_indices:
                self.working_maps[idx] = list(src_values)
                if src_idx < len(self.working_allow_duplicates):
                    self.working_allow_duplicates[idx] = self.working_allow_duplicates[
                        src_idx
                    ]
                if src_idx < len(self.working_show_average):
                    self.working_show_average[idx] = self.working_show_average[src_idx]
            self._load_group(self.current_group_index)

    def accept(self):
        self._save_current_group()
        for idx, field in enumerate(self.fields):
            box_count = len(field.boxes)
            values = self.working_maps[idx] if idx < len(self.working_maps) else []
            if len(values) < box_count:
                values = values + [""] * (box_count - len(values))
            elif len(values) > box_count:
                values = values[:box_count]
            field.value_map = values

            name = (
                self.working_names[idx] if idx < len(self.working_names) else field.name
            )
            name = name.strip() if isinstance(name, str) else field.name
            if name:
                field.name = name

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
        super().accept()


class MainWindow(QMainWindow):
    ROT_CYCLE = ROTATION_CODES

    def __init__(self):
        super().__init__()
        self.setWindowTitle("설문지 자동 분석기")
        self.resize(1600, 900)

        self.preset = TemplatePreset()
        self.pages = []
        self.file_paths = []

        self.preset_dir = Path(os.getenv("LOCALAPPDATA")) / "CheckFinder" / "presets"
        self.preset_dir.mkdir(parents=True, exist_ok=True)
        self.current_preset_name = None

        self.is_a_view = False  # False = B안(1열 세로연결), True = A안(2열 세로연결)
        self.rot_idx = 0
        self.page_H = 0
        self.page_W = 0

        self.pending_boxes = []
        self.selected_boxes = []

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

        self.reverse_number_action = page_menu.addAction("번호 역순: OFF")
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

        settings_menu = file_menu.addMenu("설정")
        dup_action = settings_menu.addAction("중복 허용")
        dup_action.triggered.connect(self.open_value_mapping)

        cache_action = file_menu.addAction("캐시 삭제")
        cache_action.triggered.connect(self.clear_cache)

        self._sync_rotation_actions()
        self._sync_reverse_numbering_state()
        self._sync_view_toggle_text()

    def _init_ui(self):
        self._init_menu()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # 상단 툴바 버튼
        btn_layout = QHBoxLayout()

        group_btn = QPushButton("선택 묶기 (그룹화)")
        group_btn.setStyleSheet("background-color: #2196F3; color: white;")
        group_btn.clicked.connect(self.group_boxes)

        self.value_map_btn = QPushButton("값 매핑")
        self.value_map_btn.clicked.connect(self.open_value_mapping)

        self.comment_field_btn = QPushButton("의견 칸으로 지정")
        self.comment_field_btn.clicked.connect(self.assign_comment_field)

        del_btn = QPushButton("선택 박스 삭제")
        del_btn.setStyleSheet("background-color: #f44336; color: white;")
        del_btn.clicked.connect(self.delete_selected_boxes)

        exec_btn = QPushButton("▶ 분석 실행")
        exec_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold;"
        )
        exec_btn.clicked.connect(self.execute_analysis)

        if hasattr(self, "file_menu_btn"):
            self.file_menu_btn.setFixedHeight(group_btn.sizeHint().height())
            btn_layout.addWidget(self.file_menu_btn)

        # 미세 회전 각도 조절
        btn_layout.addWidget(QLabel("미세 회전:"))
        self.fine_angle_spin = QDoubleSpinBox()
        self.fine_angle_spin.setRange(-10.0, 10.0)
        self.fine_angle_spin.setSingleStep(0.5)
        self.fine_angle_spin.setDecimals(1)
        self.fine_angle_spin.setValue(0.0)
        self.fine_angle_spin.setSuffix("°")
        self.fine_angle_spin.setFixedWidth(80)
        self.fine_angle_spin.valueChanged.connect(self.change_fine_angle)
        btn_layout.addWidget(self.fine_angle_spin)

        btn_layout.addWidget(group_btn)
        btn_layout.addWidget(self.value_map_btn)
        btn_layout.addWidget(self.comment_field_btn)
        btn_layout.addWidget(del_btn)
        btn_layout.addStretch(1)
        btn_layout.addWidget(exec_btn)

        main_layout.addLayout(btn_layout)

        # 단일 거대 캔버스 배치
        self.canvas = MainCanvas(self)
        main_layout.addWidget(self.canvas)

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
        self.preset.fields.clear()
        self.is_a_view = False

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
        self.preset.reverse_numbering = bool(checked)
        self._sync_reverse_numbering_state()

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
            QMessageBox.information(self, "알림", "값을 매핑할 그룹이 없습니다.")
            return

        dialog = ValueMappingDialog(
            self, self.preset.fields, self.preset.reverse_numbering
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.update_canvas()

    def assign_comment_field(self):
        if not self.preset.fields:
            QMessageBox.information(self, "알림", "의견 칸으로 지정할 그룹이 없습니다.")
            return

        names = [f.name for f in self.preset.fields]
        name, ok = QInputDialog.getItem(
            self, "의견 칸 지정", "그룹 선택:", names, 0, False
        )
        if not ok or not name:
            return

        target = next((f for f in self.preset.fields if f.name == name), None)
        if not target:
            return

        if target.is_comment:
            reply = QMessageBox.question(
                self,
                "의견 칸 해제",
                "이미 의견 칸입니다. 해제할까요?",
            )
            if reply == QMessageBox.StandardButton.Yes:
                target.is_comment = False
                QMessageBox.information(self, "완료", "의견 칸 지정이 해제되었습니다.")
        else:
            target.is_comment = True
            QMessageBox.information(self, "완료", "의견 칸으로 지정되었습니다.")

    def _sanitize_config_name(self, name: str) -> str:
        invalid_chars = '<>:"/\\|?*'
        cleaned = "".join("_" if ch in invalid_chars else ch for ch in name).strip()
        return cleaned

    def _serialize_config(self) -> dict:
        return {
            "page_count": self.preset.page_count,
            "fine_angle": self.preset.fine_angle,
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

    def _update_page_size(self):
        if not self.pages:
            self.page_H = 0
            self.page_W = 0
            return
        sample = apply_rotation(
            self.pages[0], self.preset.rot_code, self.preset.fine_angle
        )
        self.page_H, self.page_W = sample.shape[:2]

        # 가로 길이(W)가 세로 길이(H)보다 크면(가로 모드) 2페이지 보기를 강제로 끕니다.
        if self.page_W > self.page_H:
            self.is_a_view = False

        self._sync_view_toggle_text()

    def _apply_loaded_preset(self, data: dict, preset_name: str = ""):
        self.preset = TemplatePreset(
            page_count=int(data.get("page_count", 1)),
            fine_angle=float(data.get("fine_angle", 0.0)),
            rot_code=int(data.get("rot_code", -1)),
            reverse_numbering=bool(data.get("reverse_numbering", False)),
            template_dilate_pct=float(data.get("template_dilate_pct", 0.3)),
            fields=[],
        )
        self.preset.fields = [Field.from_dict(f) for f in data.get("fields", [])]
        self.pending_boxes = [Box.from_dict(b) for b in data.get("pending_boxes", [])]
        self.is_a_view = bool(data.get("is_a_view", False))
        self.selected_boxes.clear()
        self._sync_rotation_index()
        self._sync_fine_angle_spin()
        self._sync_reverse_numbering_state()

        saved_templates = self._load_template_images(preset_name) if preset_name else []

        if self.file_paths:
            raw_pages = load_pdf_pages(self.file_paths[0])[: self.preset.page_count]
            if len(raw_pages) < self.preset.page_count:
                self.preset.page_count = len(raw_pages)

            if saved_templates and len(saved_templates) >= self.preset.page_count:
                # 저장된 템플릿 기준으로 새 PDF 페이지 정렬
                aligned = []
                for i in range(self.preset.page_count):
                    aligner = ImageAligner(saved_templates[i])
                    img = apply_rotation(
                        raw_pages[i], self.preset.rot_code, self.preset.fine_angle
                    )
                    aligned.append(aligner.align(img))
                self.pages = aligned
            else:
                self.pages = raw_pages

            self._filter_boxes_outside_page_count()
            self._update_page_size()
        elif saved_templates:
            # PDF 없이 프리셋만 로드한 경우: 저장된 템플릿을 표시
            self.pages = saved_templates[: self.preset.page_count]
            self._update_page_size()

        self.update_canvas()
        self._sync_view_toggle_text()

    def save_preset(self):
        if self.current_preset_name:
            self._save_preset_to_name(self.current_preset_name)
        else:
            self.save_preset_as()

    def save_preset_as(self):
        name, ok = QInputDialog.getText(
            self, "프리셋 저장", "프리셋 이름을 입력하세요:"
        )
        if not ok:
            return
        name = self._sanitize_config_name(name)
        if not name:
            QMessageBox.warning(self, "경고", "유효한 프리셋 이름을 입력하세요.")
            return
        self._save_preset_to_name(name)

    def _save_template_images(self, name: str):
        if not self.pages:
            return
        for i, page in enumerate(self.pages):
            img = apply_rotation(page, self.preset.rot_code, self.preset.fine_angle)
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

    def _save_preset_to_name(self, name: str):
        data = self._serialize_config()
        path = self.preset_dir / f"{name}.json"
        try:
            with open(path, "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"프리셋 저장 실패: {exc}")
            return
        self._save_template_images(name)
        self.current_preset_name = name

    def _list_config_names(self) -> list[str]:
        return sorted(path.stem for path in self.preset_dir.glob("*.json"))

    def _load_preset_by_name(self, name: str):
        path = self.preset_dir / f"{name}.json"
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"프리셋 불러오기 실패: {exc}")
            return
        self.current_preset_name = name
        self._apply_loaded_preset(data, preset_name=name)

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

    def load_pdf(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "PDF 선택", "", "PDF Files (*.pdf)"
        )
        if not paths:
            return

        self.file_paths = paths
        page_count, ok = QInputDialog.getInt(
            self, "템플릿 설정", "설문지 1부당 페이지 수를 입력하세요:", 1, 1, 10
        )
        if not ok:
            return

        progress = self._show_progress_dialog("PDF 로드", "PDF 로딩 중...")
        progress_cb = self._make_progress_cb(progress)

        self.preset.page_count = page_count

        # 첫 PDF로 기본 페이지 로드 (초기 표시용)
        self.pages = load_pdf_pages(
            self.file_paths[0],
            progress_cb=self._wrap_progress(0, 40, "PDF 로딩 중...", progress_cb),
        )[:page_count]

        self._reset_state_for_new_pdf()
        self._update_page_size()
        self.update_canvas()
        self._sync_view_toggle_text()

        # 다중 PDF 템플릿 생성 (더 정확한 템플릿)
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
            )
            if multi_templates:
                # 병합된 템플릿으로 self.pages 교체
                new_pages = []
                for i in range(page_count):
                    if i in multi_templates:
                        new_pages.append(multi_templates[i])
                    elif i < len(self.pages):
                        new_pages.append(self.pages[i])
                if new_pages:
                    self.pages = new_pages
                    self._update_page_size()

        self.auto_detect(
            progress_cb=self._wrap_progress(70, 30, "체크박스 탐지 중...", progress_cb),
            prebuilt_templates=multi_templates,
        )
        progress_cb(100, "PDF 로드 완료")
        progress.close()
        QMessageBox.information(self, "완료", "PDF가 성공적으로 로드되었습니다.")

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

        clear_all_cache()

        if self.pages and self.file_paths:
            self.auto_detect()
        else:
            QMessageBox.information(self, "완료", "캐시가 삭제되었습니다.")

    def change_fine_angle(self, angle: float):
        """미세 회전 각도 조절 시 동작합니다."""
        self.preset.fine_angle = angle
        if not self.pages:
            return
        self._update_page_size()
        self.auto_detect()

    def change_rotation(self, index: int):
        """메뉴에서 회전 방향을 선택하면 동작합니다."""
        # 중복 실행 방지
        if self.rot_idx == index:
            return

        self.rot_idx = index

        # 변경된 콤보박스 항목 순서에 맞게 OpenCV 회전 코드 매핑
        # 0: 원본 0°, 1: 좌측 90°, 2: 우측 90°, 3: 180°
        self.preset.rot_code = ROTATION_MAP.get(index, -1)
        self._sync_rotation_actions()

        if not self.pages:
            return

        self._update_page_size()
        self.auto_detect()

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

    def update_canvas(self):
        if not self.pages:
            return

        drawn_pages = []
        for i, page in enumerate(self.pages):
            img = apply_rotation(page, self.preset.rot_code, self.preset.fine_angle)
            canvas_img = img.copy()

            if canvas_img.shape[:2] != (self.page_H, self.page_W):
                canvas_img = cv2.resize(canvas_img, (self.page_W, self.page_H))

            text_entries = []
            for field in self.preset.fields:
                for b in field.boxes:
                    if b.page_idx == i:
                        color = (
                            (0, 255, 255) if b in self.selected_boxes else (0, 200, 0)
                        )
                        thick = 4 if b in self.selected_boxes else 2
                        cv2.rectangle(
                            canvas_img, (b.x, b.y), (b.x + b.w, b.y + b.h), color, thick
                        )
                        text_entries.append((field.name, b.x, max(0, b.y - 20), color))

            if text_entries:
                canvas_img = self._draw_texts(canvas_img, text_entries)

            for b in self.pending_boxes:
                if b.page_idx == i:
                    color = (0, 255, 255) if b in self.selected_boxes else (255, 100, 0)
                    thick = 4 if b in self.selected_boxes else 2
                    cv2.rectangle(
                        canvas_img, (b.x, b.y), (b.x + b.w, b.y + b.h), color, thick
                    )

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

    def add_pending_box_from_stitched(self, st_x, st_y, w, h):
        if self.is_a_view:
            col = st_x // self.page_W
            row = st_y // self.page_H
            page_idx = row * 2 + col
            local_x = st_x % self.page_W
            local_y = st_y % self.page_H
        else:
            page_idx = st_y // self.page_H
            local_x = st_x
            local_y = st_y % self.page_H

        if page_idx >= self.preset.page_count:
            return

        clamped_w = min(w, self.page_W - local_x)
        clamped_h = min(h, self.page_H - local_y)

        self.pending_boxes.append(Box(page_idx, local_x, local_y, clamped_w, clamped_h))
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

    def handle_selection_from_stitched(self, x, y, w, h, shift_pressed):
        sel_rect = QRectF(x, y, w, h)
        all_boxes = self._all_boxes()

        if w < 5 and h < 5:
            clicked = None
            for b in all_boxes:
                b_rect = self.get_stitched_rect(b)
                if b_rect.contains(x, y):
                    if clicked is None or (b.w * b.h < clicked.w * clicked.h):
                        clicked = b

            if clicked:
                if clicked in self.selected_boxes:
                    self.selected_boxes.remove(clicked)
                else:
                    if not shift_pressed:
                        self.selected_boxes.clear()
                    self.selected_boxes.append(clicked)
            else:
                if not shift_pressed:
                    self.selected_boxes.clear()
        else:
            if not shift_pressed:
                self.selected_boxes.clear()
            for b in all_boxes:
                b_rect = self.get_stitched_rect(b)
                if sel_rect.intersects(b_rect):
                    if b not in self.selected_boxes:
                        self.selected_boxes.append(b)

        self.update_canvas()

    def delete_selected_boxes(self):
        if not self.selected_boxes:
            return

        for b in self.selected_boxes:
            if b in self.pending_boxes:
                self.pending_boxes.remove(b)
            else:
                for field in self.preset.fields:
                    if b in field.boxes:
                        field.boxes.remove(b)
                        if not field.boxes:
                            self.preset.fields.remove(field)

        self.selected_boxes.clear()
        self.update_canvas()

    def group_boxes(self):
        if not self.selected_boxes:
            QMessageBox.warning(
                self,
                "알림",
                "선택된 박스가 없습니다.\n우클릭 드래그로 묶을 박스를 선택해주세요.",
            )
            return

        name, ok = QInputDialog.getText(
            self,
            "항목 그룹화",
            "문항 이름을 입력하세요",
        )
        if ok and name:
            new_field = Field(name=name, boxes=[])
            for b in self.selected_boxes:
                if b in self.pending_boxes:
                    self.pending_boxes.remove(b)
                    new_field.boxes.append(b)
                else:
                    for field in self.preset.fields:
                        if b in field.boxes:
                            field.boxes.remove(b)
                            new_field.boxes.append(b)

            self.preset.fields = [f for f in self.preset.fields if f.boxes]
            self.preset.fields.append(new_field)
            self.selected_boxes.clear()
            self.update_canvas()

    def auto_detect(self, progress_cb=None, prebuilt_templates=None):
        """
        체크박스를 자동으로 탐지하고, 수평으로 같은 라인에 있는 항목을
        Q1, Q2, Q3 등의 그룹(Field)으로 자동 할당합니다.

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

        # ── 캐시 확인 ──
        cached = load_checkbox_cache(
            self.file_paths,
            self.preset.page_count,
            self.preset.rot_code,
            self.preset.fine_angle,
        )
        if cached is not None:
            report(0, "캐시된 체크박스 불러오는 중...")
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
            self.update_canvas()
            return

        question_number = 1

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
                )
            else:
                templates = generate_ui_templates(
                    self.file_paths[0],
                    self.preset.page_count,
                    self.preset.rot_code,
                    self.preset.fine_angle,
                    progress_cb=template_progress,
                )

        total_pages = len(self.pages)
        report(70, "체크박스 탐지 중...")

        detected_cache: dict[int, list[tuple[int, int, int, int]]] = {}

        for i, page in enumerate(self.pages):
            if templates and i in templates:
                img = templates[i]
            else:
                img = apply_rotation(page, self.preset.rot_code, self.preset.fine_angle)

            detected = auto_detect_checkboxes(img)
            detected_cache[i] = detected

            # Box 객체로 변환
            boxes = [Box(i, b[0], b[1], b[2], b[3]) for b in detected]

            rows = self._group_boxes_by_row(boxes)

            # 2. 각 줄에 대해 좌측부터 우측 방향으로 정렬(x 좌표 기준) 후 Field(그룹)로 할당
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
        )

        report(100, "체크박스 탐지 완료")
        self.update_canvas()

    def execute_analysis(self):
        if not self.file_paths or not self.preset.fields:
            QMessageBox.warning(self, "경고", "파일이나 생성된 템플릿 항목이 없습니다.")
            return

        progress = self._show_progress_dialog("분석", "분석 준비 중...")
        progress_cb = self._make_progress_cb(progress)

        success = run_analysis(
            self.file_paths, self.pages, self.preset, progress_cb=progress_cb
        )
        progress_cb(100, "분석 완료")
        progress.close()

        if success:
            QMessageBox.information(self, "완료", "분석이 완료되었습니다.")
        else:
            QMessageBox.critical(self, "오류", "분석 중 문제가 발생했습니다.")
