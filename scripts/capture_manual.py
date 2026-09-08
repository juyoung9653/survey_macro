"""Capture actual Qt windows using fictional data, without reading user PDFs."""
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter
from PyQt6.QtWidgets import QApplication
from src.localization import install_korean_translations
from src.models import Box, Field, TemplatePreset
from src.ui import MainWindow, ValueMappingDialog


def example_page():
    image = QImage(1000, 640, QImage.Format.Format_RGB888)
    image.fill(QColor('white'))
    painter = QPainter(image)
    painter.setPen(QColor('#172b4d'))
    painter.setFont(QFont('Malgun Gothic', 24, QFont.Weight.Bold))
    painter.drawText(60, 65, '프로그램 이용 만족도 설문')
    painter.setFont(QFont('Malgun Gothic', 13))
    painter.drawText(60, 100, '사용 안내용 가상 설문 · 실제 응답 자료가 아닙니다')
    painter.setPen(QColor('#444444'))
    painter.drawLine(60, 122, 935, 122)
    painter.drawText(60, 174, '1. 프로그램 이용에 얼마나 만족하시나요? (한 가지만 선택)')
    boxes = []
    for x, label in zip((80, 320, 560), ('만족', '보통', '불만족')):
        painter.drawRect(x, 205, 28, 28)
        painter.drawText(x + 45, 229, label)
        boxes.append(Box(page_idx=0, x=x, y=205, w=28, h=28))
    painter.drawText(60, 330, '2. 좋았던 점이나 바라는 점을 적어주세요.')
    painter.drawRect(80, 365, 800, 160)
    painter.end()
    data = image.bits()
    data.setsize(image.sizeInBytes())
    page = np.frombuffer(data, np.uint8).reshape(640, image.bytesPerLine())[:, :3000].reshape(640, 1000, 3)
    return page[:, :, ::-1].copy(), boxes


def main():
    output = ROOT / '설명서' / 'images'
    output.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    font_path = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / 'malgun.ttf'
    if QFontDatabase.addApplicationFont(str(font_path)) < 0:
        raise RuntimeError('Korean font unavailable for screenshots')
    app.setFont(QFont('Malgun Gothic', 10))
    install_korean_translations(app)
    with tempfile.TemporaryDirectory() as local:
        old_local = os.environ.get('LOCALAPPDATA')
        os.environ['LOCALAPPDATA'] = local
        try:
            window = MainWindow()
            window.resize(1480, 840)
            window.show()
            page, boxes = example_page()
            window.file_paths = ['안내용_만족도설문.pdf']
            window.pages = [page]
            window.preset = TemplatePreset(page_count=1, reverse_numbering=False)
            window._update_page_size()
            # Exercise the application's real detection/grouping path. Only
            # cache I/O is isolated: the fictional PDF does not exist on disk.
            with patch('src.ui.load_checkbox_cache', return_value=None), patch('src.ui.save_checkbox_cache'):
                window.auto_detect(prebuilt_templates={0: page}, mark_dirty=False)
            if len(window.preset.fields) != 1 or len(window.preset.fields[0].boxes) != 3:
                raise RuntimeError('Example must auto-detect three choices grouped as Q1')
            boxes = window.preset.fields[0].boxes
            def capture(name):
                window.update_canvas()
                window._refresh_document_status()
                app.processEvents()
                window.canvas.fitInView(window.canvas.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
                app.processEvents()
                if not window.grab().save(str(output/name)):
                    raise RuntimeError('Screenshot save failed')
            capture('01-pdf.png')
            capture('02-select.png')
            window.selected_boxes = []
            window.pending_boxes = []
            average_field = Field(name='이용 만족도', boxes=boxes, show_average=True)
            average_dialog = ValueMappingDialog(window, [average_field], True)
            average_dialog.resize(620, 510)
            average_dialog.show()
            app.processEvents()
            if not average_dialog.average_check.isChecked() or any(
                average_dialog.table.item(row, 1).text()
                for row in range(average_dialog.table.rowCount())
            ):
                raise RuntimeError('Average example must keep choice names empty')
            if not average_dialog.grab().save(str(output/'03-average.png')):
                raise RuntimeError('Screenshot save failed')
            average_dialog.reject()
            window.preset.fields = [Field(name='이용 만족도', boxes=boxes, value_map=['만족','보통','불만족'])]
            mapping = ValueMappingDialog(window, window.preset.fields, False)
            mapping.resize(620, 510)
            mapping.show()
            app.processEvents()
            if not mapping.grab().save(str(output/'03-options.png')):
                raise RuntimeError('Screenshot save failed')
            mapping.reject()
            window.current_preset_name = '만족도설문'
            capture('04-analyze.png')
            window._preset_dirty = False
            window.close()
        finally:
            if old_local is None:
                os.environ.pop('LOCALAPPDATA', None)
            else:
                os.environ['LOCALAPPDATA'] = old_local
    print('Captured 5 actual UI screenshots with fictional sample data.')


if __name__ == '__main__':
    main()
