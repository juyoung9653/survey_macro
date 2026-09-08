import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from src.models import Box, Field
from src.ui import MainWindow


_APP = QApplication.instance() or QApplication([])


def _boxes(start, count):
    return [Box(0, start + index * 20, 0, 10, 10) for index in range(count)]


class FieldNumberingEditingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_local_app_data = os.environ.get("LOCALAPPDATA")
        os.environ["LOCALAPPDATA"] = self.temp.name
        self.window = MainWindow()
        self.window._preset_dirty = False

    def tearDown(self):
        self.window._preset_dirty = False
        self.window.close()
        self.window.deleteLater()
        _APP.processEvents()
        if self.old_local_app_data is None:
            os.environ.pop("LOCALAPPDATA", None)
        else:
            os.environ["LOCALAPPDATA"] = self.old_local_app_data
        self.temp.cleanup()

    def test_removing_box_preserves_remaining_physical_names_for_override(self):
        boxes = _boxes(0, 3)
        field = Field("forward", boxes, ["left", "middle", "right"], reverse_numbering=False)

        MainWindow._remove_box_ids_from_field(field, {id(boxes[1])}, True)

        self.assertEqual(field.boxes, [boxes[0], boxes[2]])
        self.assertEqual(field.value_map, ["left", "right"])

        reverse_boxes = _boxes(100, 3)
        reverse_field = Field("reverse", reverse_boxes, ["right", "middle", "left"], reverse_numbering=True)
        MainWindow._remove_box_ids_from_field(reverse_field, {id(reverse_boxes[1])}, False)
        self.assertEqual(reverse_field.boxes, [reverse_boxes[0], reverse_boxes[2]])
        self.assertEqual(reverse_field.value_map, ["right", "left"])

    def test_regroup_mixed_directions_preserves_names_and_makes_explicit_field(self):
        forward_boxes = _boxes(0, 2)
        reverse_boxes = _boxes(100, 2)
        forward = Field("forward", forward_boxes, ["A", "B"], reverse_numbering=False)
        reverse = Field("reverse", reverse_boxes, ["D", "C"], reverse_numbering=True)
        self.window.preset.reverse_numbering = True
        self.window.preset.fields = [forward, reverse]
        self.window.selected_boxes = [forward_boxes[1], reverse_boxes[0]]

        with patch("src.ui.QInputDialog.getText", return_value=("combined", True)):
            self.window.group_boxes()

        combined = next(field for field in self.window.preset.fields if field.name == "combined")
        self.assertEqual(combined.boxes, [forward_boxes[1], reverse_boxes[0]])
        self.assertTrue(combined.reverse_numbering)
        self.assertEqual(combined.value_map, ["C", "B"])
        self.assertEqual(forward.value_map, ["A"])
        self.assertEqual(reverse.value_map, ["D"])

    def test_comment_map_is_not_reordered_by_box_removal(self):
        boxes = _boxes(0, 2)
        comment = Field("memo", boxes, ["keep", "suffix"], is_comment=True, reverse_numbering=True)

        MainWindow._remove_box_ids_from_field(comment, {id(boxes[0])}, False)

        self.assertEqual(comment.boxes, [boxes[1]])
        self.assertEqual(comment.value_map, ["keep", "suffix"])


if __name__ == "__main__":
    unittest.main()
