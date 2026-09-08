import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from src.models import Box, Field
from src.processor import _label_number
from src.ui import MainWindow, ValueMappingDialog


_APP = QApplication.instance() or QApplication([])


def _boxes(count):
    return [Box(0, index * 20, 0, 12, 12) for index in range(count)]


def _displayed_names(dialog):
    return [dialog.table.item(row, 1).text() for row in range(dialog.table.rowCount())]


def _physical_names(field, reverse):
    """Names a scanner sees at physical left-to-right box indexes."""
    return [
        field.value_map[_label_number(len(field.boxes), index, reverse) - 1]
        for index in range(1, len(field.boxes) + 1)
    ]


class ReverseNumberingTests(unittest.TestCase):
    def setUp(self):
        self.local_app_data = tempfile.TemporaryDirectory()
        self.previous_local_app_data = os.environ.get("LOCALAPPDATA")
        os.environ["LOCALAPPDATA"] = self.local_app_data.name
        self.window = MainWindow()
        self.window._preset_dirty = False

    def tearDown(self):
        self.window._preset_dirty = False
        self.window.close()
        self.window.deleteLater()
        _APP.processEvents()
        if self.previous_local_app_data is None:
            os.environ.pop("LOCALAPPDATA", None)
        else:
            os.environ["LOCALAPPDATA"] = self.previous_local_app_data
        self.local_app_data.cleanup()

    def test_toggle_preserves_physical_names_and_mapping_dialog_row_order(self):
        field = Field("Q1", _boxes(3), ["one", "two", "three"])
        self.window.preset.fields = [field]
        self.window.preset.reverse_numbering = True
        before_physical = _physical_names(field, True)
        before_dialog = ValueMappingDialog(self.window, [field], True)
        before_rows = _displayed_names(before_dialog)

        self.window.toggle_reverse_numbering(False)

        after_dialog = ValueMappingDialog(self.window, [field], False)
        self.assertEqual(field.value_map, ["three", "two", "one"])
        self.assertEqual(_physical_names(field, False), before_physical)
        self.assertEqual(_displayed_names(after_dialog), before_rows)
        before_dialog.close()
        after_dialog.close()

    def test_partial_maps_are_padded_reversed_and_reversible(self):
        partial = Field("Q1", _boxes(3), ["left", "middle"])
        long_map = Field("Q2", _boxes(2), ["a", "b", "suffix"])
        comment = Field("memo", _boxes(2), ["keep", "this"], is_comment=True)
        average = Field("average", _boxes(2), [])
        self.window.preset.fields = [partial, long_map, comment, average]
        self.window.preset.reverse_numbering = True

        self.window.toggle_reverse_numbering(False)
        self.assertEqual(partial.value_map, ["", "middle", "left"])
        self.assertEqual(long_map.value_map, ["b", "a", "suffix"])
        self.assertEqual(comment.value_map, ["keep", "this"])
        self.assertEqual(average.value_map, [])

        self.window.toggle_reverse_numbering(True)
        self.assertEqual(partial.value_map, ["left", "middle", ""])
        self.assertEqual(long_map.value_map, ["a", "b", "suffix"])
        self.assertEqual(comment.value_map, ["keep", "this"])

    def test_same_state_is_a_noop_and_history_snapshots_rebase(self):
        field = Field("Q1", _boxes(2), ["one", "two"])
        self.window.preset.fields = [field]
        self.window.preset.reverse_numbering = True
        self.window._preset_dirty = False
        old_snapshot = ([Field("Q1", _boxes(2), ["old-1", "old-2"])], [])
        redo_snapshot = ([Field("Q1", _boxes(2), ["redo-1", "redo-2"])], [])
        self.window._undo_history = [(old_snapshot, "old")]
        self.window._redo_history = [(redo_snapshot, "redo")]

        self.window.toggle_reverse_numbering(True)
        self.assertEqual(field.value_map, ["one", "two"])
        self.assertFalse(self.window._preset_dirty)

        self.window.toggle_reverse_numbering(False)
        self.assertEqual(field.value_map, ["two", "one"])
        self.assertEqual(self.window._undo_history[-1][0][0][0].value_map, ["old-2", "old-1"])
        self.assertEqual(self.window._redo_history[-1][0][0][0].value_map, ["redo-2", "redo-1"])
        self.window.undo_edit()
        self.assertEqual(self.window.preset.fields[0].value_map, ["old-2", "old-1"])
        self.window.redo_edit()
        self.assertEqual(self.window.preset.fields[0].value_map, ["two", "one"])

    def test_field_serialization_preserves_toggled_map_and_processor_labels(self):
        field = Field("Q1", _boxes(3), ["north", "center", "south"])
        self.window.preset.fields = [field]
        self.window.preset.reverse_numbering = True
        before = _physical_names(field, True)
        self.window.toggle_reverse_numbering(False)

        restored = Field.from_dict(json.loads(json.dumps(field.to_dict())))
        self.assertEqual(_physical_names(restored, False), before)
        self.assertEqual([_label_number(3, index, False) for index in (1, 2, 3)], [1, 2, 3])
        self.assertEqual([_label_number(3, index, True) for index in (1, 2, 3)], [3, 2, 1])


if __name__ == "__main__":
    unittest.main()
