import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.models import Box, Field, TemplatePreset
from src.ui import MainWindow


class _EditorHarness:
    _capture_edit_state = MainWindow._capture_edit_state
    _restore_edit_state = MainWindow._restore_edit_state
    _commit_edit = MainWindow._commit_edit
    undo_edit = MainWindow.undo_edit
    redo_edit = MainWindow.redo_edit
    _all_boxes = MainWindow._all_boxes
    _group_boxes_by_row = staticmethod(MainWindow._group_boxes_by_row)
    _boxes_in_reading_order = MainWindow._boxes_in_reading_order
    _box_is_selected = MainWindow._box_is_selected
    _box_from_stitched_rect = MainWindow._box_from_stitched_rect
    add_pending_box_from_stitched = MainWindow.add_pending_box_from_stitched
    _unique_field_name = MainWindow._unique_field_name
    add_comment_box_from_stitched = MainWindow.add_comment_box_from_stitched
    get_stitched_rect = MainWindow.get_stitched_rect
    box_at_stitched = MainWindow.box_at_stitched
    select_box = MainWindow.select_box
    _boxes_in_selection_range = MainWindow._boxes_in_selection_range
    select_box_range = MainWindow.select_box_range
    clear_box_selection = MainWindow.clear_box_selection
    bounded_move_delta = MainWindow.bounded_move_delta
    move_selected_boxes = MainWindow.move_selected_boxes
    _resized_geometry = MainWindow._resized_geometry
    resize_box = MainWindow.resize_box
    _remove_box_ids_from_field = staticmethod(
        MainWindow._remove_box_ids_from_field
    )
    delete_selected_boxes = MainWindow.delete_selected_boxes
    group_boxes = MainWindow.group_boxes

    def __init__(self, page_count=2, *, is_a_view=False):
        self.preset = TemplatePreset(page_count=page_count)
        self.pending_boxes = []
        self.selected_boxes = []
        self._selection_anchor = None
        self.page_W = 100
        self.page_H = 200
        self.is_a_view = is_a_view
        self._undo_history = []
        self._redo_history = []
        self._edit_history_limit = 50
        self.canvas_updates = 0

    def update_canvas(self):
        self.canvas_updates += 1

    def _update_history_buttons(self):
        pass


class BoxEditingTests(unittest.TestCase):
    def test_canceling_page_count_keeps_the_previous_document(self):
        window = SimpleNamespace(
            file_paths=["previous.pdf"],
            preset=TemplatePreset(page_count=2),
        )

        with (
            patch(
                "src.ui.QFileDialog.getOpenFileNames",
                return_value=(["new.pdf"], "PDF Files (*.pdf)"),
            ),
            patch("src.ui.QInputDialog.getInt", return_value=(2, False)),
        ):
            MainWindow.load_pdf(window)

        self.assertEqual(window.file_paths, ["previous.pdf"])

    def test_new_pdf_state_clears_the_active_preset_name(self):
        status_updates = []
        window = SimpleNamespace(
            pending_boxes=[Box(0, 1, 1, 10, 10)],
            selected_boxes=[Box(0, 1, 1, 10, 10)],
            preset=TemplatePreset(
                page_count=2,
                page_fine_angles=[0.2, -0.1],
                fields=[Field(name="기존")],
            ),
            current_preset_name="기존 프리셋",
            _pages_are_canonical=True,
            _analysis_reference_pages=[object()],
            is_a_view=True,
            _undo_history=[],
            _redo_history=[],
            _refresh_document_status=lambda: status_updates.append(True),
        )

        MainWindow._reset_state_for_new_pdf(window)

        self.assertIsNone(window.current_preset_name)
        self.assertEqual(window.preset.fields, [])
        self.assertEqual(window.preset.page_fine_angles, [])
        self.assertEqual(status_updates, [True])

    def test_added_box_uses_stitched_page_coordinates_and_can_be_undone(self):
        window = _EditorHarness()

        window.add_pending_box_from_stitched(10, 215, 20, 30)

        self.assertEqual(len(window.pending_boxes), 1)
        self.assertEqual(
            (
                window.pending_boxes[0].page_idx,
                window.pending_boxes[0].x,
                window.pending_boxes[0].y,
                window.pending_boxes[0].w,
                window.pending_boxes[0].h,
            ),
            (1, 10, 15, 20, 30),
        )

        window.undo_edit()
        self.assertEqual(window.pending_boxes, [])

        window.redo_edit()
        self.assertEqual(len(window.pending_boxes), 1)
        self.assertEqual(window.pending_boxes[0].page_idx, 1)

    def test_two_page_view_maps_to_the_correct_source_page(self):
        window = _EditorHarness(page_count=4, is_a_view=True)

        box = window._box_from_stitched_rect(112, 218, 25, 30)

        self.assertIsNotNone(box)
        self.assertEqual((box.page_idx, box.x, box.y), (3, 12, 18))

    def test_free_entry_tool_creates_a_named_free_entry_field(self):
        window = _EditorHarness()
        window.preset.fields.append(Field(name="자유기입"))

        window.add_comment_box_from_stitched(5, 10, 40, 25)

        field = window.preset.fields[-1]
        self.assertEqual(field.name, "자유기입 2")
        self.assertTrue(field.is_comment)
        self.assertEqual(len(field.boxes), 1)

    def test_move_is_clamped_to_page_and_resize_keeps_minimum_size(self):
        window = _EditorHarness(page_count=1)
        box = Box(0, 10, 10, 20, 20)
        window.pending_boxes = [box]
        window.selected_boxes = [box]

        window.move_selected_boxes(-100, 500)
        self.assertEqual((box.x, box.y), (0, 180))

        window.resize_box(box, "se", -100, -100)
        self.assertEqual((box.w, box.h), (8, 8))
        self.assertEqual(len(window._undo_history), 2)

    def test_ctrl_click_toggles_individual_boxes(self):
        window = _EditorHarness(page_count=1)
        first = Box(0, 10, 10, 10, 10)
        second = Box(0, 30, 10, 10, 10)
        window.pending_boxes = [first, second]

        window.select_box(first)
        window.select_box(second, additive=True)
        self.assertEqual(window.selected_boxes, [first, second])

        window.select_box(first, additive=True)
        self.assertEqual(window.selected_boxes, [second])
        self.assertIs(window._selection_anchor, second)

    def test_shift_click_selects_horizontal_vertical_and_rectangular_ranges(self):
        window = _EditorHarness(page_count=1)
        grid = [
            [Box(0, 10 + column * 20, 10 + row * 20, 10, 10) for column in range(3)]
            for row in range(3)
        ]
        window.pending_boxes = [box for row in grid for box in row]

        window.select_box(grid[0][0])
        window.select_box_range(grid[0][2])
        self.assertEqual(set(map(id, window.selected_boxes)), set(map(id, grid[0])))

        window.select_box(grid[0][0])
        window.select_box_range(grid[2][0])
        self.assertEqual(
            set(map(id, window.selected_boxes)),
            {id(grid[row][0]) for row in range(3)},
        )

        window.select_box(grid[0][0])
        window.select_box_range(grid[2][2])
        self.assertEqual(len(window.selected_boxes), 9)

    def test_delete_keeps_value_mapping_aligned_and_undo_restores_it(self):
        window = _EditorHarness(page_count=1)
        boxes = [
            Box(0, 10, 10, 10, 10),
            Box(0, 30, 10, 10, 10),
            Box(0, 50, 10, 10, 10),
        ]
        field = Field(name="만족도", boxes=boxes, value_map=["좋음", "보통", "나쁨"])
        window.preset.fields = [field]
        window.selected_boxes = [boxes[1]]

        window.delete_selected_boxes(confirm=False)

        self.assertEqual(field.boxes, [boxes[0], boxes[2]])
        self.assertEqual(field.value_map, ["좋음", "나쁨"])

        window.undo_edit()
        restored = window.preset.fields[0]
        self.assertEqual(restored.value_map, ["좋음", "보통", "나쁨"])
        self.assertEqual(len(restored.boxes), 3)

    def test_regrouping_preserves_existing_selected_values(self):
        window = _EditorHarness(page_count=1)
        first = Box(0, 10, 10, 10, 10)
        second = Box(0, 30, 10, 10, 10)
        pending = Box(0, 50, 10, 10, 10)
        window.preset.fields = [
            Field(name="기존", boxes=[first, second], value_map=["A", "B"])
        ]
        window.pending_boxes = [pending]
        window.selected_boxes = [pending, second]

        with patch("src.ui.QInputDialog.getText", return_value=("새 문항", True)):
            window.group_boxes()

        self.assertEqual(window.preset.fields[0].boxes, [first])
        self.assertEqual(window.preset.fields[0].value_map, ["A"])
        new_field = window.preset.fields[1]
        self.assertEqual(new_field.boxes, [second, pending])
        self.assertEqual(new_field.value_map, ["B", ""])
        self.assertEqual(window.pending_boxes, [])

    def test_duplicate_question_name_is_rejected_before_regrouping(self):
        window = _EditorHarness(page_count=1)
        existing = Box(0, 10, 10, 10, 10)
        pending = Box(0, 30, 10, 10, 10)
        window.preset.fields = [Field(name="Q1", boxes=[existing])]
        window.pending_boxes = [pending]
        window.selected_boxes = [pending]

        with (
            patch("src.ui.QInputDialog.getText", return_value=("q1", True)),
            patch("src.ui.QMessageBox.warning") as warning,
        ):
            window.group_boxes()

        warning.assert_called_once()
        self.assertEqual([field.name for field in window.preset.fields], ["Q1"])
        self.assertEqual(window.pending_boxes, [pending])

    def test_committed_edit_marks_the_setup_as_modified(self):
        window = _EditorHarness(page_count=1)

        window.add_pending_box_from_stitched(10, 10, 20, 20)

        self.assertTrue(window._preset_dirty)


if __name__ == "__main__":
    unittest.main()
