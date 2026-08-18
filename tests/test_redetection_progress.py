import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, mock_open, patch

from PyQt6.QtWidgets import QMessageBox
import numpy as np

from src.models import Box, Field, TemplatePreset
from src.ui import MainWindow, ROTATION_MAP


class RedetectionProgressTests(unittest.TestCase):
    def test_reverse_numbering_defaults_to_on(self):
        self.assertTrue(TemplatePreset().reverse_numbering)

    def test_incompatible_preset_is_not_selected_and_reports_load_failure(self):
        message = "현재 PDF와 일치하지 않는 프리셋이라 불러올 수 없습니다."
        window = SimpleNamespace(
            preset_dir=Path("presets"),
            file_paths=[],
            current_preset_name=None,
            _apply_loaded_preset=Mock(side_effect=ValueError(message)),
        )

        with (
            patch("builtins.open", mock_open(read_data="{}")),
            patch("src.ui.QMessageBox.critical") as critical,
        ):
            MainWindow._load_preset_by_name(window, "different-form")

        self.assertIsNone(window.current_preset_name)
        self.assertIn(message, critical.call_args.args[2])

    def test_page_angles_are_serialized_with_preset(self):
        window = SimpleNamespace(
            preset=TemplatePreset(
                page_count=2,
                fine_angle=0.2,
                page_fine_angles=[-0.4, 0.3],
            ),
            is_a_view=False,
            pending_boxes=[],
        )

        data = MainWindow._serialize_config(window)

        self.assertEqual(data["page_fine_angles"], [-0.4, 0.3])

    def test_canonical_preset_page_is_not_rotated_for_display(self):
        page = np.full((40, 30), 255, np.uint8)
        window = SimpleNamespace(
            _pages_are_canonical=True,
            preset=TemplatePreset(fine_angle=0.5, page_fine_angles=[-0.2]),
        )

        with patch("src.ui.apply_rotation") as rotate:
            displayed = MainWindow._configured_template_page(window, page, 0)

        self.assertIs(displayed, page)
        rotate.assert_not_called()

    def test_analysis_uses_saved_reference_instead_of_aligned_display_page(self):
        display_page = np.full((40, 30), 240, np.uint8)
        saved_page = np.full((40, 30), 250, np.uint8)
        window = SimpleNamespace(
            preset=TemplatePreset(page_count=1),
            pages=[display_page],
            _pages_are_canonical=True,
            _analysis_reference_pages=[saved_page],
        )

        pages, preprocessed = MainWindow._analysis_input_pages(window)

        self.assertIs(pages[0], saved_page)
        self.assertTrue(preprocessed)

    def test_legacy_preset_load_marks_saved_template_alignment_as_canonical(self):
        raw = np.full((40, 30), 245, np.uint8)
        saved = np.full((40, 30), 250, np.uint8)
        aligned = np.full((40, 30), 255, np.uint8)
        aligner = SimpleNamespace(align=Mock(return_value=aligned))
        window = SimpleNamespace(
            file_paths=["survey.pdf"],
            selected_boxes=[],
            _wrap_progress=MainWindow._wrap_progress,
            _sync_rotation_index=Mock(),
            _sync_fine_angle_spin=Mock(),
            _sync_reverse_numbering_state=Mock(),
            _load_template_images=Mock(return_value=[saved]),
            _filter_boxes_outside_page_count=Mock(),
            _estimate_page_fine_angles=Mock(),
            _update_page_size=Mock(),
            update_canvas=Mock(),
            _sync_view_toggle_text=Mock(),
        )
        data = {
            "page_count": 1,
            "fine_angle": 0.0,
            "rot_code": -1,
            "fields": [],
            "pending_boxes": [],
        }

        with (
            patch("src.ui.load_pdf_pages", return_value=[raw]),
            patch("src.ui.generate_ui_templates", return_value={0: raw}),
            patch(
                "src.ui.remap_preset_to_detected_layout",
                return_value=SimpleNamespace(accepted=False),
            ),
            patch("src.ui.ImageAligner", return_value=aligner),
            patch("src.ui.apply_rotation", return_value=raw) as rotate,
        ):
            MainWindow._apply_loaded_preset(window, data, preset_name="legacy")

        self.assertEqual(window.preset.page_fine_angles, [])
        self.assertTrue(window.preset.reverse_numbering)
        self.assertTrue(window._pages_are_canonical)
        self.assertIs(window.pages[0], aligned)
        rotate.assert_called_once_with(raw, -1, 0.0)
        window._estimate_page_fine_angles.assert_called_once()

    def test_preset_load_uses_detected_affine_for_canonical_coordinates(self):
        raw = np.full((60, 40), 245, np.uint8)
        saved = np.full((40, 30), 250, np.uint8)
        current_template = np.full((60, 40), 255, np.uint8)
        remapped_config = TemplatePreset(
            page_count=1,
            page_fine_angles=[0.2],
            fields=[Field(name="Q", boxes=[Box(0, 12, 14, 20, 20)])],
        )
        remapped_pending = [Box(0, 5, 6, 10, 10)]
        remap_result = SimpleNamespace(
            accepted=True,
            config=remapped_config,
            auxiliary_boxes=remapped_pending,
            matched_boxes=1,
            expected_boxes=1,
            page_transforms={
                0: np.array(
                    [[40 / 30, 0.0, 0.0], [0.0, 60 / 40, 0.0]],
                    dtype=np.float64,
                )
            },
        )
        window = SimpleNamespace(
            file_paths=["first.pdf", "second.pdf"],
            pages=[raw],
            _pages_are_canonical=False,
            preset=TemplatePreset(page_count=1, page_fine_angles=[0.2]),
            selected_boxes=[],
            _wrap_progress=MainWindow._wrap_progress,
            _sync_rotation_index=Mock(),
            _sync_fine_angle_spin=Mock(),
            _sync_reverse_numbering_state=Mock(),
            _load_template_images=Mock(return_value=[saved]),
            _filter_boxes_outside_page_count=Mock(),
            _estimate_page_fine_angles=Mock(),
            _update_page_size=Mock(),
            update_canvas=Mock(),
            _sync_view_toggle_text=Mock(),
        )
        data = {
            "page_count": 1,
            "fine_angle": 0.0,
            "rot_code": -1,
            "fields": [
                {
                    "name": "Q",
                    "boxes": [
                        {"page_idx": 0, "x": 9, "y": 9, "w": 15, "h": 14}
                    ],
                }
            ],
            "pending_boxes": [
                {"page_idx": 0, "x": 3, "y": 4, "w": 9, "h": 7}
            ],
        }

        with (
            patch("src.ui.load_pdf_pages") as load_pages,
            patch(
                "src.ui.generate_ui_templates",
                return_value={0: current_template},
            ) as generate_single,
            patch("src.ui.generate_ui_templates_multi") as generate_multi,
            patch(
                "src.ui.remap_preset_to_detected_layout",
                return_value=remap_result,
            ) as remap,
            patch("src.ui.ImageAligner") as aligner,
        ):
            MainWindow._apply_loaded_preset(window, data, preset_name="detected")

        self.assertEqual(window.pages[0].shape, saved.shape)
        self.assertTrue(window._pages_are_canonical)
        self.assertIs(window._analysis_reference_pages[0], saved)
        mapped_box = window.preset.fields[0].boxes[0]
        # Detected boxes determine the page transform, while the validated
        # preset geometry stays in its original canonical coordinate system.
        self.assertEqual(
            (mapped_box.x, mapped_box.y, mapped_box.w, mapped_box.h),
            (9, 9, 15, 14),
        )
        mapped_pending = window.pending_boxes[0]
        self.assertEqual(
            (mapped_pending.x, mapped_pending.y, mapped_pending.w, mapped_pending.h),
            (3, 4, 9, 7),
        )
        remap.assert_called_once()
        self.assertEqual(generate_single.call_args.args[0], "first.pdf")
        generate_multi.assert_not_called()
        aligner.assert_not_called()
        load_pages.assert_not_called()
        window._estimate_page_fine_angles.assert_not_called()

    def test_preset_load_rejects_a_different_form_and_restores_editor_state(self):
        raw = np.full((60, 40), 245, np.uint8)
        saved = np.full((40, 30), 250, np.uint8)
        previous_preset = TemplatePreset(
            page_count=1,
            page_fine_angles=[0.2],
            fields=[Field(name="old", boxes=[Box(0, 2, 3, 8, 8)])],
        )
        previous_pending = [Box(0, 4, 5, 6, 6)]
        selected_box = previous_preset.fields[0].boxes[0]
        previous_selected = [selected_box]
        window = SimpleNamespace(
            file_paths=["survey.pdf"],
            pages=[raw],
            _pages_are_canonical=False,
            preset=previous_preset,
            pending_boxes=previous_pending,
            selected_boxes=previous_selected,
            is_a_view=True,
            _wrap_progress=MainWindow._wrap_progress,
            _sync_rotation_index=Mock(),
            _sync_fine_angle_spin=Mock(),
            _sync_reverse_numbering_state=Mock(),
            _load_template_images=Mock(return_value=[saved]),
            _filter_boxes_outside_page_count=Mock(),
            _estimate_page_fine_angles=Mock(),
            _update_page_size=Mock(),
            update_canvas=Mock(),
            _sync_view_toggle_text=Mock(),
        )
        data = {
            "page_count": 1,
            "fine_angle": 0.0,
            "page_fine_angles": [0.1],
            "rot_code": -1,
            "fields": [
                Field(name="new", boxes=[Box(0, 12, 14, 20, 20)]).to_dict()
            ],
            "pending_boxes": [Box(0, 8, 9, 10, 10).to_dict()],
        }
        remap_result = SimpleNamespace(
            accepted=False,
            compatible=False,
            matched_boxes=0,
            expected_boxes=1,
        )

        with (
            patch(
                "src.ui.generate_ui_templates",
                return_value={0: raw},
            ),
            patch(
                "src.ui.remap_preset_to_detected_layout",
                return_value=remap_result,
            ),
            patch("src.ui.ImageAligner") as aligner,
        ):
            with self.assertRaisesRegex(
                ValueError, "일치하지 않는 프리셋이라 불러올 수 없습니다"
            ):
                MainWindow._apply_loaded_preset(
                    window, data, preset_name="different-form"
                )

        self.assertIs(window.preset, previous_preset)
        self.assertEqual(window.pages, [raw])
        self.assertFalse(window._pages_are_canonical)
        self.assertEqual(window.pending_boxes, previous_pending)
        self.assertEqual(window.selected_boxes, [selected_box])
        self.assertTrue(window.is_a_view)
        aligner.assert_not_called()

    def test_page_angles_are_estimated_once_per_template_page(self):
        progress = []
        window = SimpleNamespace(
            pages=[
                np.full((80, 60), 255, np.uint8),
                np.full((80, 60), 255, np.uint8),
            ],
            preset=TemplatePreset(page_count=2, fine_angle=0.2),
        )

        with (
            patch("src.ui.apply_rotation", side_effect=lambda image, *_args: image),
            patch("src.ui.estimate_deskew_angle", side_effect=[-0.4, 0.3]) as estimate,
        ):
            angles = MainWindow._estimate_page_fine_angles(
                window,
                progress_cb=lambda value, message: progress.append(
                    (value, message)
                ),
            )

        self.assertEqual(angles, [-0.4, 0.3])
        self.assertEqual(window.preset.page_fine_angles, angles)
        self.assertEqual(estimate.call_count, 2)
        self.assertEqual(progress[-1][0], 100)

    def test_redetection_forwards_progress_and_always_closes_dialog(self):
        dialog = SimpleNamespace(close=Mock())
        events: list[tuple[int, str]] = []

        def progress_cb(value: int, message: str = ""):
            events.append((value, message))

        window = SimpleNamespace(
            pages=[object()],
            file_paths=["survey.pdf"],
            _show_progress_dialog=Mock(return_value=dialog),
            _make_progress_cb=Mock(return_value=progress_cb),
            _wrap_progress=MainWindow._wrap_progress,
            auto_detect=Mock(
                side_effect=lambda progress_cb: progress_cb(
                    45, "템플릿 생성 중..."
                )
            ),
        )
        prepare = Mock(side_effect=lambda callback: callback(50, "준비 중..."))

        result = MainWindow._redetect_checkboxes_with_progress(
            window, "미세 회전 적용", before_detect=prepare
        )

        self.assertTrue(result)
        window._show_progress_dialog.assert_called_once_with(
            "미세 회전 적용", "체크박스 재탐색 준비 중..."
        )
        window.auto_detect.assert_called_once()
        prepare.assert_called_once()
        self.assertIn((5, "준비 중..."), events)
        self.assertIn((50, "템플릿 생성 중..."), events)
        self.assertEqual(events[-1], (100, "체크박스 재탐색 완료"))
        dialog.close.assert_called_once_with()

    def test_redetection_closes_dialog_when_detection_fails(self):
        dialog = SimpleNamespace(close=Mock())
        window = SimpleNamespace(
            pages=[object()],
            file_paths=["survey.pdf"],
            _show_progress_dialog=Mock(return_value=dialog),
            _make_progress_cb=Mock(return_value=Mock()),
            auto_detect=Mock(side_effect=RuntimeError("failed")),
        )

        with self.assertRaisesRegex(RuntimeError, "failed"):
            MainWindow._redetect_checkboxes_with_progress(window, "재탐색")

        dialog.close.assert_called_once_with()

    def test_redetection_restores_raw_pages_before_using_new_coordinates(self):
        dialog = SimpleNamespace(close=Mock())
        events: list[tuple[int, str]] = []

        def progress_cb(value: int, message: str = ""):
            events.append((value, message))

        reload_pages = Mock(
            side_effect=lambda callback: callback(100, "원본 페이지 준비 완료")
        )
        window = SimpleNamespace(
            pages=[object()],
            file_paths=["survey.pdf"],
            _pages_are_canonical=True,
            _show_progress_dialog=Mock(return_value=dialog),
            _make_progress_cb=Mock(return_value=progress_cb),
            _wrap_progress=MainWindow._wrap_progress,
            _reload_raw_template_pages=reload_pages,
            auto_detect=Mock(
                side_effect=lambda progress_cb: progress_cb(50, "탐지 중...")
            ),
        )

        result = MainWindow._redetect_checkboxes_with_progress(window, "재탐색")

        self.assertTrue(result)
        reload_pages.assert_called_once()
        self.assertIn((10, "원본 페이지 준비 완료"), events)
        self.assertIn((55, "탐지 중..."), events)
        dialog.close.assert_called_once_with()

    def test_cache_clear_redetects_through_progress_dialog(self):
        window = SimpleNamespace(
            pages=[object()],
            file_paths=["survey.pdf"],
            _redetect_checkboxes_with_progress=Mock(),
        )

        with (
            patch(
                "src.ui.QMessageBox.question",
                return_value=QMessageBox.StandardButton.Yes,
            ),
            patch("src.ui.clear_all_cache") as clear_cache,
        ):
            MainWindow.clear_cache(window)

        clear_cache.assert_not_called()
        window._redetect_checkboxes_with_progress.assert_called_once()
        call = window._redetect_checkboxes_with_progress.call_args
        self.assertEqual(call.args, ("캐시 삭제 후 재탐색",))
        callback = Mock()
        call.kwargs["before_detect"](callback)
        clear_cache.assert_called_once_with()
        self.assertEqual(callback.call_count, 2)

    def test_fine_angle_and_rotation_redetect_with_progress(self):
        window = SimpleNamespace(
            preset=SimpleNamespace(
                fine_angle=0.0,
                rot_code=-1,
                page_count=1,
                page_fine_angles=[],
            ),
            pages=[object()],
            rot_idx=0,
            _update_page_size=Mock(),
            _estimate_page_fine_angles=Mock(),
            _sync_rotation_actions=Mock(),
            _redetect_checkboxes_with_progress=Mock(
                side_effect=lambda _title, before_detect=None: (
                    before_detect(Mock()) if before_detect else True
                )
            ),
        )

        MainWindow.change_fine_angle(window, 1.5)

        self.assertEqual(window.preset.fine_angle, 1.5)
        window._redetect_checkboxes_with_progress.assert_called_with(
            "미세 회전 적용"
        )

        MainWindow.change_rotation(window, 1)

        self.assertEqual(window.preset.rot_code, ROTATION_MAP[1])
        call = window._redetect_checkboxes_with_progress.call_args
        self.assertEqual(call.args, ("회전 적용",))
        self.assertIn("before_detect", call.kwargs)
        window._estimate_page_fine_angles.assert_called_once()
        self.assertEqual(window._update_page_size.call_count, 2)


if __name__ == "__main__":
    unittest.main()
