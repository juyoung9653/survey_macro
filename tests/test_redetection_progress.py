import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PyQt6.QtWidgets import QMessageBox

from src.ui import MainWindow, ROTATION_MAP


class RedetectionProgressTests(unittest.TestCase):
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
            auto_detect=Mock(
                side_effect=lambda progress_cb: progress_cb(
                    45, "템플릿 생성 중..."
                )
            ),
        )
        prepare = Mock()

        result = MainWindow._redetect_checkboxes_with_progress(
            window, "미세 회전 적용", before_detect=prepare
        )

        self.assertTrue(result)
        window._show_progress_dialog.assert_called_once_with(
            "미세 회전 적용", "체크박스 재탐색 준비 중..."
        )
        window.auto_detect.assert_called_once_with(progress_cb=progress_cb)
        prepare.assert_called_once_with()
        self.assertIn((0, "캐시 삭제 중..."), events)
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
        window._redetect_checkboxes_with_progress.assert_called_once_with(
            "캐시 삭제 후 재탐색", before_detect=clear_cache
        )

    def test_fine_angle_and_rotation_redetect_with_progress(self):
        window = SimpleNamespace(
            preset=SimpleNamespace(fine_angle=0.0, rot_code=-1),
            pages=[object()],
            rot_idx=0,
            _update_page_size=Mock(),
            _sync_rotation_actions=Mock(),
            _redetect_checkboxes_with_progress=Mock(),
        )

        MainWindow.change_fine_angle(window, 1.5)

        self.assertEqual(window.preset.fine_angle, 1.5)
        window._redetect_checkboxes_with_progress.assert_called_with(
            "미세 회전 적용"
        )

        MainWindow.change_rotation(window, 1)

        self.assertEqual(window.preset.rot_code, ROTATION_MAP[1])
        window._redetect_checkboxes_with_progress.assert_called_with("회전 적용")
        self.assertEqual(window._update_page_size.call_count, 2)


if __name__ == "__main__":
    unittest.main()
