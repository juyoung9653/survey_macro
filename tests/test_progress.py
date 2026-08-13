import unittest
from datetime import datetime

from src.progress import ProgressTiming, format_duration


class ProgressTimingTests(unittest.TestCase):
    def test_duration_format_uses_minutes_then_hours(self):
        self.assertEqual(format_duration(0), "00:00")
        self.assertEqual(format_duration(65), "01:05")
        self.assertEqual(format_duration(3661), "1:01:01")

    def test_label_shows_fractional_progress_elapsed_and_completion_time(self):
        monotonic = [100.0]
        wall_clock = [datetime(2026, 8, 13, 12, 0, 0)]
        timing = ProgressTiming(
            monotonic_fn=lambda: monotonic[0],
            wall_clock_fn=lambda: wall_clock[0],
        )

        monotonic[0] = 110.0
        wall_clock[0] = datetime(2026, 8, 13, 12, 0, 10)
        label = timing.label(25.25, "설문 분석 중")

        self.assertIn("설문 분석 중", label)
        self.assertIn("진행률 25.2%", label)
        self.assertIn("경과 00:10", label)
        self.assertIn("남은 시간 약 00:30", label)
        self.assertIn("완료 예상 12:00", label)

    def test_estimate_waits_for_enough_runtime_and_completion_shows_total(self):
        monotonic = [50.0]
        timing = ProgressTiming(monotonic_fn=lambda: monotonic[0])

        monotonic[0] = 51.0
        self.assertIn("완료 예상 계산 중", timing.label(10.0))

        monotonic[0] = 62.0
        completed = timing.label(100.0, "완료")
        self.assertIn("진행률 100.0%", completed)
        self.assertIn("총 소요 00:12", completed)


if __name__ == "__main__":
    unittest.main()
