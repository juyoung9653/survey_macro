import unittest
from unittest.mock import patch

from src.resources import (
    AdaptiveResourceController,
    MemorySnapshot,
    ResourceUnavailableError,
)


class AdaptiveResourceControllerTests(unittest.TestCase):
    def test_cpu_budget_leaves_one_of_four_logical_cores_free(self):
        cpu_samples = iter((0.0, 0.20))
        controller = AdaptiveResourceController(
            cpu_count=4,
            memory_provider=lambda: MemorySnapshot(1000, 1000),
            external_cpu_provider=lambda: next(cpu_samples),
            sleep_fn=lambda _seconds: None,
        )

        with (
            patch("src.resources.cv2.getNumThreads", return_value=4),
            patch("src.resources.cv2.setNumThreads") as set_threads,
        ):
            controller.start()
            status = controller.checkpoint()
            controller.close()

        self.assertEqual(status.opencv_threads, 2)
        self.assertEqual(
            [call.args[0] for call in set_threads.call_args_list], [3, 2, 4]
        )

    def test_memory_pause_uses_resume_hysteresis(self):
        memory_samples = iter(
            (
                MemorySnapshot(1000, 200),
                MemorySnapshot(1000, 260),
                MemorySnapshot(1000, 330),
            )
        )
        sleeps = []
        messages = []
        controller = AdaptiveResourceController(
            cpu_count=4,
            reserve_fraction=0.25,
            resume_fraction=0.32,
            safety_fraction=0.0,
            memory_provider=lambda: next(memory_samples),
            external_cpu_provider=lambda: 0.0,
            sleep_fn=sleeps.append,
        )

        with (
            patch("src.resources.cv2.getNumThreads", return_value=4),
            patch("src.resources.cv2.setNumThreads"),
        ):
            controller.start()
            status = controller.checkpoint(status_cb=messages.append)
            controller.close()

        self.assertEqual(status.available_memory_bytes, 330)
        self.assertEqual(sleeps, [0.5, 0.5])
        self.assertEqual(len(messages), 2)
        self.assertTrue(all("CPU" not in message for message in messages))
        self.assertTrue(all("RAM" not in message for message in messages))
        self.assertTrue(all("시스템 여유 확보 중" in message for message in messages))

    def test_timeout_error_does_not_expose_cpu_or_memory_values(self):
        controller = AdaptiveResourceController(
            cpu_count=4,
            max_wait_seconds=0.0,
            memory_provider=lambda: MemorySnapshot(1000, 100),
            external_cpu_provider=lambda: 0.9,
            sleep_fn=lambda _seconds: None,
        )

        with (
            patch("src.resources.cv2.getNumThreads", return_value=4),
            patch("src.resources.cv2.setNumThreads"),
        ):
            controller.start()
            with self.assertRaises(ResourceUnavailableError) as raised:
                controller.checkpoint(stage="설문 분석")
            controller.close()

        message = str(raised.exception)
        self.assertNotIn("CPU", message)
        self.assertNotIn("RAM", message)
        self.assertIn("다른 프로그램을 닫고", message)

    def test_cpu_pause_resumes_below_hysteresis_threshold(self):
        cpu_samples = iter((0.0, 0.60, 0.48, 0.44))
        sleeps = []
        controller = AdaptiveResourceController(
            cpu_count=4,
            memory_provider=lambda: MemorySnapshot(1000, 1000),
            external_cpu_provider=lambda: next(cpu_samples),
            sleep_fn=sleeps.append,
        )

        with (
            patch("src.resources.cv2.getNumThreads", return_value=4),
            patch("src.resources.cv2.setNumThreads"),
        ):
            controller.start()
            status = controller.checkpoint()
            controller.close()

        self.assertEqual(sleeps, [0.5, 0.5])
        self.assertEqual(status.opencv_threads, 1)

    def test_predicted_working_memory_is_reserved_before_start(self):
        memory_samples = iter(
            (
                MemorySnapshot(1000, 440),
                MemorySnapshot(1000, 500),
            )
        )
        controller = AdaptiveResourceController(
            cpu_count=4,
            reserve_fraction=0.25,
            resume_fraction=0.32,
            safety_fraction=0.0,
            memory_provider=lambda: next(memory_samples),
            external_cpu_provider=lambda: 0.0,
            sleep_fn=lambda _seconds: None,
        )

        with (
            patch("src.resources.cv2.getNumThreads", return_value=4),
            patch("src.resources.cv2.setNumThreads"),
        ):
            controller.start()
            status = controller.checkpoint(required_memory_bytes=200)
            controller.close()

        self.assertEqual(status.available_memory_bytes, 500)


if __name__ == "__main__":
    unittest.main()
