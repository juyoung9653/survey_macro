import unittest
from unittest.mock import patch

from src.resources import (
    AdaptiveResourceController,
    MemorySnapshot,
    ResourceUnavailableError,
)


class AdaptiveResourceControllerTests(unittest.TestCase):
    def test_cpu_budget_targets_five_percent_after_external_load(self):
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

        self.assertEqual(status.opencv_threads, 3)
        self.assertEqual(
            [call.args[0] for call in set_threads.call_args_list], [4, 3, 4]
        )

    def test_fractional_cpu_budget_averages_to_ninety_five_percent(self):
        cpu_samples = iter((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        controller = AdaptiveResourceController(
            cpu_count=4,
            memory_provider=lambda: MemorySnapshot(1000, 1000),
            external_cpu_provider=lambda: next(cpu_samples),
            sleep_fn=lambda _seconds: None,
        )

        with (
            patch("src.resources.cv2.getNumThreads", return_value=4),
            patch("src.resources.cv2.setNumThreads"),
        ):
            controller.start()
            thread_counts = [controller.checkpoint().opencv_threads for _ in range(5)]
            controller.close()

        self.assertEqual(thread_counts, [3, 4, 4, 4, 4])
        self.assertEqual(sum(thread_counts) / len(thread_counts), 3.8)

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
            memory_reserve_fraction=0.25,
            memory_resume_fraction=0.32,
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
        cpu_samples = iter((0.0, 0.80, 0.68, 0.64))
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
            memory_reserve_fraction=0.25,
            memory_resume_fraction=0.32,
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
        self.assertEqual(status.reserve_memory_bytes, 250)

    def test_default_memory_gate_leaves_five_percent_after_predicted_work(self):
        memory_samples = iter(
            (
                MemorySnapshot(1000, 249),
                MemorySnapshot(1000, 250),
            )
        )
        controller = AdaptiveResourceController(
            cpu_count=4,
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

        self.assertEqual(status.available_memory_bytes, 250)
        self.assertEqual(status.reserve_memory_bytes, 50)

    def test_parallel_plan_is_bounded_by_current_memory_headroom(self):
        controller = AdaptiveResourceController(
            cpu_count=8,
            memory_provider=lambda: MemorySnapshot(1000, 550),
            external_cpu_provider=lambda: 0.0,
            sleep_fn=lambda _seconds: None,
        )

        with (
            patch("src.resources.cv2.getNumThreads", return_value=4),
            patch("src.resources.cv2.setNumThreads"),
        ):
            controller.start()
            plan = controller.parallel_checkpoint(
                required_memory_per_worker_bytes=100,
                pending_tasks=20,
            )
            controller.close()

        self.assertEqual(plan.total_cpu_threads, 7)
        self.assertEqual(plan.worker_count, 5)
        self.assertEqual(plan.opencv_threads, 1)

    def test_parallel_plan_shrinks_with_external_load_and_pending_work(self):
        cpu_samples = iter((0.0, 0.50))
        controller = AdaptiveResourceController(
            cpu_count=8,
            memory_provider=lambda: MemorySnapshot(1000, 1000),
            external_cpu_provider=lambda: next(cpu_samples),
            sleep_fn=lambda _seconds: None,
        )

        with (
            patch("src.resources.cv2.getNumThreads", return_value=4),
            patch("src.resources.cv2.setNumThreads"),
        ):
            controller.start()
            plan = controller.parallel_checkpoint(
                required_memory_per_worker_bytes=100,
                pending_tasks=2,
            )
            controller.close()

        self.assertEqual(plan.total_cpu_threads, 3)
        self.assertEqual(plan.worker_count, 2)
        self.assertEqual(plan.opencv_threads, 1)


if __name__ == "__main__":
    unittest.main()
