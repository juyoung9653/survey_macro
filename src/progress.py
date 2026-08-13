from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timedelta


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds + 0.5))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class ProgressTiming:
    """Track elapsed time and produce a smoothed completion estimate."""

    def __init__(
        self,
        monotonic_fn: Callable[[], float] = time.monotonic,
        wall_clock_fn: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._monotonic_fn = monotonic_fn
        self._wall_clock_fn = wall_clock_fn
        self._started_at = monotonic_fn()
        self._last_estimated_progress = 0.0
        self._estimated_total_seconds: float | None = None

    def elapsed_seconds(self) -> float:
        return max(0.0, self._monotonic_fn() - self._started_at)

    def _remaining_seconds(self, progress: float, elapsed: float) -> float | None:
        if progress >= 100.0:
            return 0.0
        if elapsed < 2.0 or progress < 5.0:
            return None

        if progress > self._last_estimated_progress:
            observed_total = elapsed * 100.0 / progress
            if self._estimated_total_seconds is None:
                self._estimated_total_seconds = observed_total
            else:
                self._estimated_total_seconds = (
                    self._estimated_total_seconds * 0.4 + observed_total * 0.6
                )
            self._last_estimated_progress = progress

        if self._estimated_total_seconds is None:
            return None
        return max(0.0, self._estimated_total_seconds - elapsed)

    def label(self, progress: float, message: str = "") -> str:
        progress = max(0.0, min(100.0, float(progress)))
        elapsed = self.elapsed_seconds()
        elapsed_text = format_duration(elapsed)
        prefix = message.strip() or "처리 중..."

        if progress >= 100.0:
            details = f"진행률 100.0%  ·  총 소요 {elapsed_text}"
            return f"{prefix}\n{details}"

        remaining = self._remaining_seconds(progress, elapsed)
        if remaining is None:
            details = (
                f"진행률 {progress:.1f}%  ·  경과 {elapsed_text}  ·  "
                "완료 예상 계산 중"
            )
            return f"{prefix}\n{details}"

        now = self._wall_clock_fn()
        completion = now + timedelta(seconds=remaining)
        if completion.date() == now.date():
            completion_text = completion.strftime("%H:%M")
        else:
            completion_text = completion.strftime("%m-%d %H:%M")
        details = (
            f"진행률 {progress:.1f}%  ·  경과 {elapsed_text}  ·  "
            f"남은 시간 약 {format_duration(remaining)}  ·  "
            f"완료 예상 {completion_text}"
        )
        return f"{prefix}\n{details}"
