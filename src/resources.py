import ctypes
import math
import os
import time
from dataclasses import dataclass
from typing import Callable

import cv2


_MIB = 1024 * 1024


@dataclass(frozen=True)
class MemorySnapshot:
    total_bytes: int
    available_bytes: int


@dataclass(frozen=True)
class ResourceStatus:
    opencv_threads: int
    external_cpu_fraction: float
    available_memory_bytes: int
    reserve_memory_bytes: int


class ResourceUnavailableError(RuntimeError):
    pass


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _FileTime(ctypes.Structure):
    _fields_ = [
        ("dwLowDateTime", ctypes.c_ulong),
        ("dwHighDateTime", ctypes.c_ulong),
    ]


def _file_time_value(value: _FileTime) -> int:
    return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)


def get_memory_snapshot() -> MemorySnapshot:
    if os.name == "nt":
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError()
        return MemorySnapshot(int(status.ullTotalPhys), int(status.ullAvailPhys))

    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    total_pages = int(os.sysconf("SC_PHYS_PAGES"))
    available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    return MemorySnapshot(page_size * total_pages, page_size * available_pages)


class ExternalCpuSampler:
    """Measure system CPU use excluding this process between checkpoints."""

    def __init__(self):
        self._previous: tuple[int, int, int, int] | None = None
        self._fallback_previous: tuple[float, float] | None = None

    def __call__(self) -> float:
        if os.name == "nt":
            return self._sample_windows()
        return self._sample_fallback()

    def _sample_windows(self) -> float:
        idle = _FileTime()
        kernel = _FileTime()
        user = _FileTime()
        kernel32 = ctypes.windll.kernel32
        kernel32.GetSystemTimes.argtypes = [
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
        ]
        kernel32.GetSystemTimes.restype = ctypes.c_int
        if not kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        ):
            return 0.0

        creation = _FileTime()
        exit_time = _FileTime()
        process_kernel = _FileTime()
        process_user = _FileTime()
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetProcessTimes.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
        ]
        kernel32.GetProcessTimes.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        if not kernel32.GetProcessTimes(
            process,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(process_kernel),
            ctypes.byref(process_user),
        ):
            return 0.0

        current = (
            _file_time_value(idle),
            _file_time_value(kernel),
            _file_time_value(user),
            _file_time_value(process_kernel) + _file_time_value(process_user),
        )
        previous = self._previous
        self._previous = current
        if previous is None:
            return 0.0

        idle_delta = max(0, current[0] - previous[0])
        kernel_delta = max(0, current[1] - previous[1])
        user_delta = max(0, current[2] - previous[2])
        process_delta = max(0, current[3] - previous[3])
        total_delta = kernel_delta + user_delta
        if total_delta <= 0:
            return 0.0
        system_busy = max(0, total_delta - idle_delta)
        external_busy = max(0, system_busy - process_delta)
        return min(1.0, external_busy / total_delta)

    def _sample_fallback(self) -> float:
        now = time.monotonic()
        process_time = time.process_time()
        previous = self._fallback_previous
        self._fallback_previous = (now, process_time)
        if previous is None:
            return 0.0

        wall_delta = max(1e-6, now - previous[0])
        process_delta = max(0.0, process_time - previous[1])
        cpu_count = max(1, os.cpu_count() or 1)
        try:
            system_fraction = min(1.0, os.getloadavg()[0] / cpu_count)
        except (AttributeError, OSError):
            system_fraction = 0.0
        process_fraction = min(1.0, process_delta / (wall_delta * cpu_count))
        return max(0.0, system_fraction - process_fraction)


class AdaptiveResourceController:
    """Keep a best-effort CPU/RAM reserve at safe processing checkpoints."""

    def __init__(
        self,
        reserve_fraction: float = 0.25,
        resume_fraction: float = 0.32,
        safety_fraction: float = 0.02,
        max_wait_seconds: float = 120.0,
        poll_interval: float = 0.5,
        cpu_count: int | None = None,
        memory_provider: Callable[[], MemorySnapshot] | None = None,
        external_cpu_provider: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        if not 0.0 < reserve_fraction < 1.0:
            raise ValueError("reserve_fraction must be between 0 and 1")
        if not reserve_fraction < resume_fraction < 1.0:
            raise ValueError("resume_fraction must exceed reserve_fraction")

        self.reserve_fraction = reserve_fraction
        self.resume_fraction = resume_fraction
        self.safety_fraction = max(0.0, safety_fraction)
        self.max_wait_seconds = max(0.0, max_wait_seconds)
        self.poll_interval = max(0.05, poll_interval)
        self.cpu_count = max(1, cpu_count or os.cpu_count() or 1)
        self.max_opencv_threads = max(
            1, math.floor(self.cpu_count * (1.0 - reserve_fraction))
        )
        self._memory_provider = memory_provider or get_memory_snapshot
        self._external_cpu_provider = external_cpu_provider or ExternalCpuSampler()
        self._sleep = sleep_fn
        self._previous_opencv_threads: int | None = None
        self._current_opencv_threads: int | None = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()

    def start(self) -> None:
        if self._previous_opencv_threads is not None:
            return
        self._previous_opencv_threads = max(1, cv2.getNumThreads())
        self._external_cpu_provider()  # prime delta-based samplers
        self._set_opencv_threads(self.max_opencv_threads)

    def close(self) -> None:
        if self._previous_opencv_threads is None:
            return
        cv2.setNumThreads(self._previous_opencv_threads)
        self._current_opencv_threads = self._previous_opencv_threads
        self._previous_opencv_threads = None

    def checkpoint(
        self,
        required_memory_bytes: int = 0,
        stage: str = "분석",
        status_cb: Callable[[str], None] | None = None,
    ) -> ResourceStatus:
        if self._previous_opencv_threads is None:
            self.start()

        required_memory_bytes = max(0, int(required_memory_bytes))
        started = time.monotonic()
        memory_paused = False
        cpu_paused = False

        while True:
            memory = self._memory_provider()
            external_cpu = min(1.0, max(0.0, self._external_cpu_provider()))
            reserve_bytes = math.ceil(memory.total_bytes * self.reserve_fraction)
            safety_bytes = math.ceil(memory.total_bytes * self.safety_fraction)
            start_threshold = reserve_bytes + safety_bytes + required_memory_bytes
            resume_threshold = max(
                start_threshold,
                math.ceil(memory.total_bytes * self.resume_fraction),
            )
            memory_threshold = resume_threshold if memory_paused else start_threshold

            minimum_thread_fraction = 1.0 / self.cpu_count
            cpu_pause_threshold = max(
                0.0, 1.0 - self.reserve_fraction - minimum_thread_fraction
            )
            cpu_resume_threshold = max(0.0, cpu_pause_threshold - 0.05)
            current_cpu_threshold = (
                cpu_resume_threshold if cpu_paused else cpu_pause_threshold
            )
            memory_blocked = memory.available_bytes < memory_threshold
            cpu_blocked = external_cpu > current_cpu_threshold

            if not memory_blocked and not cpu_blocked:
                available_for_app = max(
                    minimum_thread_fraction,
                    1.0 - self.reserve_fraction - external_cpu,
                )
                desired_threads = math.floor(
                    self.cpu_count * available_for_app + 1e-9
                )
                desired_threads = min(
                    self.max_opencv_threads, max(1, desired_threads)
                )
                self._set_opencv_threads(desired_threads)
                return ResourceStatus(
                    desired_threads,
                    external_cpu,
                    memory.available_bytes,
                    reserve_bytes,
                )

            memory_paused = memory_paused or memory_blocked
            cpu_paused = cpu_paused or cpu_blocked
            if time.monotonic() - started >= self.max_wait_seconds:
                raise ResourceUnavailableError(
                    f"{stage} 중 시스템 여유를 확보하지 못했습니다. "
                    "다른 프로그램을 닫고 다시 시도해주세요."
                )

            self._set_opencv_threads(1)
            if status_cb:
                status_cb(f"{stage}: 시스템 여유 확보 중...")
            self._sleep(self.poll_interval)

    def _set_opencv_threads(self, thread_count: int) -> None:
        thread_count = max(1, int(thread_count))
        if thread_count == self._current_opencv_threads:
            return
        cv2.setNumThreads(thread_count)
        self._current_opencv_threads = thread_count
