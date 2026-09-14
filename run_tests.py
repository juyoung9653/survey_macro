"""Run the required suite, including real PDFs; skipped checks are not a pass."""
import os
from pathlib import Path
import unittest


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    root = Path(__file__).resolve().parent
    os.chdir(root)
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.skipped:
        print("전체 검증 미완료: 생략된 테스트가 있습니다.")
    raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)
