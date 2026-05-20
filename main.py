import importlib.util
import subprocess
import sys


def ensure_packages() -> bool:
    required_packages = [
        ("PyQt6", "PyQt6"),
        ("cv2", "opencv-python-headless"),
        ("fitz", "PyMuPDF"),
        ("numpy", "numpy"),
        ("openpyxl", "openpyxl"),
        ("PIL", "Pillow"),
    ]

    for import_name, pip_name in required_packages:
        if importlib.util.find_spec(import_name) is None:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pip_name]
                )
            except Exception as exc:
                print(f"패키지 설치 실패: {pip_name} ({exc})")
                return False

    return True


if __name__ == "__main__":
    if not ensure_packages():
        sys.exit(1)

    from PyQt6.QtWidgets import QApplication

    from src.ui import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
