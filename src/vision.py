import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import fitz
import numpy as np
from pystackreg import StackReg


def _report_progress(progress_cb, value: int, message: str = ""):
    if progress_cb:
        progress_cb(value, message)


def _render_single_page(
    pdf_path: str, page_num: int, dpi: int, gray: bool = False
) -> np.ndarray:
    """Render a single page using its own fitz.Document instance (thread-safe)."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    if gray:
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)
    else:
        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()
    return img


def load_pdf_pages(
    pdf_path: str,
    dpi: int = 300,
    max_workers: int = 4,
    progress_cb=None,
    gray: bool = False,
) -> list[np.ndarray]:
    """Load PDF pages as OpenCV images with temp-cache and multi-threading."""

    stat = os.stat(pdf_path)
    unique_str = (
        f"{os.path.abspath(pdf_path)}_{stat.st_size}_{stat.st_mtime}_{dpi}_{gray}"
    )
    cache_key = hashlib.md5(unique_str.encode("utf-8")).hexdigest()

    cache_dir = os.path.join(tempfile.gettempdir(), "pdf_vision_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}.npz")

    _report_progress(progress_cb, 0, "PDF loading...")

    if os.path.exists(cache_file):
        try:
            with np.load(cache_file) as data:
                keys = sorted(data.files, key=lambda x: int(x.split("_")[1]))
                pages = [data[k] for k in keys]
                _report_progress(progress_cb, 100, "Generating checkboxes...")
                return pages
        except Exception as e:
            print(f"Cache load failed, regenerating: {e}")

    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    doc.close()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_render_single_page, pdf_path, p_num, dpi, gray): p_num
            for p_num in range(num_pages)
        }
        pages = [None] * num_pages
        completed = 0

        for future in as_completed(futures):
            p_num = futures[future]
            pages[p_num] = future.result()
            completed += 1
            _report_progress(
                progress_cb,
                int(completed / num_pages * 100),
                f"PDF loading... ({completed}/{num_pages})",
            )

    try:
        np.savez(cache_file, *pages)
    except Exception as e:
        print(f"Cache save failed: {e}")

    return pages


class ImageAligner:
    """TurboReg (pystackreg) based subpixel image alignment.

    Uses RIGID_BODY (rotation + translation, 3 DOF) optimized for scanned documents.
    Registration runs on downscaled images (short side 800px) for speed,
    then the transform is scaled up and applied at full resolution via cv2.warpAffine.
    """

    _REG_SHORT_SIDE = 800

    def __init__(self, ref_img: np.ndarray):
        self.ref_gray = (
            cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if ref_img.ndim == 3 else ref_img
        )
        self.ref_h, self.ref_w = self.ref_gray.shape[:2]
        self.sr = StackReg(StackReg.RIGID_BODY)

        ref_short = min(self.ref_h, self.ref_w)
        if ref_short > self._REG_SHORT_SIDE:
            self._scale = self._REG_SHORT_SIDE / ref_short
            self._ref_small = cv2.resize(
                self.ref_gray, None, fx=self._scale, fy=self._scale,
                interpolation=cv2.INTER_AREA
            ).astype(np.float64)
        else:
            self._scale = 1.0
            self._ref_small = self.ref_gray.astype(np.float64)

    def _resize_to_ref(self, img: np.ndarray) -> np.ndarray:
        if img.shape[:2] == (self.ref_h, self.ref_w):
            return img
        return cv2.resize(img, (self.ref_w, self.ref_h), interpolation=cv2.INTER_AREA)

    def align(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        if gray.shape[:2] != (self.ref_h, self.ref_w):
            gray = cv2.resize(
                gray, (self.ref_w, self.ref_h), interpolation=cv2.INTER_AREA
            )

        try:
            if self._scale < 1.0:
                gray_small = cv2.resize(
                    gray, None, fx=self._scale, fy=self._scale,
                    interpolation=cv2.INTER_AREA
                ).astype(np.float64)
            else:
                gray_small = gray.astype(np.float64)

            tmat_3x3 = self.sr.register(self._ref_small, gray_small)

            if self._scale < 1.0:
                tmat_3x3[0, 2] /= self._scale
                tmat_3x3[1, 2] /= self._scale

            M = tmat_3x3[:2, :].astype(np.float32)
            return cv2.warpAffine(
                img, M, (self.ref_w, self.ref_h), flags=cv2.INTER_LINEAR
            )
        except Exception:
            return self._resize_to_ref(img)


def apply_rotation(img: np.ndarray, rot_code: int, fine_angle: float) -> np.ndarray:
    if rot_code != -1:
        img = cv2.rotate(img, rot_code)
    if abs(fine_angle) >= 0.01:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), fine_angle, 1.0)
        img = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
    return img


def auto_detect_checkboxes(
    image: np.ndarray, min_w=24, max_w=600, min_h=24, max_h=200
) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 12))

    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    grid = cv2.add(h_lines, v_lines)
    grid = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=1)

    inv_grid = cv2.bitwise_not(grid)

    _, _, stats, _ = cv2.connectedComponentsWithStats(
        inv_grid, connectivity=4, ltype=cv2.CV_32S
    )

    boxes = []
    for stat in stats[1:]:
        x, y, w, h, area = stat
        if (min_w <= w <= max_w) and (min_h <= h <= max_h):
            aspect_ratio = w / float(h)
            if 0.1 <= aspect_ratio <= 10.0:
                boxes.append((int(x), int(y), int(w), int(h)))

    boxes = _remove_nested_boxes(boxes)
    boxes = _filter_isolated_boxes(boxes)

    return boxes


def _remove_nested_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    if len(boxes) <= 1:
        return boxes

    sorted_boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    keep = []
    for box in sorted_boxes:
        x, y, w, h = box
        contained = False
        for other in keep:
            ox, oy, ow, oh = other
            if ox <= x and oy <= y and ox + ow >= x + w and oy + oh >= y + h:
                contained = True
                break
        if not contained:
            keep.append(box)
    return keep


def _filter_isolated_boxes(
    boxes: list[tuple[int, int, int, int]],
    size_ratio: float = 0.1,
) -> list[tuple[int, int, int, int]]:
    if len(boxes) <= 1:
        return boxes

    def _size_similar(w, h, w2, h2):
        return (
            abs(w - w2) / max(w, w2) <= size_ratio
            and abs(h - h2) / max(h, h2) <= size_ratio
        )

    valid = []
    for i, (x, y, w, h) in enumerate(boxes):
        cx, cy = x + w / 2, y + h / 2

        hit_x = False
        hit_y = False

        for j, (x2, y2, w2, h2) in enumerate(boxes):
            if i == j:
                continue
            if y2 <= cy <= y2 + h2 and _size_similar(w, h, w2, h2):
                hit_x = True
            if x2 <= cx <= x2 + w2 and _size_similar(w, h, w2, h2):
                hit_y = True
            if hit_x and hit_y:
                break

        if hit_x or hit_y:
            valid.append((x, y, w, h))

    return valid


def _checkbox_cache_key(
    pdf_paths: list[str],
    page_count: int,
    rot_code: int,
    fine_angle: float,
    dpi: int = 300,
) -> str:
    parts = []
    for p in pdf_paths:
        try:
            stat = os.stat(p)
            parts.append(f"{os.path.abspath(p)}_{stat.st_size}_{stat.st_mtime}")
        except Exception:
            parts.append(p)
    unique_str = f"{'|'.join(parts)}_{page_count}_{rot_code}_{fine_angle}_{dpi}"
    return hashlib.md5(unique_str.encode("utf-8")).hexdigest()


def load_checkbox_cache(
    pdf_paths: list[str], page_count: int, rot_code: int, fine_angle: float
) -> dict[int, list[tuple[int, int, int, int]]] | None:
    key = _checkbox_cache_key(pdf_paths, page_count, rot_code, fine_angle)
    cache_dir = os.path.join(tempfile.gettempdir(), "pdf_checkbox_cache")
    cache_file = os.path.join(cache_dir, f"{key}.json")
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        result = {}
        for k, v in raw.items():
            result[int(k)] = [tuple(b) for b in v]
        return result
    except Exception:
        return None


def clear_all_cache():
    import shutil

    for cache_dir_name in ("pdf_vision_cache", "pdf_checkbox_cache"):
        cache_dir = os.path.join(tempfile.gettempdir(), cache_dir_name)
        if os.path.isdir(cache_dir):
            try:
                shutil.rmtree(cache_dir)
            except Exception:
                pass


def save_checkbox_cache(
    pdf_paths: list[str],
    page_count: int,
    rot_code: int,
    fine_angle: float,
    boxes_by_page: dict[int, list[tuple[int, int, int, int]]],
) -> None:
    key = _checkbox_cache_key(pdf_paths, page_count, rot_code, fine_angle)
    cache_dir = os.path.join(tempfile.gettempdir(), "pdf_checkbox_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{key}.json")
    try:
        raw = {str(k): [list(b) for b in v] for k, v in boxes_by_page.items()}
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(raw, f)
    except Exception:
        pass


def cleanup_old_cache(max_days: int = 30):
    import time

    now = time.time()
    cutoff = now - max_days * 86400
    for cache_dir_name in ("pdf_vision_cache", "pdf_checkbox_cache"):
        cache_dir = os.path.join(tempfile.gettempdir(), cache_dir_name)
        if not os.path.isdir(cache_dir):
            continue
        for fname in os.listdir(cache_dir):
            fpath = os.path.join(cache_dir, fname)
            try:
                if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                    os.remove(fpath)
            except Exception:
                pass
