import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import fitz
import numpy as np


_PDF_CACHE_VERSION = 2
_CHECKBOX_CACHE_VERSION = 2


def _report_progress(progress_cb, value: int, message: str = ""):
    if progress_cb:
        progress_cb(value, message)


def _render_single_page(
    pdf_path: str, page_num: int, dpi: int, gray: bool = False
) -> np.ndarray:
    """멀티스레딩을 위해 개별 스레드에서 문서를 열고 렌더링하는 내부 헬퍼 함수
    gray=True 시 직접 흑백(2D)으로 렌더링 (BGR 대비 1/3 메모리)"""
    with fitz.open(pdf_path) as doc:
        page = doc[page_num]
        if gray:
            pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
            return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)

        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        if pix.n == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img


def load_pdf_pages(
    pdf_path: str,
    dpi: int = 300,
    max_workers: int = 4,
    progress_cb=None,
    gray: bool = False,
    page_indices: list[int] | None = None,
) -> list[np.ndarray]:
    """PDF를 읽어 OpenCV 이미지 리스트로 반환 (Temp 캐시 및 멀티스레딩 최적화)"""
    requested_indices = None if page_indices is None else list(page_indices)

    # 1. 고유 캐시 키 생성 (파일 경로 + 파일 크기 + 마지막 수정 시간 + DPI)
    # 원본 PDF 파일이 수정되거나 해상도(DPI) 설정이 바뀌면 자동으로 새로운 캐시를 생성합니다.
    stat = os.stat(pdf_path)
    unique_str = (
        f"v{_PDF_CACHE_VERSION}_{os.path.abspath(pdf_path)}_"
        f"{stat.st_size}_{stat.st_mtime_ns}_{dpi}_{gray}"
    )
    if requested_indices is not None:
        selection_key = json.dumps(requested_indices, separators=(",", ":"))
        unique_str = f"{unique_str}_{selection_key}"
    cache_key = hashlib.md5(unique_str.encode("utf-8")).hexdigest()

    # OS의 기본 Temp 폴더 (Windows의 경우 %TEMP%) 내에 전용 캐시 폴더 생성
    cache_dir = os.path.join(tempfile.gettempdir(), "pdf_vision_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}.npz")

    _report_progress(progress_cb, 0, "PDF 로딩 시작")
    if requested_indices == []:
        _report_progress(progress_cb, 100, "체크박스 자동 생성 중...")
        return []

    # 2. 캐시가 존재하면 PDF를 다시 읽지 않고 디스크에서 바로 로드 (초고속)
    if os.path.exists(cache_file):
        try:
            with np.load(cache_file) as data:
                # np.savez는 arr_0, arr_1 순서로 배열을 저장하므로, 순서에 맞게 정렬하여 불러옵니다.
                keys = sorted(data.files, key=lambda x: int(x.split("_")[1]))
                pages = [data[k] for k in keys]
                _report_progress(progress_cb, 100, "체크박스 자동 생성 중...")
                return pages
        except Exception as e:
            print(f"캐시 로드 실패, 새로 생성합니다: {e}")

    # 3. 캐시가 없을 경우 원본 PDF 처리 (멀티스레딩 적용)
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    doc.close()

    if requested_indices is None:
        render_indices = list(range(num_pages))
    else:
        render_indices = [
            page_num for page_num in requested_indices if 0 <= page_num < num_pages
        ]

    if not render_indices:
        _report_progress(progress_cb, 100, "체크박스 자동 생성 중...")
        return []

    # ThreadPoolExecutor를 사용해 여러 페이지를 동시에 렌더링합니다.
    # max_workers는 보통 CPU 코어 수에 맞추는 것이 좋으며, 기본값 4 정도가 안정적입니다.
    worker_count = min(max_workers, len(render_indices))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_render_single_page, pdf_path, p_num, dpi, gray): output_idx
            for output_idx, p_num in enumerate(render_indices)
        }
        pages_by_output: dict[int, np.ndarray] = {}
        completed = 0

        for future in as_completed(futures):
            output_idx = futures[future]
            pages_by_output[output_idx] = future.result()
            completed += 1
            _report_progress(
                progress_cb,
                int(completed / len(render_indices) * 100),
                f"PDF 로딩... ({completed}/{len(render_indices)})",
            )

    pages = [pages_by_output[i] for i in range(len(render_indices))]

    # 4. 다음 실행을 위해 결과를 Temp 폴더에 저장 (.npz 확장자)
    try:
        # 가변 인자(*pages)를 사용해 numpy 배열 리스트를 한 번에 저장합니다.
        np.savez(cache_file, *pages)
    except Exception as e:
        print(f"캐시 저장 실패: {e}")

    return pages


class ImageAligner:
    def __init__(self, ref_img: np.ndarray):
        self.ref_gray = (
            cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if ref_img.ndim == 3 else ref_img
        )
        self.ref_h, self.ref_w = self.ref_gray.shape[:2]
        self.orb = cv2.ORB_create(2000)
        self.kp1, self.des1 = self.orb.detectAndCompute(self.ref_gray, None)

    def _resize_to_ref(self, img: np.ndarray) -> np.ndarray:
        if img.shape[:2] == (self.ref_h, self.ref_w):
            return img
        return cv2.resize(img, (self.ref_w, self.ref_h), interpolation=cv2.INTER_AREA)

    def align(self, img: np.ndarray) -> np.ndarray:
        if self.des1 is None:
            return self._resize_to_ref(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        kp2, des2 = self.orb.detectAndCompute(gray, None)
        if des2 is None:
            return self._resize_to_ref(img)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(matcher.match(self.des1, des2), key=lambda m: m.distance)
        top_matches = matches[: max(4, int(len(matches) * 0.15))]

        if len(top_matches) < 3:
            return self._resize_to_ref(img)

        pts1 = np.float32([self.kp1[m.queryIdx].pt for m in top_matches]).reshape(
            -1, 1, 2
        )
        pts2 = np.float32([kp2[m.trainIdx].pt for m in top_matches]).reshape(-1, 1, 2)

        # affine transform: rotation + translation + scale, no perspective (no twisting)
        M, _ = cv2.estimateAffine2D(pts2, pts1, ransacReprojThreshold=3.0)
        if M is None:
            return self._resize_to_ref(img)

        # --- ECC 정밀 정합 (sub-pixel refinement) ---
        # ORB 특징점 기반 affine을 초기값으로, ECC로 픽셀 단위 미세 조정
        try:
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                50,
                1e-6,
            )
            # 입력 크기를 ref에 맞춰 resize (ECC는 같은 크기 요구)
            if gray.shape[:2] != (self.ref_h, self.ref_w):
                gray = cv2.resize(
                    gray, (self.ref_w, self.ref_h), interpolation=cv2.INTER_AREA
                )
            M_refined, _ = cv2.findTransformECC(
                self.ref_gray,
                gray,
                M.copy(),
                cv2.MOTION_AFFINE,
                criteria,
                None,
                5,
            )
            M = M_refined
        except Exception:
            pass  # ECC 실패 시 ORB 결과 그대로 사용

        return cv2.warpAffine(img, M, (self.ref_w, self.ref_h))


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

    # 1. 적응형 이진화: 스캔본의 그림자나 얼룩을 무시하고 테두리 선만 뚜렷하게 추출
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )

    # 2. 가로선/세로선 추출 (작은 체크박스도 잡기 위해 커널 크기를 12로 조정)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 12))

    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # 3. 선을 합쳐 표의 뼈대와 체크박스 테두리 완성
    grid = cv2.add(h_lines, v_lines)

    # 4. 스캔 손실로 인해 미세하게 끊어진 선들을 이어줌
    grid = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=1)

    # 5. 윤곽선 반전 (선이 검은색, 박스 안쪽 빈 공간이 흰색이 됨)
    inv_grid = cv2.bitwise_not(grid)

    # 6. 흰색 영역(박스 안쪽 공간) 찾기
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        inv_grid, connectivity=4, ltype=cv2.CV_32S
    )

    boxes = []
    for stat in stats[1:]:  # 0번은 보통 배경 전체이므로 건너뜀
        x, y, w, h, area = stat

        # 7. 비율 및 크기 조건 완화
        # 표 안의 길쭉한 직사각형 칸들도 모두 체크박스로 잡을 수 있도록 폭을 넓힘
        if (min_w <= w <= max_w) and (min_h <= h <= max_h):
            aspect_ratio = w / float(h)
            if 0.1 <= aspect_ratio <= 10.0:
                boxes.append((int(x), int(y), int(w), int(h)))

    # 8. 중첩 박스 제거 (더 큰 박스 안에 완전히 포함된 작은 박스 삭제)
    boxes = _remove_nested_boxes(boxes)

    # 9. 고립 박스 제거 (중심에서 선 뻤을 때 만나는 박스 없으면 삭제)
    boxes = _filter_isolated_boxes(boxes)

    return boxes


def _remove_nested_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """다른 박스 안에 완전히 포함된 내부 박스를 제거합니다."""
    if len(boxes) <= 1:
        return boxes

    # 면적 큰 순으로 정렬 (큰 박스가 먼저 오도록)
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
    """
    각 박스 중심에서 x축(가로), y축(세로)으로 선을 뻗어
    처음 만난 박스와 크기를 비교. 크기 비슷한 박스가 하나도 없으면 제거.

    size_ratio: w/h 각각 이 비율 이내 차이면 크기 비슷한 것으로 간주.
    """
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

        hit_x = False  # x축 선이 크기 비슷한 박스를 만남
        hit_y = False  # y축 선이 크기 비슷한 박스를 만남

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
            parts.append(f"{os.path.abspath(p)}_{stat.st_size}_{stat.st_mtime_ns}")
        except Exception:
            parts.append(p)
    unique_str = (
        f"v{_CHECKBOX_CACHE_VERSION}_{'|'.join(parts)}_"
        f"{page_count}_{rot_code}_{fine_angle}_{dpi}"
    )
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
    """체크박스, PDF 페이지, UI 템플릿 캐시를 모두 삭제합니다."""
    import shutil

    for cache_dir_name in (
        "pdf_vision_cache",
        "pdf_checkbox_cache",
        "pdf_ui_template_cache",
    ):
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
    """temp 폴더 내 PDF 캐시 중 max_d일 지난 파일 삭제"""
    import time

    now = time.time()
    cutoff = now - max_days * 86400
    for cache_dir_name in (
        "pdf_vision_cache",
        "pdf_checkbox_cache",
        "pdf_ui_template_cache",
    ):
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
