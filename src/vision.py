import hashlib
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import fitz
import numpy as np


def _report_progress(progress_cb, value: int, message: str = ""):
    if progress_cb:
        progress_cb(value, message)


def _render_single_page(
    pdf_path: str, page_num: int, dpi: int, gray: bool = False
) -> np.ndarray:
    """멀티스레딩을 위해 개별 스레드에서 문서를 열고 렌더링하는 내부 헬퍼 함수
    gray=True 시 직접 흑백(2D)으로 렌더링 (BGR 대비 1/3 메모리)"""
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
    dpi: int = 200,
    max_workers: int = 4,
    progress_cb=None,
    gray: bool = False,
) -> list[np.ndarray]:
    """PDF를 읽어 OpenCV 이미지 리스트로 반환 (Temp 캐시 및 멀티스레딩 최적화)"""

    # 1. 고유 캐시 키 생성 (파일 경로 + 파일 크기 + 마지막 수정 시간 + DPI)
    # 원본 PDF 파일이 수정되거나 해상도(DPI) 설정이 바뀌면 자동으로 새로운 캐시를 생성합니다.
    stat = os.stat(pdf_path)
    unique_str = f"{os.path.abspath(pdf_path)}_{stat.st_size}_{stat.st_mtime}_{dpi}"
    cache_key = hashlib.md5(unique_str.encode("utf-8")).hexdigest()

    # OS의 기본 Temp 폴더 (Windows의 경우 %TEMP%) 내에 전용 캐시 폴더 생성
    cache_dir = os.path.join(tempfile.gettempdir(), "pdf_vision_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}.npz")

    _report_progress(progress_cb, 0, "PDF 로딩 시작")

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

    # ThreadPoolExecutor를 사용해 여러 페이지를 동시에 렌더링합니다.
    # max_workers는 보통 CPU 코어 수에 맞추는 것이 좋으며, 기본값 4 정도가 안정적입니다.
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
                f"PDF 로딩... ({completed}/{num_pages})",
            )

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

        if len(top_matches) < 4:
            return self._resize_to_ref(img)

        pts1 = np.float32([self.kp1[m.queryIdx].pt for m in top_matches]).reshape(
            -1, 1, 2
        )
        pts2 = np.float32([kp2[m.trainIdx].pt for m in top_matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)

        if H is not None:
            return cv2.warpPerspective(img, H, (self.ref_w, self.ref_h))

        M, _ = cv2.estimateAffinePartial2D(pts2, pts1)
        if M is not None:
            return cv2.warpAffine(img, M, (self.ref_w, self.ref_h))

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
    image: np.ndarray, min_w=10, max_w=600, min_h=10, max_h=200
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

    return boxes
