import copy
import gc
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np

from .export import export_to_excel
from .models import Box, Field, TemplatePreset
from .resources import AdaptiveResourceController, ResourceUnavailableError
from .vision import (
    ImageAligner,
    apply_rotation,
    auto_detect_checkboxes,
    load_pdf_pages,
)


_UI_TEMPLATE_SAMPLE_LIMIT = 31
_UI_TEMPLATE_CACHE_VERSION = 5
_CHECKBOX_MIN_BORDER_CONFIDENCE = 0.35
_MIB = 1024 * 1024
_ANALYSIS_SAMPLE_WORK = 2.0
_ANALYSIS_TEMPLATE_WORK = 1.0
_ANALYSIS_REUSED_SURVEY_WORK = 1.0
_ANALYSIS_RENDERED_SURVEY_WORK = 3.0
_ANALYSIS_PROGRESS_START = 2.0
_ANALYSIS_PROGRESS_SPAN = 95.0


@dataclass
class _CheckboxInkInfo:
    ink_pixels: int
    area: int
    box: Box
    border_confidence: float
    mask_bounds: tuple[int, int, int, int]
    ink_mask: np.ndarray


@dataclass
class _CheckboxHaloInfo:
    ink_pixels: int
    area: int
    box: Box
    mask_bounds: tuple[int, int, int, int]
    ink_mask: np.ndarray


@dataclass
class _CheckboxFieldAnalysis:
    checkbox_infos: list[_CheckboxInkInfo]
    reliable_direct: list[bool]
    direct_inks: list[int]
    direct_areas: list[int]
    direct_results: list[bool]
    halo_infos: list[_CheckboxHaloInfo]


def _ui_template_cache_path(
    pdf_paths: list[str],
    page_count: int,
    rot_code: int,
    fine_angle: float,
    mode: str,
) -> Path | None:
    parts = [
        f"v{_UI_TEMPLATE_CACHE_VERSION}",
        mode,
        str(page_count),
        str(rot_code),
        str(fine_angle),
        str(_UI_TEMPLATE_SAMPLE_LIMIT),
    ]
    try:
        for path in pdf_paths:
            stat = os.stat(path)
            parts.append(
                f"{os.path.abspath(path)}:{stat.st_size}:{stat.st_mtime_ns}"
            )
    except OSError:
        return None

    key = hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()
    return Path(tempfile.gettempdir()) / "pdf_ui_template_cache" / f"{key}.npz"


def _load_ui_template_cache(cache_path: Path | None) -> dict[int, np.ndarray] | None:
    if cache_path is None or not cache_path.exists():
        return None
    try:
        with np.load(cache_path) as data:
            indices = data["arr_0"].astype(int).tolist()
            templates = {
                int(page_index): data[f"arr_{array_index + 1}"]
                for array_index, page_index in enumerate(indices)
            }
        os.utime(cache_path, None)
        return templates or None
    except Exception:
        return None


def _save_ui_template_cache(
    cache_path: Path | None, templates: dict[int, np.ndarray]
) -> None:
    if cache_path is None or not templates:
        return
    temp_path = cache_path.with_name(f"{cache_path.name}.tmp.npz")
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ordered_items = sorted(templates.items())
        indices = np.array([key for key, _ in ordered_items], dtype=np.int32)
        arrays = [value for _, value in ordered_items]
        np.savez_compressed(str(temp_path), indices, *arrays)
        os.replace(temp_path, cache_path)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _file_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _build_file_labels(file_paths: list[str]) -> list[str]:
    """Windows 대소문자 규칙까지 고려해 Excel/PDF 출력명을 유일하게 만듭니다."""
    stems = [Path(path).stem for path in file_paths]
    normalized = [os.path.normcase(stem) for stem in stems]
    totals = {name: normalized.count(name) for name in set(normalized)}
    reserved = set(normalized)
    used: set[str] = set()
    labels = []

    for stem, normalized_stem in zip(stems, normalized):
        if totals[normalized_stem] == 1 and normalized_stem not in used:
            label = stem
        else:
            suffix = 1
            while True:
                candidate = f"{stem}_{suffix}"
                normalized_candidate = os.path.normcase(candidate)
                if normalized_candidate not in reserved and normalized_candidate not in used:
                    label = candidate
                    break
                suffix += 1
        used.add(os.path.normcase(label))
        labels.append(label)

    return labels


def _encode_jpeg(img: np.ndarray, quality: int = 85) -> bytes | None:
    """그레이/BGR 이미지를 불필요한 색공간 복사 없이 JPEG로 인코딩합니다."""
    success, buf = cv2.imencode(
        ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    return buf.tobytes() if success else None



def _build_vector_page(
    target_doc, base_img: np.ndarray, annotations: list, img_quality: int = 85
) -> None:
    """벡터 PDF 페이지 생성: 배경 이미지(JPEG) + 벡터 사각형/텍스트 오버레이."""
    h, w = base_img.shape[:2]
    page = target_doc.new_page(width=w, height=h)
    image_bytes = _encode_jpeg(base_img, img_quality)
    if image_bytes:
        page.insert_image(page.rect, stream=image_bytes)
    for bx, by, bw, bh, label, is_ticked in annotations:
        color = (0, 1, 0) if is_ticked else (1, 0, 0)
        rect = fitz.Rect(bx, by, bx + bw, by + bh)
        page.draw_rect(rect, color=color, width=2)
        page.insert_text(fitz.Point(bx, max(0, by - 5)), label, fontsize=8, color=color)


def _insert_encoded_img_into_pdf(target_doc, image_bytes: bytes) -> None:
    """인코딩된 JPEG 한 장을 target_doc에 추가합니다."""
    img_doc = fitz.open("jpg", image_bytes)
    page_doc = None
    try:
        pdf_bytes = img_doc.convert_to_pdf()
        page_doc = fitz.open("pdf", pdf_bytes)
        target_doc.insert_pdf(page_doc)
    finally:
        if page_doc is not None:
            page_doc.close()
        img_doc.close()


def _insert_img_into_pdf(target_doc, img: np.ndarray, quality: int = 85) -> None:
    """numpy 이미지를 JPEG로 인코딩해 target_doc에 페이지로 추가합니다."""
    image_bytes = _encode_jpeg(img, quality)
    if image_bytes:
        _insert_encoded_img_into_pdf(target_doc, image_bytes)


def sort_boxes_z_pattern(boxes: list[Box]) -> list[Box]:
    sorted_boxes = []
    pages = sorted(list(set(b.page_idx for b in boxes)))
    for p in pages:
        p_boxes = [b for b in boxes if b.page_idx == p]
        p_boxes.sort(key=lambda b: b.y)
        if not p_boxes:
            continue

        rows = []
        current_row = [p_boxes[0]]
        row_y_threshold = 15

        for b in p_boxes[1:]:
            if abs(b.y - current_row[0].y) <= row_y_threshold:
                current_row.append(b)
            else:
                rows.append(sorted(current_row, key=lambda x: x.x))
                current_row = [b]
        if current_row:
            rows.append(sorted(current_row, key=lambda x: x.x))

        sorted_boxes.extend([box for row in rows for box in row])
    return sorted_boxes


def is_contiguous_group(boxes: list[Box]) -> bool:
    if len(boxes) <= 1:
        return False

    for i in range(len(boxes) - 1):
        for j in range(i + 1, len(boxes)):
            b1, b2 = boxes[i], boxes[j]
            if b1.page_idx == b2.page_idx and abs(b1.y - b2.y) < 15:
                left, right = (b1, b2) if b1.x < b2.x else (b2, b1)
                gap = right.x - (left.x + left.w)

                # 간격이 15픽셀 미만이거나 박스끼리 겹쳐있는(음수) 경우 무조건 뭉쳐있는 것으로 판단
                if gap < 15:
                    return True

    return False


def expand_isolated_boxes(
    boxes: list[Box], all_boxes: list[Box], scale_factor: float = 2.0
) -> list[Box]:
    """자신의 문항뿐만 아니라 문서 전체의 박스(all_boxes)를 대상으로 충돌을 검사합니다."""
    expanded = []
    for box in boxes:
        new_box = copy.copy(box)

        target_w = box.w * scale_factor
        target_h = box.h * scale_factor

        dw = (target_w - box.w) / 2
        dh = (target_h - box.h) / 2

        max_dw, max_dh = dw, dh

        # 수정됨: boxes가 아닌 all_boxes와 비교하여 다른 문항의 박스도 침범하지 않도록 함
        for other in all_boxes:
            # 자기 자신과는 비교하지 않음 (객체 메모리 주소로 비교)
            if box is other or box.page_idx != other.page_idx:
                continue

            cx1, cy1 = box.x + box.w / 2, box.y + box.h / 2
            cx2, cy2 = other.x + other.w / 2, other.y + other.h / 2

            dist_x = abs(cx1 - cx2) - (box.w + other.w) / 2
            dist_y = abs(cy1 - cy2) - (box.h + other.h) / 2

            if abs(cy1 - cy2) < (box.h + other.h) / 2 + 15:
                if dist_x > 0:
                    max_dw = min(max_dw, dist_x / 2.1)
                else:
                    max_dw = min(max_dw, 2)

            if abs(cx1 - cx2) < (box.w + other.w) / 2 + 15:
                if dist_y > 0:
                    max_dh = min(max_dh, dist_y / 2.1)
                else:
                    max_dh = min(max_dh, 2)

        max_dw = max(0, max_dw)
        max_dh = max(0, max_dh)

        new_box.x = int(max(0, box.x - max_dw))
        new_box.y = int(max(0, box.y - max_dh))
        new_box.w = int(box.w + max_dw * 2)
        new_box.h = int(box.h + max_dh * 2)

        expanded.append(new_box)

    return expanded


# ==========================================
# 1. 템플릿 및 잉크 추출 모듈
# ==========================================


def _median_uint8_inplace(stack: np.ndarray) -> np.ndarray:
    """uint8 스택을 제자리 partition해 np.median(...).astype(uint8)과 동일하게 계산."""
    count = stack.shape[0]
    if count == 1:
        return stack[0].copy()

    upper = count // 2
    if count % 2:
        stack.partition(upper, axis=0)
        return stack[upper].copy()

    lower = upper - 1
    stack.partition((lower, upper), axis=0)
    total = stack[lower].astype(np.uint16)
    total += stack[upper]
    return (total // 2).astype(np.uint8)


def generate_dynamic_templates(
    pages_by_local_idx: dict[int, list],
) -> dict[int, np.ndarray]:
    templates = {}
    for local_p, pages in pages_by_local_idx.items():
        if not pages:
            continue

        if isinstance(pages[0], bytes):
            images = []
            for data in pages:
                if not data:
                    continue
                image = cv2.imdecode(
                    np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE
                )
                if image is not None:
                    images.append(image)
        else:
            images = list(pages)

        filtered = _filter_blank_pages(images)
        if not filtered:
            continue

        stack = np.stack(filtered, axis=0)
        del images, filtered
        templates[local_p] = _median_uint8_inplace(stack)
        del stack

    return templates


def _filter_blank_pages(
    images: list[np.ndarray], std_thresh: float = 1.5
) -> list[np.ndarray]:
    """평균보다 현저히 어두운(잉크 많은) 이미지를 제외하고 깨끗한 페이지만 반환."""
    if len(images) <= 3:
        return images  # 표본 적으면 필터 의미 없음
    means = np.array([np.mean(img) for img in images])
    mean_of_means = np.mean(means)
    std_of_means = np.std(means)
    if std_of_means == 0:
        return images
    # 밝기 임계값: 평균 - N*표준편차 보다 어두우면 outlier
    threshold = mean_of_means - std_thresh * std_of_means
    return [img for img, m in zip(images, means) if m >= threshold]


def generate_ui_templates(
    pdf_path: str,
    page_count: int,
    rot_code: int,
    fine_angle: float,
    progress_cb=None,
) -> dict[int, np.ndarray]:
    """UI에서 자동 탐지를 수행하기 전, PDF 전체를 읽어 깔끔한 빈 템플릿을 생성해 반환합니다."""
    if page_count <= 0:
        return {}

    cache_path = _ui_template_cache_path(
        [pdf_path], page_count, rot_code, fine_angle, "single"
    )
    cached_templates = _load_ui_template_cache(cache_path)
    if cached_templates is not None:
        if progress_cb:
            progress_cb(100, "캐시된 템플릿 불러오기 완료")
        return {
            key: cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
            for key, value in cached_templates.items()
        }

    try:
        with fitz.open(pdf_path) as doc:
            sample_page_count = min(
                len(doc), page_count * _UI_TEMPLATE_SAMPLE_LIMIT
            )
        pages = load_pdf_pages(
            pdf_path,
            progress_cb=progress_cb,
            gray=True,
            page_indices=list(range(sample_page_count)),
        )
    except Exception:
        return {}

    if not pages:
        return {}

    # 체크박스 테두리는 얇아서 표본마다 ECC의 미세 affine 변형이 달라지면
    # 중앙값 템플릿에서 끊어질 수 있습니다. 템플릿 합성은 ORB 정합만 사용합니다.
    aligners = [
        ImageAligner(
            apply_rotation(p, rot_code, fine_angle),
            refine_ecc=False,
        )
        for p in pages[:page_count]
    ]

    survey_count = _survey_count(len(pages), page_count)

    pages_by_local_idx = {i: [] for i in range(page_count)}

    for survey_idx in range(survey_count):
        for local_p in range(page_count):
            global_p = survey_idx * page_count + local_p
            if global_p >= len(pages):
                break

            orig = apply_rotation(pages[global_p], rot_code, fine_angle)
            aligner = aligners[local_p] if local_p < len(aligners) else aligners[-1]

            aligned = aligner.align(orig)
            if len(pages_by_local_idx[local_p]) < _UI_TEMPLATE_SAMPLE_LIMIT:
                success, encoded = cv2.imencode(".png", aligned)
                if success:
                    pages_by_local_idx[local_p].append(encoded.tobytes())

    dynamic_templates = generate_dynamic_templates(pages_by_local_idx)
    _save_ui_template_cache(cache_path, dynamic_templates)

    # auto_detect_checkboxes 함수는 BGR 형태를 요구하므로 변환해서 반환합니다.
    bgr_templates = {}
    for k, v in dynamic_templates.items():
        bgr_templates[k] = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

    return bgr_templates


def generate_ui_templates_multi(
    pdf_paths: list[str],
    page_count: int,
    rot_code: int,
    fine_angle: float,
    progress_cb=None,
) -> dict[int, np.ndarray]:
    """여러 PDF에서 템플릿을 생성하고 병합하여 더 정확한 템플릿을 만듭니다."""
    if not pdf_paths or page_count <= 0:
        return {}

    cache_path = _ui_template_cache_path(
        pdf_paths, page_count, rot_code, fine_angle, "multi"
    )
    cached_templates = _load_ui_template_cache(cache_path)
    if cached_templates is not None:
        if progress_cb:
            progress_cb(100, "캐시된 병합 템플릿 불러오기 완료")
        return {
            key: cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
            for key, value in cached_templates.items()
        }

    page_totals = []
    full_capacities = []
    partial_page_counts = []
    for fpath in pdf_paths:
        try:
            with fitz.open(fpath) as doc:
                total_pages = len(doc)
        except Exception:
            total_pages = 0
        page_totals.append(total_pages)
        full_capacities.append(total_pages // page_count)
        partial_page_counts.append(total_pages % page_count)

    # 완전한 설문을 먼저 균등 배분하고, 남는 한도에만 partial survey를 사용합니다.
    full_quotas = [0] * len(pdf_paths)
    remaining = _UI_TEMPLATE_SAMPLE_LIMIT
    while remaining > 0:
        progressed = False
        for index, capacity in enumerate(full_capacities):
            if full_quotas[index] >= capacity:
                continue
            full_quotas[index] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break

    selected_page_counts = [quota * page_count for quota in full_quotas]
    if remaining > 0:
        for index, partial_pages in enumerate(partial_page_counts):
            if partial_pages <= 0:
                continue
            selected_page_counts[index] += partial_pages
            remaining -= 1
            if remaining == 0:
                break

    all_by_local_idx = {i: [] for i in range(page_count)}
    ref_aligners: dict[int, ImageAligner] = {}

    for f_i, (fpath, sample_page_count) in enumerate(
        zip(pdf_paths, selected_page_counts)
    ):
        if sample_page_count <= 0:
            if progress_cb:
                progress_cb(
                    int((f_i + 1) / len(pdf_paths) * 100), "템플릿 병합 중..."
                )
            continue

        sample_page_count = min(page_totals[f_i], sample_page_count)
        try:
            pages = load_pdf_pages(
                fpath,
                gray=True,
                page_indices=list(range(sample_page_count)),
            )
        except Exception:
            continue

        survey_count = _survey_count(len(pages), page_count)
        for survey_idx in range(survey_count):
            for local_p in range(page_count):
                global_p = survey_idx * page_count + local_p
                if global_p >= len(pages):
                    break

                orig = apply_rotation(pages[global_p], rot_code, fine_angle)
                aligner = ref_aligners.get(local_p)
                if aligner is None:
                    aligner = ImageAligner(orig, refine_ecc=False)
                    ref_aligners[local_p] = aligner
                aligned = aligner.align(orig)
                success, encoded = cv2.imencode(".png", aligned)
                if success:
                    all_by_local_idx[local_p].append(encoded.tobytes())

        if progress_cb:
            progress_cb(int((f_i + 1) / len(pdf_paths) * 100), "템플릿 병합 중...")

    if progress_cb:
        progress_cb(100, "템플릿 병합 완료")

    dynamic_templates = generate_dynamic_templates(all_by_local_idx)
    _save_ui_template_cache(cache_path, dynamic_templates)

    bgr_templates = {}
    for k, v in dynamic_templates.items():
        bgr_templates[k] = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

    return bgr_templates


def _best_shift_by_correlation(
    template_mask: np.ndarray,
    padded_target: np.ndarray,
    max_shift: int,
    reference_pixels: int,
) -> tuple[float, int, int, int]:
    """제한된 이동 범위의 모든 겹침을 OpenCV 상관맵 한 번으로 계산합니다."""
    point_data = cv2.findNonZero(template_mask)
    if point_data is None:
        return float("-inf"), 0, 0, 0

    candidate_pixels = len(point_data)
    x, y, width, height = cv2.boundingRect(point_data)
    template_roi = np.ascontiguousarray(
        template_mask[y : y + height, x : x + width]
    )
    target_roi = np.ascontiguousarray(
        padded_target[
            y : y + height + max_shift * 2,
            x : x + width + max_shift * 2,
        ]
    )
    correlation = cv2.matchTemplate(target_roi, template_roi, cv2.TM_CCORR)
    _, max_value, _, max_location = cv2.minMaxLoc(correlation)

    # 두 마스크 값이 0 또는 255이므로 상관값을 겹친 픽셀 수로 환산할 수 있습니다.
    overlap = int(round(max_value / (255.0 * 255.0)))
    dx = max_location[0] - max_shift
    dy = max_location[1] - max_shift
    # 회전 보간으로 마스크 면적이 변하는 후보만 약하게 감점합니다.
    score = float(overlap) - abs(candidate_pixels - reference_pixels) * 0.2
    return score, overlap, dx, dy


def _align_template_mask_by_coverage(
    template_mask: np.ndarray,
    target_mask: np.ndarray,
    max_angle: float = 0.6,
    max_shift: int = 8,
) -> np.ndarray:
    """템플릿 선이 대상의 어두운 픽셀을 가장 많이 덮도록 미세 정합합니다.

    체크박스 생성용 페이지 정합과는 완전히 분리된 후처리입니다. 이미 ORB/ECC로
    정합된 페이지의 잔여 오차만 보정하므로 탐색 범위를 작게 제한합니다.
    """
    h, w = target_mask.shape[:2]
    if template_mask.shape[:2] != (h, w):
        template_mask = cv2.resize(
            template_mask, (w, h), interpolation=cv2.INTER_NEAREST
        )

    template_pixels = cv2.countNonZero(template_mask)
    target_pixels = cv2.countNonZero(target_mask)
    if template_pixels < 32 or target_pixels < 32:
        return template_mask

    identity_overlap = cv2.countNonZero(
        cv2.bitwise_and(template_mask, target_mask)
    )
    identity_score = float(identity_overlap)
    # 1~2px 확장은 뒤에서 적용되므로 97% 이상 맞으면 추가 탐색의 이득이 없습니다.
    if identity_overlap >= template_pixels * 0.97:
        return template_mask

    # 긴 변을 최대 900px로 줄여 각도와 대략적인 이동량을 빠르게 찾습니다.
    search_scale = min(0.5, 900.0 / max(h, w))
    search_w = max(1, round(w * search_scale))
    search_h = max(1, round(h * search_scale))
    # INTER_AREA로 축소한 뒤 낮은 임계값으로 다시 이진화해 가는 선의 소실을 줄입니다.
    small_template = cv2.resize(
        template_mask, (search_w, search_h), interpolation=cv2.INTER_AREA
    )
    small_target = cv2.resize(
        target_mask, (search_w, search_h), interpolation=cv2.INTER_AREA
    )
    _, small_template = cv2.threshold(
        small_template, 32, 255, cv2.THRESH_BINARY
    )
    _, small_target = cv2.threshold(small_target, 32, 255, cv2.THRESH_BINARY)
    small_template_pixels = cv2.countNonZero(small_template)
    if small_template_pixels == 0:
        return template_mask

    angle_step = 0.1
    angle_count = max(0, round(max_angle / angle_step))
    angles = [step * angle_step for step in range(-angle_count, angle_count + 1)]
    small_shift = max(0, int(np.ceil(max_shift * search_scale)))
    small_padded_target = cv2.copyMakeBorder(
        small_target,
        small_shift,
        small_shift,
        small_shift,
        small_shift,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    center = (search_w / 2.0, search_h / 2.0)

    best_coarse_key = (float("-inf"), 0, float("-inf"))
    best_coarse = (0.0, 0, 0)

    for angle in angles:
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            small_template,
            matrix,
            (search_w, search_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        score, overlap, dx, dy = _best_shift_by_correlation(
            rotated, small_padded_target, small_shift, small_template_pixels
        )
        motion = abs(angle) + abs(dx) + abs(dy)
        key = (score, overlap, -motion)
        if key > best_coarse_key:
            best_coarse_key = key
            best_coarse = (angle, dx, dy)

    coarse_angle, _, _ = best_coarse
    fine_angles = {
        round(max(-max_angle, coarse_angle - 0.05), 2),
        round(coarse_angle, 2),
        round(min(max_angle, coarse_angle + 0.05), 2),
    }

    # 원본 크기의 무변환 점수를 기준으로 두어 정합 결과가 더 나빠지는 것을 방지합니다.
    best_key = (identity_score, identity_overlap, 0.0)
    best_transform = (0.0, 0, 0)
    full_center = (w / 2.0, h / 2.0)
    full_padded_target = cv2.copyMakeBorder(
        target_mask,
        max_shift,
        max_shift,
        max_shift,
        max_shift,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    for angle in sorted(fine_angles):
        matrix = cv2.getRotationMatrix2D(full_center, angle, 1.0)
        rotated = cv2.warpAffine(
            template_mask,
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        score, overlap, dx, dy = _best_shift_by_correlation(
            rotated, full_padded_target, max_shift, template_pixels
        )
        motion = abs(angle) + abs(dx) + abs(dy)
        key = (score, overlap, -motion)
        if key > best_key:
            best_key = key
            best_transform = (angle, dx, dy)

    best_angle, best_dx, best_dy = best_transform
    min_overlap_gain = max(32, round(template_pixels * 0.001))
    is_identity = best_angle == 0.0 and best_dx == 0 and best_dy == 0
    has_meaningful_gain = best_key[1] >= identity_overlap + min_overlap_gain
    if is_identity or not has_meaningful_gain:
        return template_mask

    matrix = cv2.getRotationMatrix2D(full_center, best_angle, 1.0)
    matrix[0, 2] += best_dx
    matrix[1, 2] += best_dy
    return cv2.warpAffine(
        template_mask,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def extract_pure_ink_mask(
    target_gray: np.ndarray,
    template_gray: np.ndarray,
    template_dilate_pct: float = 0.3,
    prepared_template_mask: np.ndarray | None = None,
) -> np.ndarray:
    """템플릿을 대상에 미세 정합해 제거하고 순수 사용자 잉크만 추출합니다."""
    if target_gray.ndim == 3:
        target_gray = cv2.cvtColor(target_gray, cv2.COLOR_BGR2GRAY)

    # 1. 대상과 템플릿의 어두운 픽셀 마스크 생성
    _, target_mask = cv2.threshold(target_gray, 200, 255, cv2.THRESH_BINARY_INV)
    if prepared_template_mask is None:
        if template_gray.ndim == 3:
            template_gray = cv2.cvtColor(template_gray, cv2.COLOR_BGR2GRAY)
        _, template_mask = cv2.threshold(
            template_gray, 200, 255, cv2.THRESH_BINARY_INV
        )
    else:
        template_mask = prepared_template_mask

    # 2. 템플릿 마스크만 미세 회전·이동하여 대상의 인쇄선을 최대한 덮음
    template_mask = _align_template_mask_by_coverage(template_mask, target_mask)

    # 3. 남은 미세 정합 오차만큼 템플릿 마스크 확장
    dilate_px = max(0, round(template_dilate_pct * 5))
    if dilate_px > 0:
        template_mask = cv2.dilate(
            template_mask,
            np.ones((3, 3), np.uint8),
            iterations=dilate_px,
        )

    # 4. 대상 이미지에서 템플릿 영역을 지움
    cleaned = target_gray.copy()
    cleaned[template_mask > 0] = 255

    # 5. 남은 어두운 픽셀이 순수 잉크
    blur = cv2.GaussianBlur(cleaned, (3, 3), 0)
    _, pure_ink_mask = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)

    # 6. 모폴로지 노이즈 제거
    pure_ink_mask = cv2.erode(pure_ink_mask, np.ones((2, 2), np.uint8), iterations=1)
    pure_ink_mask = cv2.dilate(pure_ink_mask, np.ones((3, 3), np.uint8), iterations=1)

    return pure_ink_mask


def extract_ink_info_from_mask(pure_ink_mask: np.ndarray, box: Box) -> tuple[int, int]:
    h_img, w_img = pure_ink_mask.shape[:2]

    x1 = max(0, box.x)
    y1 = max(0, box.y)
    x2 = min(w_img, box.x + box.w)
    y2 = min(h_img, box.y + box.h)

    if x2 <= x1 or y2 <= y1:
        return 0, 0

    roi_target = pure_ink_mask[y1:y2, x1:x2]
    ink_pixels = cv2.countNonZero(roi_target)
    area = (x2 - x1) * (y2 - y1)

    return ink_pixels, area


def _is_checkbox_like(box: Box, image_shape: tuple[int, ...]) -> bool:
    """작은 정사각형 체크박스만 내부 잉크 직접 판독 대상으로 분류합니다."""
    if box.w < 8 or box.h < 8:
        return False

    short_side = min(box.w, box.h)
    long_side = max(box.w, box.h)
    if long_side / short_side > 1.35:
        return False

    image_h, image_w = image_shape[:2]
    max_side = max(48, round(min(image_h, image_w) * 0.045))
    return long_side <= max_side


def _refine_checkbox_box(
    target_gray: np.ndarray,
    box: Box,
) -> tuple[Box, float]:
    """예상 위치 주변의 네모 테두리를 찾아 체크박스 좌표를 국소 보정합니다."""
    if target_gray.ndim == 3:
        target_gray = cv2.cvtColor(target_gray, cv2.COLOR_BGR2GRAY)

    image_h, image_w = target_gray.shape[:2]
    if box.w <= 0 or box.h <= 0 or box.w > image_w or box.h > image_h:
        return copy.copy(box), 0.0

    short_side = min(box.w, box.h)
    search_radius = max(2, min(10, round(short_side * 0.4)))
    min_candidate_x = max(0, box.x - search_radius)
    min_candidate_y = max(0, box.y - search_radius)
    max_candidate_x = min(image_w - box.w, box.x + search_radius)
    max_candidate_y = min(image_h - box.h, box.y + search_radius)
    if max_candidate_x < min_candidate_x or max_candidate_y < min_candidate_y:
        return copy.copy(box), 0.0

    border_width = max(1, round(short_side * 0.1))
    edge_tolerance = max(2, round(short_side * 0.2))
    span_padding = max(1, round(short_side * 0.1))
    context_padding = edge_tolerance + span_padding + border_width
    context_x1 = max(0, min_candidate_x - context_padding)
    context_y1 = max(0, min_candidate_y - context_padding)
    context_x2 = min(
        image_w, max_candidate_x + box.w + context_padding
    )
    context_y2 = min(
        image_h, max_candidate_y + box.h + context_padding
    )
    context = target_gray[context_y1:context_y2, context_x1:context_x2]

    # auto_detect_checkboxes가 반환하는 좌표는 양식에 따라 외곽선 또는
    # 사각형의 흰 내부 영역일 수 있습니다. 각 후보 가장자리 주변에서 실제
    # 선을 독립적으로 찾으면 두 좌표 표현을 모두 처리하면서, 글자 획 두 개의
    # 교차점은 닫힌 사각형으로 인정하지 않을 수 있습니다.
    darkness = np.clip(
        (230.0 - context.astype(np.float32)) / 80.0,
        0.0,
        1.0,
    )
    integral = cv2.integral(darkness, sdepth=cv2.CV_64F)
    context_h, context_w = context.shape[:2]
    candidate_xs = np.arange(min_candidate_x, max_candidate_x + 1, dtype=np.int32)
    candidate_ys = np.arange(min_candidate_y, max_candidate_y + 1, dtype=np.int32)
    offsets = np.arange(-edge_tolerance, edge_tolerance + 1, dtype=np.int32)

    def rectangle_means(
        x_start: np.ndarray,
        y_start: np.ndarray,
        x_end: np.ndarray,
        y_end: np.ndarray,
    ) -> np.ndarray:
        local_x1 = np.clip(x_start - context_x1, 0, context_w)
        local_y1 = np.clip(y_start - context_y1, 0, context_h)
        local_x2 = np.clip(x_end - context_x1, 0, context_w)
        local_y2 = np.clip(y_end - context_y1, 0, context_h)
        local_x1, local_y1, local_x2, local_y2 = np.broadcast_arrays(
            local_x1, local_y1, local_x2, local_y2
        )
        totals = (
            integral[local_y2, local_x2]
            - integral[local_y1, local_x2]
            - integral[local_y2, local_x1]
            + integral[local_y1, local_x1]
        )
        areas = (local_x2 - local_x1) * (local_y2 - local_y1)
        means = np.zeros_like(totals, dtype=np.float64)
        np.divide(totals, areas, out=means, where=areas > 0)
        return means

    horizontal_x1 = candidate_xs[None, :, None] - span_padding
    horizontal_x2 = candidate_xs[None, :, None] + box.w + span_padding
    top_y1 = candidate_ys[:, None, None] + offsets[None, None, :]
    top_y2 = top_y1 + border_width
    bottom_y2 = candidate_ys[:, None, None] + box.h + offsets[None, None, :]
    bottom_y1 = bottom_y2 - border_width
    top = rectangle_means(horizontal_x1, top_y1, horizontal_x2, top_y2).max(
        axis=2
    )
    bottom = rectangle_means(
        horizontal_x1, bottom_y1, horizontal_x2, bottom_y2
    ).max(axis=2)

    vertical_y1 = candidate_ys[:, None, None] - span_padding
    vertical_y2 = candidate_ys[:, None, None] + box.h + span_padding
    left_x1 = candidate_xs[None, :, None] + offsets[None, None, :]
    left_x2 = left_x1 + border_width
    right_x2 = candidate_xs[None, :, None] + box.w + offsets[None, None, :]
    right_x1 = right_x2 - border_width
    left = rectangle_means(left_x1, vertical_y1, left_x2, vertical_y2).max(axis=2)
    right = rectangle_means(right_x1, vertical_y1, right_x2, vertical_y2).max(
        axis=2
    )

    side_scores = np.stack((top, bottom, left, right))
    scores = side_scores.min(axis=0) * 0.85 + side_scores.mean(axis=0) * 0.15
    motion_penalty = (
        np.abs(candidate_xs[None, :] - box.x)
        + np.abs(candidate_ys[:, None] - box.y)
    ) * 0.003
    adjusted_scores = scores - motion_penalty
    best_flat_index = int(np.argmax(adjusted_scores))
    best_y_index, best_x_index = np.unravel_index(
        best_flat_index, adjusted_scores.shape
    )
    confidence = float(scores[best_y_index, best_x_index])
    best_x = int(candidate_xs[best_x_index])
    best_y = int(candidate_ys[best_y_index])

    refined = copy.copy(box)
    if confidence >= _CHECKBOX_MIN_BORDER_CONFIDENCE:
        refined.x = best_x
        refined.y = best_y
    return refined, confidence


def extract_checkbox_ink_info(
    target_gray: np.ndarray,
    box: Box,
) -> _CheckboxInkInfo:
    """체크박스 테두리를 피한 내부의 실제 어두운 연결 성분을 측정합니다.

    동적 템플릿에 반복 체크가 섞여도 원본 페이지의 빈 내부만 직접 보기 때문에
    체크 표시가 템플릿과 함께 지워지지 않습니다.
    """
    if target_gray.ndim == 3:
        target_gray = cv2.cvtColor(target_gray, cv2.COLOR_BGR2GRAY)

    refined, confidence = _refine_checkbox_box(target_gray, box)
    image_h, image_w = target_gray.shape[:2]
    short_side = min(refined.w, refined.h)
    border_width = max(1, round(short_side * 0.1))
    margin = max(border_width + 1, round(short_side * 0.2))

    x1 = max(0, refined.x + margin)
    y1 = max(0, refined.y + margin)
    x2 = min(image_w, refined.x + refined.w - margin)
    y2 = min(image_h, refined.y + refined.h - margin)
    if x2 <= x1 or y2 <= y1:
        return _CheckboxInkInfo(
            0,
            0,
            refined,
            confidence,
            (x1, y1, x1, y1),
            np.zeros((0, 0), np.uint8),
        )

    interior = target_gray[y1:y2, x1:x2]
    paper_level = float(np.percentile(interior, 90))
    ink_threshold = int(np.clip(paper_level - 18.0, 170.0, 225.0))
    raw_mask = (interior < ink_threshold).astype(np.uint8)

    component_mask = np.zeros_like(raw_mask)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        raw_mask, connectivity=8
    )
    min_component_area = max(2, round(raw_mask.size * 0.01))
    for component_idx in range(1, component_count):
        if stats[component_idx, cv2.CC_STAT_AREA] >= min_component_area:
            component_mask[labels == component_idx] = 255

    return _CheckboxInkInfo(
        cv2.countNonZero(component_mask),
        int(raw_mask.size),
        refined,
        confidence,
        (x1, y1, x2, y2),
        component_mask,
    )


def _extract_checkbox_halo_info(
    pure_ink_mask: np.ndarray,
    box: Box,
    target_gray: np.ndarray | None = None,
    box_is_refined: bool = False,
) -> _CheckboxHaloInfo:
    """박스 테두리에 실제로 연결된 외부 체크 획만 추출합니다.

    단순히 박스 주변의 모든 잉크를 합산하면 인접 글자와 미세하게 어긋난
    빈 사각 테두리까지 체크로 세게 됩니다. 원본 페이지가 있으면 박스 위치를
    다시 확인한 뒤, 테두리와 연결된 성분에서 사각 프레임과 내부를 제거합니다.
    이렇게 하면 중앙값 템플릿에 반복 체크가 포함된 경우에도 외부 획을 복구할
    수 있습니다.
    """
    working_box = copy.copy(box)
    if target_gray is not None and not box_is_refined:
        if target_gray.ndim == 3:
            target_gray = cv2.cvtColor(target_gray, cv2.COLOR_BGR2GRAY)
        refined_box, confidence = _refine_checkbox_box(target_gray, working_box)
        if confidence >= _CHECKBOX_MIN_BORDER_CONFIDENCE:
            working_box = refined_box

    image_h, image_w = pure_ink_mask.shape[:2]
    short_side = min(working_box.w, working_box.h)
    padding = max(4, round(short_side * 0.7))
    x1 = max(0, working_box.x - padding)
    y1 = max(0, working_box.y - padding)
    x2 = min(image_w, working_box.x + working_box.w + padding)
    y2 = min(image_h, working_box.y + working_box.h + padding)
    if x2 <= x1 or y2 <= y1:
        return _CheckboxHaloInfo(
            0,
            0,
            working_box,
            (x1, y1, x1, y1),
            np.zeros((0, 0), np.uint8),
        )

    candidate = (pure_ink_mask[y1:y2, x1:x2] > 0).astype(np.uint8) * 255
    if target_gray is not None:
        raw_roi = target_gray[y1:y2, x1:x2]
        paper_level = float(np.percentile(raw_roi, 90))
        ink_threshold = int(np.clip(paper_level - 22.0, 160.0, 225.0))
        raw_dark = (raw_roi < ink_threshold).astype(np.uint8) * 255
        candidate = cv2.bitwise_or(candidate, raw_dark)

    local_x = working_box.x - x1
    local_y = working_box.y - y1
    local_x2 = local_x + working_box.w
    local_y2 = local_y + working_box.h

    # 체크 획은 보통 사각 테두리를 가로질러 밖으로 나갑니다. 1px 연결
    # 보강 후 테두리 링과 닿지 않는 인접 글자·먼지는 후보에서 제외합니다.
    connected = cv2.dilate(candidate, np.ones((3, 3), np.uint8), iterations=1)
    component_count, labels, _, _ = cv2.connectedComponentsWithStats(
        (connected > 0).astype(np.uint8), connectivity=8
    )
    anchor_width = max(1, round(short_side * 0.1))
    anchor = np.zeros_like(candidate)
    cv2.rectangle(
        anchor,
        (max(0, local_x), max(0, local_y)),
        (min(anchor.shape[1] - 1, local_x2 - 1), min(anchor.shape[0] - 1, local_y2 - 1)),
        255,
        thickness=anchor_width * 2 + 1,
    )
    anchored_labels = set(int(value) for value in np.unique(labels[anchor > 0]))
    anchored_labels.discard(0)
    anchored = np.zeros_like(candidate)
    for component_idx in anchored_labels:
        anchored[(labels == component_idx) & (candidate > 0)] = 255

    # 실제 박스 테두리와 그 정합 잔상은 넉넉한 띠로 지우고, halo가 내부
    # 직접 점수와 중복되지 않도록 박스 안쪽도 모두 제외합니다.
    border_band = max(2, round(short_side * 0.2))
    outer_x1 = max(0, local_x - border_band)
    outer_y1 = max(0, local_y - border_band)
    outer_x2 = min(anchored.shape[1], local_x2 + border_band)
    outer_y2 = min(anchored.shape[0], local_y2 + border_band)
    border_mask = np.zeros_like(anchored)
    border_mask[outer_y1:outer_y2, outer_x1:outer_x2] = 255
    inner_x1 = min(anchored.shape[1], max(0, local_x + border_band))
    inner_y1 = min(anchored.shape[0], max(0, local_y + border_band))
    inner_x2 = min(anchored.shape[1], max(0, local_x2 - border_band))
    inner_y2 = min(anchored.shape[0], max(0, local_y2 - border_band))
    if inner_x2 > inner_x1 and inner_y2 > inner_y1:
        border_mask[inner_y1:inner_y2, inner_x1:inner_x2] = 0
    anchored[border_mask > 0] = 0
    anchored[
        max(0, local_y) : min(anchored.shape[0], local_y2),
        max(0, local_x) : min(anchored.shape[1], local_x2),
    ] = 0

    filtered = np.zeros_like(anchored)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (anchored > 0).astype(np.uint8), connectivity=8
    )
    min_component_area = max(4, round(short_side * 0.15))
    min_component_span = max(4, round(short_side * 0.2))
    for component_idx in range(1, component_count):
        component_area = stats[component_idx, cv2.CC_STAT_AREA]
        component_width = stats[component_idx, cv2.CC_STAT_WIDTH]
        component_height = stats[component_idx, cv2.CC_STAT_HEIGHT]
        # 정합 잔상은 대개 한 변을 따라 2~3px 폭으로 길게 남습니다. 실제
        # 체크 꼬리는 대각선/곡선이라 가로·세로 양쪽으로 모두 퍼집니다.
        if (
            component_area >= min_component_area
            and component_width >= min_component_span
            and component_height >= min_component_span
        ):
            filtered[labels == component_idx] = 255

    return _CheckboxHaloInfo(
        cv2.countNonZero(filtered),
        int(filtered.size),
        working_box,
        (x1, y1, x2, y2),
        filtered,
    )


def extract_checkbox_halo_ink_info(
    pure_ink_mask: np.ndarray,
    box: Box,
    target_gray: np.ndarray | None = None,
) -> tuple[int, int]:
    info = _extract_checkbox_halo_info(pure_ink_mask, box, target_gray)
    return info.ink_pixels, info.area


def _resolve_checkbox_halo_ownership(
    pure_ink_masks: dict[int, np.ndarray],
    survey_gray_pages: dict[int, np.ndarray],
    candidates: list[tuple[_CheckboxInkInfo, bool, _CheckboxHaloInfo]],
) -> None:
    """Assign one connected pen stroke to its closest supported checkbox.

    A long check drawn in one box can graze the border of a nearby blank box.
    Both local halo windows then see the same connected stroke.  Resolve that
    ambiguity page-wide: prefer a box with clear interior ink, otherwise keep
    the box containing the larger share of the external stroke.  Only nearby
    boxes compete, so unrelated marks connected through form lines cannot
    suppress one another across the page.
    """
    by_page: dict[
        int, list[tuple[_CheckboxInkInfo, bool, _CheckboxHaloInfo]]
    ] = {}
    for candidate in candidates:
        direct_info, _direct_is_checked, halo_info = candidate
        if halo_info.ink_pixels <= 0 or halo_info.ink_mask.size == 0:
            continue
        by_page.setdefault(direct_info.box.page_idx, []).append(candidate)

    for page_idx, page_candidates in by_page.items():
        target_gray = survey_gray_pages.get(page_idx)
        if target_gray is None:
            continue
        if target_gray.ndim == 3:
            target_gray = cv2.cvtColor(target_gray, cv2.COLOR_BGR2GRAY)

        paper_level = float(np.percentile(target_gray, 90))
        ink_threshold = int(np.clip(paper_level - 22.0, 160.0, 225.0))
        connected_source = (target_gray < ink_threshold).astype(np.uint8) * 255
        pure_ink = pure_ink_masks.get(page_idx)
        if pure_ink is not None:
            connected_source = cv2.bitwise_or(
                connected_source,
                (pure_ink > 0).astype(np.uint8) * 255,
            )
        connected_source = cv2.dilate(
            connected_source, np.ones((3, 3), np.uint8), iterations=1
        )
        _, labels, _, _ = cv2.connectedComponentsWithStats(
            (connected_source > 0).astype(np.uint8), connectivity=8
        )

        component_members: dict[int, list[tuple[int, int]]] = {}
        for candidate_idx, (_direct_info, _direct_is_checked, halo_info) in enumerate(
            page_candidates
        ):
            x1, y1, x2, y2 = halo_info.mask_bounds
            label_roi = labels[y1:y2, x1:x2]
            if label_roi.shape != halo_info.ink_mask.shape:
                continue
            values, counts = np.unique(
                label_roi[halo_info.ink_mask > 0], return_counts=True
            )
            for value, count in zip(values.tolist(), counts.tolist()):
                component_label = int(value)
                if component_label == 0:
                    continue
                component_members.setdefault(component_label, []).append(
                    (candidate_idx, int(count))
                )

        labels_to_remove: dict[int, set[int]] = {}
        for component_label, members in component_members.items():
            if len(members) <= 1:
                continue

            pending = {candidate_idx for candidate_idx, _count in members}
            counts_by_candidate = dict(members)
            while pending:
                seed = pending.pop()
                cluster = {seed}
                frontier = [seed]
                while frontier:
                    current_idx = frontier.pop()
                    current_box = page_candidates[current_idx][0].box
                    current_center = (
                        current_box.x + current_box.w / 2,
                        current_box.y + current_box.h / 2,
                    )
                    for other_idx in list(pending):
                        other_box = page_candidates[other_idx][0].box
                        other_center = (
                            other_box.x + other_box.w / 2,
                            other_box.y + other_box.h / 2,
                        )
                        distance = float(
                            np.hypot(
                                current_center[0] - other_center[0],
                                current_center[1] - other_center[1],
                            )
                        )
                        nearby_limit = max(
                            48.0,
                            max(
                                min(current_box.w, current_box.h),
                                min(other_box.w, other_box.h),
                            )
                            * 4.0,
                        )
                        if distance <= nearby_limit:
                            pending.remove(other_idx)
                            cluster.add(other_idx)
                            frontier.append(other_idx)

                physical_keys = {
                    (
                        page_candidates[index][0].box.x,
                        page_candidates[index][0].box.y,
                        page_candidates[index][0].box.w,
                        page_candidates[index][0].box.h,
                    )
                    for index in cluster
                }
                if len(physical_keys) <= 1:
                    continue

                def ownership_key(index: int) -> tuple[bool, int, int, int, int]:
                    direct_info, direct_is_checked, halo_info = page_candidates[index]
                    strong_direct = direct_is_checked and direct_info.ink_pixels >= max(
                        8, round(direct_info.area * 0.05)
                    )
                    return (
                        strong_direct,
                        counts_by_candidate[index],
                        direct_info.ink_pixels,
                        halo_info.ink_pixels,
                        -index,
                    )

                winner_idx = max(cluster, key=ownership_key)
                winner_box = page_candidates[winner_idx][0].box
                winner_key = (
                    winner_box.x,
                    winner_box.y,
                    winner_box.w,
                    winner_box.h,
                )
                for loser_idx in cluster:
                    loser_box = page_candidates[loser_idx][0].box
                    loser_key = (
                        loser_box.x,
                        loser_box.y,
                        loser_box.w,
                        loser_box.h,
                    )
                    if loser_key != winner_key:
                        labels_to_remove.setdefault(loser_idx, set()).add(
                            component_label
                        )

        for candidate_idx, component_labels in labels_to_remove.items():
            halo_info = page_candidates[candidate_idx][2]
            x1, y1, x2, y2 = halo_info.mask_bounds
            label_roi = labels[y1:y2, x1:x2]
            remove = np.isin(label_roi, list(component_labels))
            halo_info.ink_mask[remove] = 0
            halo_info.ink_pixels = cv2.countNonZero(halo_info.ink_mask)


# ==========================================
# 2. 평가 및 시각화 모듈
# ==========================================


def evaluate_marks(
    inks: list[int], areas: list[int], is_contiguous: bool, strict: bool = False
) -> list[bool]:
    if not inks:
        return []

    if len(inks) > 1:
        min_ink = min(inks)
        net_inks = [max(0, ink - min_ink) for ink in inks]
        max_net = max(net_inks)

        if strict:
            # 중복 허용 모드: 꼬리 침범 방지를 위해 임계값 상향
            abs_thresh = 20
            rel_thresh = 0.45 if is_contiguous else 0.30
        else:
            abs_thresh = 15 if is_contiguous else 5
            rel_thresh = 0.3 if is_contiguous else 0.15

        return [
            (net > abs_thresh) and (net >= max_net * rel_thresh) for net in net_inks
        ]

    ink, area = inks[0], areas[0]
    is_ticked = (ink > 10) or (area > 0 and (ink / area) >= 0.01)
    return [is_ticked]


def evaluate_checkbox_marks(
    inks: list[int], areas: list[int], strict: bool = False
) -> list[bool]:
    """인쇄선이 제외된 체크박스 내부 점수를 평가합니다."""
    if not inks:
        return []

    max_ink = max(inks)
    # 복수응답은 후보를 사후에 하나로 줄이지 않으므로 오히려 더 엄격해야 합니다.
    relative_threshold = 0.25 if strict else 0.15
    results = []
    for ink, area in zip(inks, areas):
        absolute_threshold = max(4, round(area * 0.025)) if area > 0 else 4
        results.append(
            ink >= absolute_threshold
            and (max_ink == 0 or ink >= max_ink * relative_threshold)
        )
    return results


def evaluate_checkbox_halo_marks(
    inks: list[int],
    areas: list[int],
    is_contiguous: bool,
    strict: bool = False,
) -> list[bool]:
    """외부 체크 획을 평가하되 얇은 사각 테두리 잔상은 제외합니다."""
    relative_results = evaluate_marks(inks, areas, is_contiguous, strict=strict)
    return [
        is_checked
        and ink >= max(12, round(np.sqrt(area) * 0.59))
        for ink, area, is_checked in zip(inks, areas, relative_results)
    ]


def enforce_single_choice(
    check_results: list[bool], inks: list[int], areas: list[int]
) -> list[bool]:
    if sum(check_results) <= 1:
        return check_results

    if not inks:
        return check_results

    min_ink = min(inks)
    net_inks = [max(0, ink - min_ink) for ink in inks]
    true_indices = [i for i, is_ticked in enumerate(check_results) if is_ticked]
    if not true_indices:
        return check_results

    best_idx = max(true_indices, key=lambda i: (net_inks[i], inks[i], areas[i]))
    return [i == best_idx for i in range(len(check_results))]


def _label_number(total: int, index: int, reverse: bool) -> int:
    if total > 1:
        return total - index + 1 if reverse else index
    return 1


def _survey_count(total_pages: int, page_count: int) -> int:
    if page_count <= 0:
        return 0
    count = total_pages // page_count
    if total_pages % page_count != 0:
        count += 1
    return count


def _file_survey_counts(file_paths: list[str], page_count: int) -> list[int]:
    """Read cheap PDF metadata up front so batch progress can be survey-based."""
    counts: list[int] = []
    for fpath in file_paths:
        doc = None
        try:
            doc = fitz.open(fpath)
            counts.append(_survey_count(len(doc), page_count))
        except Exception:
            counts.append(0)
        finally:
            if doc is not None:
                doc.close()
    return counts


def _analysis_survey_work(
    start: int, end: int, reusable_samples: int
) -> float:
    """Weight cached-sample surveys less than surveys that must be rendered."""
    start = max(0, int(start))
    end = max(start, int(end))
    reusable_samples = max(0, int(reusable_samples))
    reused = max(0, min(end, reusable_samples) - min(start, reusable_samples))
    rendered = (end - start) - reused
    return (
        reused * _ANALYSIS_REUSED_SURVEY_WORK
        + rendered * _ANALYSIS_RENDERED_SURVEY_WORK
    )


def _select_working_boxes(
    field, z_sorted_boxes: list[Box], all_boxes: list[Box]
) -> list[Box]:
    if field.is_comment or field.allow_duplicates:
        return [copy.copy(b) for b in z_sorted_boxes]

    return expand_isolated_boxes(z_sorted_boxes, all_boxes, scale_factor=2.0)


def _prepare_field_plans(
    config: TemplatePreset,
) -> list[tuple[Field, list[Box], list[Box], bool]]:
    """설문마다 동일한 박스 정렬·확장 결과를 분석 시작 전에 한 번만 계산합니다."""
    all_boxes = [box for field in config.fields for box in field.boxes]
    plans = []
    for field in config.fields:
        sorted_boxes = sort_boxes_z_pattern(field.boxes)
        working_boxes = _select_working_boxes(field, sorted_boxes, all_boxes)
        plans.append(
            (field, sorted_boxes, working_boxes, is_contiguous_group(sorted_boxes))
        )
    return plans


def _group_checkbox_rows(boxes: list[Box], tolerance: float) -> list[list[Box]]:
    rows: list[list[Box]] = []
    row_centers: list[float] = []
    for box in sorted(boxes, key=lambda item: (item.y + item.h / 2, item.x)):
        center_y = box.y + box.h / 2
        if not rows or abs(center_y - row_centers[-1]) > tolerance:
            rows.append([box])
            row_centers.append(center_y)
            continue
        rows[-1].append(box)
        row_centers[-1] = float(
            np.mean([item.y + item.h / 2 for item in rows[-1]])
        )

    for row in rows:
        row.sort(key=lambda item: item.x + item.w / 2)
    return rows


def _remap_checkbox_layout(
    config: TemplatePreset,
    templates: dict[int, np.ndarray],
) -> TemplatePreset:
    """현재 파일 템플릿에서 체크박스를 다시 찾아 오래된 프리셋 좌표를 보정합니다.

    각 행의 체크박스 개수와 좌우 순서가 모두 일치할 때만 해당 행을 교체합니다.
    탐지가 불완전한 행은 원래 좌표를 유지해 순서 밀림을 방지합니다.
    """
    adjusted = copy.deepcopy(config)

    for page_idx, template in templates.items():
        expected = [
            box
            for field in adjusted.fields
            if not field.is_comment
            for box in field.boxes
            if box.page_idx == page_idx and _is_checkbox_like(box, template.shape)
        ]
        if not expected:
            continue

        median_w = float(np.median([box.w for box in expected]))
        median_h = float(np.median([box.h for box in expected]))
        detection_image = (
            cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
            if template.ndim == 2
            else template
        )
        detected_raw = auto_detect_checkboxes(
            detection_image,
            min_w=max(8, round(median_w * 0.55)),
            max_w=max(9, round(median_w * 1.8)),
            min_h=max(8, round(median_h * 0.55)),
            max_h=max(9, round(median_h * 1.8)),
        )
        detected = [
            Box(page_idx=page_idx, x=x, y=y, w=w, h=h)
            for x, y, w, h in detected_raw
            if 0.55 <= w / max(1, h) <= 1.8
        ]
        if not detected:
            continue

        row_tolerance = max(4.0, median_h * 0.6)
        expected_rows = _group_checkbox_rows(expected, row_tolerance)
        detected_rows = _group_checkbox_rows(detected, row_tolerance)
        max_y_distance = max(12.0, median_h * 1.5)
        max_x_distance = max(median_w * 10.0, template.shape[1] * 0.15)

        # 행을 위에서부터 하나씩 소비하면 앞 행 탐지가 누락됐을 때 그 다음
        # 문항의 행을 빼앗을 수 있습니다. 전체 행을 순서 보존 1:1로 맞추고,
        # 가능한 대응 수가 같을 때 총 이동량이 가장 작은 조합을 선택합니다.
        pair_costs: dict[tuple[int, int], float] = {}
        for expected_idx, expected_row in enumerate(expected_rows):
            expected_center_y = float(
                np.mean([box.y + box.h / 2 for box in expected_row])
            )
            for detected_idx, detected_row in enumerate(detected_rows):
                if len(detected_row) != len(expected_row):
                    continue
                detected_center_y = float(
                    np.mean([box.y + box.h / 2 for box in detected_row])
                )
                y_distance = abs(detected_center_y - expected_center_y)
                if y_distance > max_y_distance:
                    continue
                x_distances = [
                    abs(source.x + source.w / 2 - target.x - target.w / 2)
                    for source, target in zip(expected_row, detected_row)
                ]
                if any(distance > max_x_distance for distance in x_distances):
                    continue
                pair_costs[(expected_idx, detected_idx)] = (
                    y_distance / max_y_distance
                    + float(np.mean(x_distances)) / max_x_distance
                )

        # DP 값: (매칭 수, 누적 비용, [(expected_idx, detected_idx), ...])
        row_count = len(expected_rows)
        detected_count = len(detected_rows)
        states: list[list[tuple[int, float, list[tuple[int, int]]] | None]] = [
            [None] * (detected_count + 1) for _ in range(row_count + 1)
        ]
        states[0][0] = (0, 0.0, [])

        def update_state(
            expected_pos: int,
            detected_pos: int,
            candidate: tuple[int, float, list[tuple[int, int]]],
        ) -> None:
            current = states[expected_pos][detected_pos]
            if current is None or candidate[0] > current[0] or (
                candidate[0] == current[0] and candidate[1] < current[1]
            ):
                states[expected_pos][detected_pos] = candidate

        for expected_pos in range(row_count + 1):
            for detected_pos in range(detected_count + 1):
                state = states[expected_pos][detected_pos]
                if state is None:
                    continue
                matched_count, total_cost, pairs = state
                if expected_pos < row_count:
                    update_state(expected_pos + 1, detected_pos, state)
                if detected_pos < detected_count:
                    update_state(expected_pos, detected_pos + 1, state)
                pair_cost = pair_costs.get((expected_pos, detected_pos))
                if (
                    pair_cost is not None
                    and expected_pos < row_count
                    and detected_pos < detected_count
                ):
                    update_state(
                        expected_pos + 1,
                        detected_pos + 1,
                        (
                            matched_count + 1,
                            total_cost + pair_cost,
                            pairs + [(expected_pos, detected_pos)],
                        ),
                    )

        best_state = states[row_count][detected_count]
        matched_pairs = best_state[2] if best_state is not None else []
        for expected_idx, detected_idx in matched_pairs:
            for source, target in zip(
                expected_rows[expected_idx], detected_rows[detected_idx]
            ):
                source.x = target.x
                source.y = target.y
                source.w = target.w
                source.h = target.h

    return adjusted


def _checkbox_layout_is_trustworthy(
    config: TemplatePreset,
    templates: dict[int, np.ndarray],
) -> bool:
    """Verify that every configured checkbox has a nearby detected frame."""
    saw_checkbox = False
    page_indices = {
        box.page_idx
        for field in config.fields
        if not field.is_comment
        for box in field.boxes
    }
    for page_idx in page_indices:
        template = templates.get(page_idx)
        if template is None:
            return False
        expected = [
            box
            for field in config.fields
            if not field.is_comment
            for box in field.boxes
            if box.page_idx == page_idx and _is_checkbox_like(box, template.shape)
        ]
        if not expected:
            continue
        saw_checkbox = True

        median_w = float(np.median([box.w for box in expected]))
        median_h = float(np.median([box.h for box in expected]))
        detection_image = (
            cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
            if template.ndim == 2
            else template
        )
        detected = [
            Box(page_idx=page_idx, x=x, y=y, w=w, h=h)
            for x, y, w, h in auto_detect_checkboxes(
                detection_image,
                min_w=max(8, round(median_w * 0.55)),
                max_w=max(9, round(median_w * 1.8)),
                min_h=max(8, round(median_h * 0.55)),
                max_h=max(9, round(median_h * 1.8)),
            )
            if 0.55 <= w / max(1, h) <= 1.8
        ]
        if len(detected) < len(expected):
            return False

        unmatched = set(range(len(detected)))
        for source in sorted(expected, key=lambda box: (box.y, box.x)):
            source_center = (source.x + source.w / 2, source.y + source.h / 2)
            center_tolerance = max(4.0, min(source.w, source.h) * 0.45)
            width_tolerance = max(3.0, source.w * 0.35)
            height_tolerance = max(3.0, source.h * 0.35)
            viable = []
            for candidate_idx in unmatched:
                candidate = detected[candidate_idx]
                center_distance = float(
                    np.hypot(
                        source_center[0] - candidate.x - candidate.w / 2,
                        source_center[1] - candidate.y - candidate.h / 2,
                    )
                )
                if (
                    center_distance <= center_tolerance
                    and abs(source.w - candidate.w) <= width_tolerance
                    and abs(source.h - candidate.h) <= height_tolerance
                ):
                    viable.append((center_distance, candidate_idx))
            if not viable:
                return False
            _, best_idx = min(viable)
            unmatched.remove(best_idx)

    return saw_checkbox


def _collect_ink_data(
    working_boxes: list[Box],
    pure_ink_masks: dict[int, np.ndarray],
) -> tuple[list[int], list[int], list[Box]]:
    inks: list[int] = []
    areas: list[int] = []
    valid_boxes: list[Box] = []

    for box in working_boxes:
        if box.page_idx not in pure_ink_masks:
            inks.append(0)
            areas.append(0)
            valid_boxes.append(box)
            continue

        ink, area = extract_ink_info_from_mask(pure_ink_masks[box.page_idx], box)
        inks.append(ink)
        areas.append(area)
        valid_boxes.append(box)

    return inks, areas, valid_boxes


# ==========================================
# 3. 파이프라인 관리 모듈
# ==========================================


def process_survey_data(
    survey_data: dict,
    config: TemplatePreset,
    dynamic_templates: dict[int, np.ndarray],
    template_masks: dict[int, np.ndarray] | None = None,
    field_plans: list[tuple[Field, list[Box], list[Box], bool]] | None = None,
    trust_checkbox_layout: bool = False,
) -> tuple[dict, dict, dict, dict, dict, dict]:
    fname = survey_data.get("fname", "")
    survey_label = survey_data["row_title"]
    row_data = {"파일명": fname, "페이지": survey_label}
    survey_gray_pages = survey_data["gray_pages"]
    # PNG 압축 해제 (메모리 절감: raw numpy 대신 PNG bytes로 저장됨)
    if survey_gray_pages:
        first = next(iter(survey_gray_pages.values()))
        if isinstance(first, bytes):
            decoded_pages = {}
            for local_p, data in survey_gray_pages.items():
                image = cv2.imdecode(
                    np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE
                )
                if image is not None:
                    decoded_pages[local_p] = image
            survey_gray_pages = decoded_pages
    survey_ink_only_images = {}
    comment_hits = set()

    # 벡터 주석 수집기 (page_idx -> list of (x, y, w, h, label, is_ticked))
    debug_annotations = {local_p: [] for local_p in survey_gray_pages}
    ink_annotations: dict[int, list] = {
        local_p: [] for local_p in survey_gray_pages
    }

    pure_ink_masks = {}
    for local_p, gray_img in survey_gray_pages.items():
        if local_p in dynamic_templates:
            pure_ink_masks[local_p] = extract_pure_ink_mask(
                gray_img,
                dynamic_templates[local_p],
                config.template_dilate_pct,
                template_masks.get(local_p) if template_masks else None,
            )

    # 반복 응답이 중앙값 템플릿에 섞여 지워진 경우에도 검토 PDF에서 보이도록,
    # 체크박스 내부 직접 추출 결과는 모든 문항 평가가 끝난 뒤 마스크에 합칩니다.
    checkbox_ink_additions: dict[int, np.ndarray] = {}
    plans = field_plans if field_plans is not None else _prepare_field_plans(config)

    checkbox_analyses: dict[int, _CheckboxFieldAnalysis] = {}
    ownership_candidates: list[
        tuple[_CheckboxInkInfo, bool, _CheckboxHaloInfo]
    ] = []
    for plan_idx, (field, scoring_boxes, _working_boxes, _is_contiguous) in enumerate(
        plans
    ):
        checkbox_mode = bool(scoring_boxes) and not field.is_comment and all(
            box.page_idx in survey_gray_pages
            and _is_checkbox_like(box, survey_gray_pages[box.page_idx].shape)
            for box in scoring_boxes
        )
        if not checkbox_mode:
            continue

        checkbox_infos = [
            extract_checkbox_ink_info(survey_gray_pages[box.page_idx], box)
            for box in scoring_boxes
        ]
        direct_inks = [info.ink_pixels for info in checkbox_infos]
        direct_areas = [info.area for info in checkbox_infos]
        reliable_direct = [
            trust_checkbox_layout
            or info.border_confidence >= _CHECKBOX_MIN_BORDER_CONFIDENCE
            for info in checkbox_infos
        ]
        direct_results = evaluate_checkbox_marks(
            direct_inks, direct_areas, strict=field.allow_duplicates
        )
        direct_results = [
            is_checked and is_reliable
            for is_checked, is_reliable in zip(direct_results, reliable_direct)
        ]
        halo_infos: list[_CheckboxHaloInfo] = []
        for info, is_reliable in zip(checkbox_infos, reliable_direct):
            if is_reliable and info.box.page_idx in pure_ink_masks:
                halo_infos.append(
                    _extract_checkbox_halo_info(
                        pure_ink_masks[info.box.page_idx],
                        info.box,
                        target_gray=survey_gray_pages[info.box.page_idx],
                        box_is_refined=True,
                    )
                )
            else:
                halo_infos.append(
                    _CheckboxHaloInfo(
                        0,
                        0,
                        info.box,
                        (info.box.x, info.box.y, info.box.x, info.box.y),
                        np.zeros((0, 0), np.uint8),
                    )
                )

        checkbox_analyses[plan_idx] = _CheckboxFieldAnalysis(
            checkbox_infos,
            reliable_direct,
            direct_inks,
            direct_areas,
            direct_results,
            halo_infos,
        )
        ownership_candidates.extend(zip(checkbox_infos, direct_results, halo_infos))

    _resolve_checkbox_halo_ownership(
        pure_ink_masks, survey_gray_pages, ownership_candidates
    )

    for plan_idx, (field, scoring_boxes, working_boxes, is_contiguous) in enumerate(
        plans
    ):
        current_inks, current_areas, current_boxes = _collect_ink_data(
            working_boxes, pure_ink_masks
        )
        checkbox_analysis = checkbox_analyses.get(plan_idx)
        checkbox_mode = checkbox_analysis is not None

        direct_inks: list[int] = []
        direct_areas: list[int] = []
        direct_results: list[bool] = []
        halo_inks: list[int] = []
        halo_areas: list[int] = []
        reliable_direct: list[bool] = []

        if checkbox_analysis is not None:
            checkbox_infos = checkbox_analysis.checkbox_infos
            reliable_direct = checkbox_analysis.reliable_direct
            direct_inks = checkbox_analysis.direct_inks
            direct_areas = checkbox_analysis.direct_areas
            direct_results = checkbox_analysis.direct_results
            halo_infos = checkbox_analysis.halo_infos
            halo_inks = [info.ink_pixels for info in halo_infos]
            halo_areas = [info.area for info in halo_infos]
            halo_results = evaluate_checkbox_halo_marks(
                halo_inks,
                halo_areas,
                is_contiguous,
                strict=field.allow_duplicates,
            )
            check_results = [
                direct or halo
                for direct, halo in zip(direct_results, halo_results)
            ]
            valid_boxes = [
                info.box if is_reliable else source_box
                for info, is_reliable, source_box in zip(
                    checkbox_infos, reliable_direct, scoring_boxes
                )
            ]

            for info, halo_info, is_reliable, halo_is_checked in zip(
                checkbox_infos,
                halo_infos,
                reliable_direct,
                halo_results,
            ):
                if not is_reliable:
                    continue
                addition = checkbox_ink_additions.setdefault(
                    info.box.page_idx,
                    np.zeros_like(survey_gray_pages[info.box.page_idx], dtype=np.uint8),
                )
                if info.ink_mask.size > 0:
                    x1, y1, x2, y2 = info.mask_bounds
                    addition[y1:y2, x1:x2] = cv2.bitwise_or(
                        addition[y1:y2, x1:x2], info.ink_mask
                    )
                if halo_is_checked and halo_info.ink_mask.size > 0:
                    x1, y1, x2, y2 = halo_info.mask_bounds
                    addition[y1:y2, x1:x2] = cv2.bitwise_or(
                        addition[y1:y2, x1:x2], halo_info.ink_mask
                    )
        else:
            check_results = evaluate_marks(
                current_inks,
                current_areas,
                is_contiguous,
                strict=field.allow_duplicates,
            )
            valid_boxes = current_boxes

        inks = halo_inks if checkbox_mode else current_inks
        areas = halo_areas if checkbox_mode else current_areas

        if field.is_comment:
            check_results = [
                (ink > 10) or (area > 0 and (ink / area) >= 0.01)
                for ink, area in zip(inks, areas)
            ]

        if field.is_comment:
            has_comment = False
            total_boxes = len(valid_boxes)

            for idx, (box, is_ticked) in enumerate(
                zip(valid_boxes, check_results), start=1
            ):
                label_number = _label_number(total_boxes, idx, config.reverse_numbering)
                label = str(label_number)

                if box.page_idx in debug_annotations:
                    debug_annotations[box.page_idx].append(
                        (box.x, box.y, box.w, box.h, label, is_ticked)
                    )
                if box.page_idx in ink_annotations:
                    ink_annotations[box.page_idx].append(
                        (box.x, box.y, box.w, box.h, label, is_ticked)
                    )

                if is_ticked:
                    comment_hits.add(box.page_idx)
                    has_comment = True

            row_data[field.name] = "있음" if has_comment else ""
            continue

        if not field.allow_duplicates:
            if checkbox_mode and sum(check_results) > 1:
                checked_indices = [
                    index for index, is_checked in enumerate(check_results) if is_checked
                ]
                best_idx = max(
                    checked_indices,
                    key=lambda index: (
                        direct_inks[index] * 3 + halo_inks[index],
                        direct_inks[index] / max(1, direct_areas[index]),
                        halo_inks[index] / max(1, halo_areas[index]),
                    ),
                )
                check_results = [
                    index == best_idx for index in range(len(check_results))
                ]
            else:
                check_results = enforce_single_choice(check_results, inks, areas)
        checked_labels = []
        total_boxes = len(valid_boxes)

        for idx, (box, is_ticked) in enumerate(
            zip(valid_boxes, check_results), start=1
        ):
            label_number = _label_number(total_boxes, idx, config.reverse_numbering)
            label = str(label_number)

            if box.page_idx in debug_annotations:
                debug_annotations[box.page_idx].append(
                    (box.x, box.y, box.w, box.h, label, is_ticked)
                )
            if box.page_idx in ink_annotations:
                ink_annotations[box.page_idx].append(
                    (box.x, box.y, box.w, box.h, label, is_ticked)
                )

            if is_ticked:
                mapped_value = ""
                if 0 < label_number <= len(field.value_map):
                    mapped_value = field.value_map[label_number - 1].strip()
                checked_labels.append((label, mapped_value))

        if checked_labels:
            output_values = [mv if mv else lbl for lbl, mv in checked_labels]
            row_data[field.name] = ",".join(output_values)
        else:
            row_data[field.name] = ""

    for local_p, addition in checkbox_ink_additions.items():
        if local_p in pure_ink_masks:
            pure_ink_masks[local_p] = cv2.bitwise_or(
                pure_ink_masks[local_p], addition
            )
        elif cv2.countNonZero(addition) > 0:
            pure_ink_masks[local_p] = addition

    survey_ink_only_images = {
        local_p: cv2.bitwise_not(mask)
        for local_p, mask in pure_ink_masks.items()
    }

    comment_pages = {
        local_p: survey_gray_pages[local_p]
        for local_p in sorted(comment_hits)
        if local_p in survey_gray_pages
    }

    return (
        row_data,
        survey_gray_pages,
        survey_ink_only_images,
        debug_annotations,
        ink_annotations,
        comment_pages,
    )


# ── Phase 1 Worker: 파일 1개에서 템플릿 샘플 수집 (스레드 안전) ──
# ── 페이지 렌더링 + 정합 헬퍼 (inner pool에서 호출) ──
def _render_aligned_page(
    doc,
    global_p: int,
    local_p: int,
    aligners: list,
    rot_code: int,
    fine_angle: float,
    dpi: int,
) -> tuple[int, np.ndarray]:
    page = doc[global_p]
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    page_img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)
    orig = apply_rotation(page_img, rot_code, fine_angle)
    a = aligners[local_p] if local_p < len(aligners) else aligners[-1]
    return local_p, a.align(orig)


def _render_survey_pages(
    doc,
    survey_idx: int,
    page_count: int,
    aligners: list,
    rot_code: int,
    fine_angle: float,
    dpi: int,
) -> dict[int, np.ndarray]:
    """한 설문의 페이지를 순차 렌더링·정합합니다."""
    sequential_result: dict[int, np.ndarray] = {}
    for local_p in range(page_count):
        global_p = survey_idx * page_count + local_p
        if global_p >= len(doc):
            break
        _, aligned = _render_aligned_page(
            doc, global_p, local_p, aligners, rot_code, fine_angle, dpi
        )
        sequential_result[local_p] = aligned
    return sequential_result


def _reference_page_pixels(alignment_references: list[np.ndarray]) -> list[int]:
    return [
        int(reference.shape[0] * reference.shape[1])
        for reference in alignment_references
    ]


def _estimate_render_memory_bytes(
    alignment_references: list[np.ndarray],
) -> int:
    pixels = _reference_page_pixels(alignment_references)
    return (max(pixels, default=0) * 10) + 64 * _MIB


def _estimate_survey_memory_bytes(
    alignment_references: list[np.ndarray],
) -> int:
    pixels = _reference_page_pixels(alignment_references)
    return (sum(pixels) * 7) + (max(pixels, default=0) * 4) + 64 * _MIB


def _estimate_template_memory_bytes(
    sample_pages: dict[int, list[bytes]],
    alignment_references: list[np.ndarray],
) -> int:
    pixels = _reference_page_pixels(alignment_references)
    peak_bytes = 0
    for local_p, samples in sample_pages.items():
        page_pixels = (
            pixels[local_p] if local_p < len(pixels) else max(pixels, default=0)
        )
        # PNG 디코딩 배열과 중앙값 partition 스택이 동시에 존재합니다.
        peak_bytes = max(peak_bytes, page_pixels * max(1, len(samples)) * 2)
    return peak_bytes + (max(pixels, default=0) * 4) + 64 * _MIB


# ── Phase 1 Worker: 파일 1개에서 템플릿 샘플 수집 (스레드 안전) ──
def _collect_template_samples(
    fpath: str,
    config: TemplatePreset,
    alignment_references: list[np.ndarray],
    dpi: int = 300,
    sample_limit: int = _UI_TEMPLATE_SAMPLE_LIMIT,
    resource_controller: AdaptiveResourceController | None = None,
    resource_status_cb=None,
    progress_cb=None,
) -> tuple[str, dict[int, list[bytes]]]:
    fname = Path(fpath).stem
    try:
        doc = fitz.open(fpath)
    except Exception as e:
        print(f"파일 로드 실패 ({fname}): {e}")
        return _file_key(fpath), {}

    page_count = config.page_count
    aligners = [ImageAligner(reference) for reference in alignment_references]
    if not aligners:
        doc.close()
        raise RuntimeError("페이지 정합 기준 이미지가 없습니다.")

    f_pages: dict[int, list[bytes]] = {i: [] for i in range(page_count)}
    survey_count = _survey_count(len(doc), page_count)
    limit = min(survey_count, sample_limit)

    try:
        for survey_idx in range(limit):
            if resource_controller is not None:
                resource_controller.checkpoint(
                    _estimate_render_memory_bytes(alignment_references),
                    stage=f"{fname} 템플릿 표본 처리",
                    status_cb=resource_status_cb,
                )
            pages = _render_survey_pages(
                doc,
                survey_idx,
                page_count,
                aligners,
                config.rot_code,
                config.fine_angle,
                dpi,
            )
            for local_p, aligned in pages.items():
                if len(f_pages[local_p]) >= sample_limit:
                    continue
                success, encoded = cv2.imencode(".png", aligned)
                f_pages[local_p].append(encoded.tobytes() if success else b"")
            if progress_cb:
                progress_cb(survey_idx + 1, limit)
    finally:
        doc.close()

    return _file_key(fpath), f_pages


def _decode_sampled_survey(
    sample_pages: dict[int, list[bytes]] | None,
    survey_idx: int,
    expected_pages: int,
) -> dict[int, np.ndarray] | None:
    """Phase 1에서 만든 lossless PNG를 재사용해 중복 렌더링·정합을 피합니다."""
    if not sample_pages:
        return None

    decoded: dict[int, np.ndarray] = {}
    for local_p in range(expected_pages):
        samples = sample_pages.get(local_p, [])
        if survey_idx >= len(samples):
            return None
        data = samples[survey_idx]
        if not data:
            return None
        image = cv2.imdecode(
            np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE
        )
        if image is None:
            return None
        decoded[local_p] = image
    return decoded


def _build_file_templates(
    file_paths: list[str],
    sample_results: dict[str, dict[int, list[bytes]]],
) -> tuple[
    dict[int, np.ndarray] | None,
    dict[str, dict[int, np.ndarray]],
]:
    """공통 기준에 정합된 표본으로 파일별 템플릿을 만듭니다.

    표본 페이지는 수집 단계에서 이미 같은 기준 이미지에 정합되어 있습니다. 파일별
    중앙값 템플릿만 다시 첫 파일에 맞추면 템플릿에만 추가 변환이 생겨 실제 분석
    페이지와 좌표계가 달라지므로, 생성된 템플릿을 그대로 유지합니다.
    """
    reference_templates: dict[int, np.ndarray] | None = None
    file_templates: dict[str, dict[int, np.ndarray]] = {}

    for fpath in file_paths:
        file_key = _file_key(fpath)
        f_pages = sample_results.get(file_key, {})
        file_template = generate_dynamic_templates(f_pages) if f_pages else {}
        file_templates[file_key] = file_template
        if reference_templates is None and file_template:
            reference_templates = file_template

    return reference_templates, file_templates


# ── Phase 2 Worker: 파일 1개 전체 분석 (스레드 안전) ──
def _analyze_single_file(
    fpath: str,
    file_label: str,
    config: TemplatePreset,
    file_template: dict[int, np.ndarray],
    reference_templates: dict[int, np.ndarray],
    alignment_references: list[np.ndarray],
    review_folder: Path,
    dpi: int = 300,
    sample_pages: dict[int, list[bytes]] | None = None,
    resource_controller: AdaptiveResourceController | None = None,
    resource_status_cb=None,
    progress_cb=None,
) -> tuple[str, list[dict], list[bytes]]:
    try:
        doc = fitz.open(fpath)
    except Exception as e:
        raise RuntimeError(f"파일 로드 실패: {e}") from e

    out_orig = None
    out_ink = None
    try:
        out_orig = fitz.open()
        out_ink = fitz.open()
        survey_count = _survey_count(len(doc), config.page_count)
        aligners = [ImageAligner(reference) for reference in alignment_references]
        if not aligners:
            raise RuntimeError("페이지 정합 기준 이미지가 없습니다.")
        f_template = file_template or reference_templates
        template_masks = {}
        for local_p, template in f_template.items():
            gray_template = (
                cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                if template.ndim == 3
                else template
            )
            template_masks[local_p] = cv2.threshold(
                gray_template, 200, 255, cv2.THRESH_BINARY_INV
            )[1]
        analysis_config = _remap_checkbox_layout(config, f_template)
        trust_checkbox_layout = _checkbox_layout_is_trustworthy(
            analysis_config, f_template
        )
        field_plans = _prepare_field_plans(analysis_config)

        page_count = config.page_count
        rot_code = config.rot_code
        fine_angle = config.fine_angle
        file_results: list[dict] = []
        comment_pages: list[bytes] = []
        survey_memory_bytes = _estimate_survey_memory_bytes(alignment_references)

        for survey_idx in range(survey_count):
            if resource_controller is not None:
                resource_controller.checkpoint(
                    survey_memory_bytes,
                    stage=f"{file_label} 설문 분석",
                    status_cb=resource_status_cb,
                )
            expected_pages = min(
                page_count, max(0, len(doc) - survey_idx * page_count)
            )
            survey_gray_pages = _decode_sampled_survey(
                sample_pages, survey_idx, expected_pages
            )
            if survey_gray_pages is None:
                survey_gray_pages = _render_survey_pages(
                    doc,
                    survey_idx,
                    page_count,
                    aligners,
                    rot_code,
                    fine_angle,
                    dpi,
                )

            survey_data = {
                "fname": file_label,
                "row_title": f"{file_label}_{survey_idx + 1}p",
                "gray_pages": survey_gray_pages,
            }

            row_data, debug_base, ink_base, debug_ann, ink_ann, cp = (
                process_survey_data(
                    survey_data,
                    analysis_config,
                    f_template,
                    template_masks,
                    field_plans,
                    trust_checkbox_layout=trust_checkbox_layout,
                )
            )

            field_values = [
                v for k, v in row_data.items() if k not in ("파일명", "페이지")
            ]
            if any(v.strip() for v in field_values):
                file_results.append(row_data)

            for local_p in sorted(debug_base):
                _build_vector_page(
                    out_orig, debug_base[local_p], debug_ann.get(local_p, [])
                )
            for local_p in sorted(ink_base):
                _build_vector_page(
                    out_ink, ink_base[local_p], ink_ann.get(local_p, [])
                )
            for local_p in sorted(cp):
                image_bytes = _encode_jpeg(cp[local_p])
                if image_bytes:
                    comment_pages.append(image_bytes)

            if progress_cb:
                progress_cb(survey_idx + 1, survey_count)

        if len(out_orig) > 0:
            out_orig.save(review_folder / f"{file_label}_원본포함.pdf")
        if len(out_ink) > 0:
            out_ink.save(review_folder / f"{file_label}_잉크추출.pdf")

        return file_label, file_results, comment_pages
    finally:
        doc.close()
        if out_orig is not None:
            out_orig.close()
        if out_ink is not None:
            out_ink.close()


def run_analysis(
    file_paths: list[str],
    template_pages: list,
    config: TemplatePreset,
    progress_cb=None,
    resource_controller: AdaptiveResourceController | None = None,
) -> bool:
    review_folder = Path("검토용")
    review_folder.mkdir(exist_ok=True)

    def report_progress(value: float, message: str = ""):
        if progress_cb:
            progress_cb(max(0, min(100, value)), message)

    report_progress(0, "분석 준비 중...")

    num_files = len(file_paths)
    if num_files == 0:
        return False
    file_labels = _build_file_labels(file_paths)
    survey_counts = _file_survey_counts(file_paths, config.page_count)
    sample_counts = [
        min(count, _UI_TEMPLATE_SAMPLE_LIMIT) for count in survey_counts
    ]
    total_work = sum(
        (sample_count * _ANALYSIS_SAMPLE_WORK)
        + _ANALYSIS_TEMPLATE_WORK
        + _analysis_survey_work(0, survey_count, sample_count)
        for sample_count, survey_count in zip(sample_counts, survey_counts)
    )

    # 정합 기준 이미지만 공유하고, 상태를 가진 ImageAligner는 파일마다 새로 만듭니다.
    alignment_references = [
        apply_rotation(p, config.rot_code, config.fine_angle)
        for p in template_pages[: config.page_count]
    ]
    if not alignment_references:
        print("페이지 정합 기준 이미지가 없습니다.")
        return False

    completed_work = 0.0

    def current_work_progress() -> float:
        if total_work <= 0:
            return _ANALYSIS_PROGRESS_START
        return _ANALYSIS_PROGRESS_START + (
            completed_work / total_work * _ANALYSIS_PROGRESS_SPAN
        )

    def report_work(message: str) -> None:
        report_progress(current_work_progress(), message)

    def advance_work(units: float, message: str) -> None:
        nonlocal completed_work
        completed_work = min(total_work, completed_work + max(0.0, units))
        report_work(message)

    # 파일별로 표본 수집 → 템플릿 생성 → 분석을 끝낸 뒤 큰 객체를 바로 해제합니다.
    # 모든 파일의 300 DPI PNG 표본을 한꺼번에 보관하거나 두 PDF를 동시에 분석하면
    # 메모리가 작은 PC에서 피크 사용량이 크게 늘어나므로 파일 단위 병렬화는 하지 않습니다.
    report_progress(_ANALYSIS_PROGRESS_START, "파일별 템플릿 생성 및 분석 중...")
    controller = resource_controller or AdaptiveResourceController()
    controller.start()
    reference_templates: dict[int, np.ndarray] | None = None
    pending_indices: list[int] = []
    all_results: list[dict] = []
    analysis_failures: list[str] = []
    completed = 0
    comment_path = Path("의견.pdf")
    try:
        comment_doc = fitz.open()
    except Exception:
        controller.close()
        raise

    def save_reference_template() -> None:
        if reference_templates is None:
            return
        template_pdf = fitz.open()
        try:
            for local_p in sorted(reference_templates):
                _insert_img_into_pdf(
                    template_pdf, reference_templates[local_p], quality=90
                )
            if len(template_pdf) > 0:
                template_pdf.save(review_folder / "00_추론된_템플릿.pdf")
        finally:
            template_pdf.close()

    def analyze_file(
        index: int,
        file_template: dict[int, np.ndarray],
        sample_pages: dict[int, list[bytes]] | None,
    ) -> None:
        nonlocal completed
        fpath = file_paths[index]
        file_label = file_labels[index]
        expected_surveys = survey_counts[index]
        analysis_done = 0
        comment_pages: list[bytes] = []

        def analysis_progress(done: int, total: int) -> None:
            nonlocal analysis_done
            bounded_done = min(expected_surveys, max(analysis_done, int(done)))
            newly_done = bounded_done - analysis_done
            previous_done = analysis_done
            analysis_done = bounded_done
            if newly_done > 0:
                advance_work(
                    _analysis_survey_work(
                        previous_done, analysis_done, sample_counts[index]
                    ),
                    f"{file_label}: 설문 분석 중 ({done}/{total}) · "
                    f"파일 {index + 1}/{num_files}",
                )

        try:
            if reference_templates is None:
                raise RuntimeError("분석 기준 템플릿이 없습니다.")
            _, file_results, comment_pages = _analyze_single_file(
                fpath,
                file_label,
                config,
                file_template,
                reference_templates,
                alignment_references,
                review_folder,
                sample_pages=sample_pages,
                resource_controller=controller,
                resource_status_cb=report_work,
                progress_cb=analysis_progress,
            )
            all_results.extend(file_results)
            for image_bytes in comment_pages:
                _insert_encoded_img_into_pdf(comment_doc, image_bytes)
        except ResourceUnavailableError:
            raise
        except Exception as e:
            analysis_failures.append(file_label)
            print(f"분석 실패 ({file_label}): {e}")
        finally:
            comment_pages.clear()
            if analysis_done < expected_surveys:
                advance_work(
                    _analysis_survey_work(
                        analysis_done, expected_surveys, sample_counts[index]
                    ),
                    f"{file_label}: 설문 분석 단계 정리 중...",
                )
            completed += 1
            report_work(f"파일 분석 완료 ({completed}/{num_files})")

    try:
        for index, (fpath, file_label) in enumerate(zip(file_paths, file_labels)):
            report_work(
                f"{file_label}: 템플릿 표본 준비 중 · 파일 {index + 1}/{num_files}"
            )
            sample_pages: dict[int, list[bytes]] = {}
            file_template: dict[int, np.ndarray] = {}
            expected_samples = sample_counts[index]
            samples_done = 0

            def sample_progress(done: int, total: int) -> None:
                nonlocal samples_done
                bounded_done = min(expected_samples, max(samples_done, int(done)))
                newly_done = bounded_done - samples_done
                samples_done = bounded_done
                if newly_done > 0:
                    advance_work(
                        newly_done * _ANALYSIS_SAMPLE_WORK,
                        f"{file_label}: 템플릿 표본 처리 중 ({done}/{total}) · "
                        f"파일 {index + 1}/{num_files}",
                    )

            try:
                _, sample_pages = _collect_template_samples(
                    fpath,
                    config,
                    alignment_references,
                    resource_controller=controller,
                    resource_status_cb=report_work,
                    progress_cb=sample_progress,
                )
                if sample_pages:
                    controller.checkpoint(
                        _estimate_template_memory_bytes(
                            sample_pages, alignment_references
                        ),
                        stage=f"{file_label} 템플릿 생성",
                        status_cb=report_work,
                    )
                    report_work(
                        f"{file_label}: 동적 템플릿 생성 중 · "
                        f"파일 {index + 1}/{num_files}"
                    )
                    file_template = generate_dynamic_templates(sample_pages)
            except ResourceUnavailableError:
                raise
            except Exception as e:
                print(f"템플릿 샘플 수집 실패 ({file_label}): {e}")
            finally:
                if samples_done < expected_samples:
                    advance_work(
                        (expected_samples - samples_done) * _ANALYSIS_SAMPLE_WORK,
                        f"{file_label}: 템플릿 표본 단계 정리 중...",
                    )
                advance_work(
                    _ANALYSIS_TEMPLATE_WORK,
                    f"{file_label}: 템플릿 준비 완료 · "
                    f"파일 {index + 1}/{num_files}",
                )

            if reference_templates is None:
                if not file_template:
                    # 기준이 생길 때까지 경로만 기억합니다. 앞 파일의 큰 PNG 표본은
                    # 보관하지 않고 나중에 기준 템플릿으로 다시 렌더링합니다.
                    pending_indices.append(index)
                    sample_pages.clear()
                    gc.collect()
                    continue

                reference_templates = file_template
                save_reference_template()
                for pending_index in pending_indices:
                    analyze_file(pending_index, reference_templates, None)
                    gc.collect()
                pending_indices.clear()

            analyze_file(
                index,
                file_template or reference_templates,
                sample_pages or None,
            )

            sample_pages.clear()
            if file_template is not reference_templates:
                file_template.clear()
            gc.collect()

        if reference_templates is None:
            print("템플릿 생성에 실패했습니다.")
            return False

        report_progress(97, "분석 결과 정리 중...")

        # 의견 이미지는 파일 분석 직후 PDF 문서로 옮겼으므로 JPEG 목록을 따로
        # 누적하지 않습니다.
        if len(comment_doc) > 0:
            comment_doc.save(comment_path)
        elif comment_path.exists():
            comment_path.unlink()
    finally:
        comment_doc.close()
        controller.close()

    # ── 엑셀 저장 ──
    report_progress(98, "엑셀 저장 중...")
    success = export_to_excel(all_results, config)
    report_progress(100, "완료")
    return success and not analysis_failures
