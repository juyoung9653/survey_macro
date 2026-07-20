import copy
import hashlib
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import fitz
import numpy as np

from .export import export_to_excel
from .models import Box, Field, TemplatePreset
from .vision import ImageAligner, apply_rotation, load_pdf_pages


_UI_TEMPLATE_SAMPLE_LIMIT = 31
_UI_TEMPLATE_CACHE_VERSION = 5


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


def _select_working_boxes(
    field, z_sorted_boxes: list[Box], all_boxes: list[Box]
) -> list[Box]:
    if field.is_comment or field.allow_duplicates:
        return [copy.copy(b) for b in z_sorted_boxes]

    return expand_isolated_boxes(z_sorted_boxes, all_boxes, scale_factor=2.0)


def _prepare_field_plans(
    config: TemplatePreset,
) -> list[tuple[Field, list[Box], bool]]:
    """설문마다 동일한 박스 정렬·확장 결과를 분석 시작 전에 한 번만 계산합니다."""
    all_boxes = [box for field in config.fields for box in field.boxes]
    plans = []
    for field in config.fields:
        sorted_boxes = sort_boxes_z_pattern(field.boxes)
        working_boxes = _select_working_boxes(field, sorted_boxes, all_boxes)
        plans.append((field, working_boxes, is_contiguous_group(sorted_boxes)))
    return plans


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
    field_plans: list[tuple[Field, list[Box], bool]] | None = None,
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
    ink_annotations: dict[int, list] = {}

    pure_ink_masks = {}
    for local_p, gray_img in survey_gray_pages.items():
        if local_p in dynamic_templates:
            mask = extract_pure_ink_mask(
                gray_img,
                dynamic_templates[local_p],
                config.template_dilate_pct,
                template_masks.get(local_p) if template_masks else None,
            )
            pure_ink_masks[local_p] = mask

            ink_display = cv2.bitwise_not(mask)
            survey_ink_only_images[local_p] = ink_display
            ink_annotations[local_p] = []

    plans = field_plans if field_plans is not None else _prepare_field_plans(config)

    for field, working_boxes, is_contiguous in plans:
        inks, areas, valid_boxes = _collect_ink_data(working_boxes, pure_ink_masks)

        if field.is_comment:
            check_results = [
                (ink > 10) or (area > 0 and (ink / area) >= 0.01)
                for ink, area in zip(inks, areas)
            ]
        else:
            check_results = evaluate_marks(
                inks, areas, is_contiguous, strict=field.allow_duplicates
            )

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


# ── Phase 1 Worker: 파일 1개에서 템플릿 샘플 수집 (스레드 안전) ──
def _collect_template_samples(
    fpath: str,
    config: TemplatePreset,
    alignment_references: list[np.ndarray],
    dpi: int = 300,
    sample_limit: int = 31,
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
        field_plans = _prepare_field_plans(config)

        page_count = config.page_count
        rot_code = config.rot_code
        fine_angle = config.fine_angle
        file_results: list[dict] = []
        comment_pages: list[bytes] = []

        for survey_idx in range(survey_count):
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
                    config,
                    f_template,
                    template_masks,
                    field_plans,
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
) -> bool:
    review_folder = Path("검토용")
    review_folder.mkdir(exist_ok=True)

    def report_progress(value: int, message: str = ""):
        if progress_cb:
            progress_cb(max(0, min(100, value)), message)

    report_progress(0, "분석 준비 중...")

    num_files = len(file_paths)
    if num_files == 0:
        return False
    file_labels = _build_file_labels(file_paths)

    # 정합 기준 이미지만 공유하고, 상태를 가진 ImageAligner는 파일 작업자마다 만듭니다.
    # 배치 처리에서 여러 스레드가 같은 OpenCV ORB 인스턴스를 동시에 쓰지 않게 합니다.
    alignment_references = [
        apply_rotation(p, config.rot_code, config.fine_angle)
        for p in template_pages[: config.page_count]
    ]
    if not alignment_references:
        print("페이지 정합 기준 이미지가 없습니다.")
        return False

    # ══════════════════════════════════════
    # Phase 1: 템플릿 샘플 병렬 수집
    # ══════════════════════════════════════
    report_progress(0, "템플릿 샘플 수집 중...")
    sample_results: dict[str, dict[int, list[bytes]]] = {}
    # OpenCV도 내부 스레드를 사용하므로 파일 단위 작업은 2개까지만 병렬화합니다.
    max_workers = min(2, num_files)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _collect_template_samples, fpath, config, alignment_references
            ): index
            for index, fpath in enumerate(file_paths)
        }
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            try:
                file_key, f_pages = future.result()
                if f_pages:
                    sample_results[file_key] = f_pages
            except Exception as e:
                print(f"템플릿 샘플 수집 실패 ({file_labels[index]}): {e}")
            completed += 1
            report_progress(
                int(completed / num_files * 8),
                f"템플릿 샘플 수집 중... ({completed}/{num_files})",
            )

    # ── 템플릿 빌드 (원본 순서 및 공통 정합 좌표 유지) ──
    report_progress(8, "템플릿 생성 중...")
    reference_templates, file_templates = _build_file_templates(
        file_paths, sample_results
    )

    if reference_templates is None:
        print("템플릿 생성에 실패했습니다.")
        return False

    for file_key in file_templates:
        if not file_templates[file_key]:
            file_templates[file_key] = reference_templates

    # ── 검토용 템플릿 PDF 저장 ──
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

    # ══════════════════════════════════════
    # Phase 2: 파일별 분석 병렬 처리
    # ══════════════════════════════════════
    report_progress(10, "분석 시작 중...")
    ordered_outputs: list[tuple[list[dict], list[bytes]] | None] = [None] * num_files
    analysis_failures: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for index, (fpath, file_label) in enumerate(zip(file_paths, file_labels)):
            file_key = _file_key(fpath)
            future = executor.submit(
                _analyze_single_file,
                fpath,
                file_label,
                config,
                file_templates.get(file_key, reference_templates),
                reference_templates,
                alignment_references,
                review_folder,
                sample_pages=sample_results.get(file_key),
            )
            futures[future] = index

        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            file_key = _file_key(file_paths[index])
            try:
                _, file_results, comment_pages = future.result()
                ordered_outputs[index] = (file_results, comment_pages)
            except Exception as e:
                analysis_failures.append(file_labels[index])
                print(f"분석 실패 ({file_labels[index]}): {e}")
            finally:
                sample_results.pop(file_key, None)
            completed += 1
            report_progress(
                10 + int(completed / num_files * 88),
                f"분석 중... ({completed}/{num_files})",
            )

    sample_results.clear()
    all_results: list[dict] = []
    all_comment_pages: list[bytes] = []
    for output in ordered_outputs:
        if output is None:
            continue
        file_results, comment_pages = output
        all_results.extend(file_results)
        all_comment_pages.extend(comment_pages)

    # ── 의견 PDF 병합 저장 ──
    comment_path = Path("의견.pdf")
    comment_doc = fitz.open()
    try:
        for image_bytes in all_comment_pages:
            _insert_encoded_img_into_pdf(comment_doc, image_bytes)
        if len(comment_doc) > 0:
            comment_doc.save(comment_path)
        elif comment_path.exists():
            comment_path.unlink()
    finally:
        comment_doc.close()
    del all_comment_pages

    # ── 엑셀 저장 ──
    report_progress(98, "엑셀 저장 중...")
    success = export_to_excel(all_results, config)
    report_progress(100, "완료")
    return success and not analysis_failures
