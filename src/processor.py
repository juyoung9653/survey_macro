import copy
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import fitz
import numpy as np

from .export import export_to_excel
from .models import Box, TemplatePreset
from .vision import ImageAligner, apply_rotation, load_pdf_pages


def _gray_to_bgr_for_encode(img: np.ndarray) -> np.ndarray:
    """gray(2D)면 BGR로 변환, 이미 BGR이면 그대로 반환"""
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def _build_vector_page(
    target_doc, base_img: np.ndarray, annotations: list, img_quality: int = 85
) -> None:
    """벡터 PDF 페이지 생성: 배경 이미지(JPEG) + 벡터 사각형/텍스트 오버레이
    base_img는 gray(2D) 또는 BGR(3D) 모두 허용. gray면 내부에서 BGR로 변환 후 인코딩."""
    h, w = base_img.shape[:2]
    page = target_doc.new_page(width=w, height=h)
    bgr = _gray_to_bgr_for_encode(base_img)
    success, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, img_quality])
    del bgr
    if success:
        page.insert_image(page.rect, stream=buf.tobytes())
    for bx, by, bw, bh, label, is_ticked in annotations:
        color = (0, 1, 0) if is_ticked else (1, 0, 0)
        rect = fitz.Rect(bx, by, bx + bw, by + bh)
        page.draw_rect(rect, color=color, width=2)
        page.insert_text(fitz.Point(bx, max(0, by - 5)), label, fontsize=8, color=color)


def _insert_img_into_pdf(target_doc, img: np.ndarray, quality: int = 85) -> None:
    """numpy 이미지를 JPEG로 인코딩해 target_doc(fitz.Document)에 페이지로 추가
    img는 gray(2D) 또는 BGR(3D) 모두 허용."""
    bgr = _gray_to_bgr_for_encode(img)
    success, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    del bgr
    if not success:
        return
    img_doc = fitz.open("jpg", buf.tobytes())
    pdf_bytes = img_doc.convert_to_pdf()
    page_doc = fitz.open("pdf", pdf_bytes)
    target_doc.insert_pdf(page_doc)
    page_doc.close()
    img_doc.close()


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


def generate_dynamic_templates(
    pages_by_local_idx: dict[int, list],
) -> dict[int, np.ndarray]:
    templates = {}
    for local_p, pages in pages_by_local_idx.items():
        if not pages:
            continue
        sample_data = pages  # 상위에서 이미 50장으로 제한됨
        # PNG 압축된 bytes면 디코딩, raw array면 그대로 사용
        if isinstance(sample_data[0], bytes):
            decoded = [
                cv2.imdecode(np.frombuffer(d, np.uint8), cv2.IMREAD_GRAYSCALE)
                for d in sample_data
            ]
            # 밝기 outlier 필터: 지나치게 어두운 페이지(체크多) 제외
            decoded = _filter_blank_pages(decoded)
            if not decoded:
                continue
            stack = np.stack(decoded, axis=0)
        else:
            filtered = _filter_blank_pages(sample_data)
            if not filtered:
                continue
            stack = np.stack(filtered, axis=0)
        templates[local_p] = np.median(stack, axis=0).astype(np.uint8)
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
    try:
        pages = load_pdf_pages(pdf_path, progress_cb=progress_cb, gray=True)
    except Exception:
        return {}

    if not pages or page_count <= 0:
        return {}

    aligners = [
        ImageAligner(apply_rotation(p, rot_code, fine_angle))
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
            if len(pages_by_local_idx[local_p]) < 50:
                pages_by_local_idx[local_p].append(
                    cv2.imencode(".png", aligned)[1].tobytes()
                )

    dynamic_templates = generate_dynamic_templates(pages_by_local_idx)

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
    if not pdf_paths:
        return {}

    all_by_local_idx = {i: [] for i in range(page_count)}
    ref_aligners = None

    for f_i, fpath in enumerate(pdf_paths):
        try:
            pages = load_pdf_pages(fpath, gray=True)
        except Exception:
            continue

        if not pages or page_count <= 0:
            continue

        if ref_aligners is None:
            ref_aligners = [
                ImageAligner(apply_rotation(p, rot_code, fine_angle))
                for p in pages[:page_count]
            ]

        survey_count = _survey_count(len(pages), page_count)

        for survey_idx in range(survey_count):
            for local_p in range(page_count):
                global_p = survey_idx * page_count + local_p
                if global_p >= len(pages):
                    break

                orig = apply_rotation(pages[global_p], rot_code, fine_angle)
                aligner = (
                    ref_aligners[local_p]
                    if local_p < len(ref_aligners)
                    else ref_aligners[-1]
                )
                aligned = aligner.align(orig)
                if len(all_by_local_idx[local_p]) < 50:
                    all_by_local_idx[local_p].append(
                        cv2.imencode(".png", aligned)[1].tobytes()
                    )

        if progress_cb:
            progress_cb(int((f_i + 1) / len(pdf_paths) * 100), "템플릿 병합 중...")

    dynamic_templates = generate_dynamic_templates(all_by_local_idx)

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
) -> np.ndarray:
    """템플릿을 대상에 미세 정합해 제거하고 순수 사용자 잉크만 추출합니다."""
    if target_gray.ndim == 3:
        target_gray = cv2.cvtColor(target_gray, cv2.COLOR_BGR2GRAY)
    if template_gray.ndim == 3:
        template_gray = cv2.cvtColor(template_gray, cv2.COLOR_BGR2GRAY)

    # 1. 대상과 템플릿의 어두운 픽셀 마스크 생성
    _, target_mask = cv2.threshold(target_gray, 200, 255, cv2.THRESH_BINARY_INV)
    _, template_mask = cv2.threshold(
        template_gray, 200, 255, cv2.THRESH_BINARY_INV
    )

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
    survey_data: dict, config: TemplatePreset, dynamic_templates: dict[int, np.ndarray]
) -> tuple[dict, dict, dict, dict, dict, dict]:
    fname = survey_data.get("fname", "")
    survey_label = survey_data["row_title"]
    row_data = {"파일명": fname, "페이지": survey_label}
    survey_gray_pages = survey_data["gray_pages"]
    # PNG 압축 해제 (메모리 절감: raw numpy 대신 PNG bytes로 저장됨)
    if survey_gray_pages:
        first = next(iter(survey_gray_pages.values()))
        if isinstance(first, bytes):
            survey_gray_pages = {
                k: cv2.imdecode(np.frombuffer(v, np.uint8), cv2.IMREAD_GRAYSCALE)
                for k, v in survey_gray_pages.items()
            }
    survey_ink_only_images = {}
    comment_hits = set()

    # 벡터 주석 수집기 (page_idx -> list of (x, y, w, h, label, is_ticked))
    debug_annotations = {local_p: [] for local_p in survey_gray_pages}
    ink_annotations: dict[int, list] = {}

    pure_ink_masks = {}
    for local_p, gray_img in survey_gray_pages.items():
        if local_p in dynamic_templates:
            mask = extract_pure_ink_mask(
                gray_img, dynamic_templates[local_p], config.template_dilate_pct
            )
            pure_ink_masks[local_p] = mask

            ink_display = cv2.bitwise_not(mask)
            survey_ink_only_images[local_p] = ink_display
            ink_annotations[local_p] = []

    all_boxes_in_config = [b for f in config.fields for b in f.boxes]

    for field in config.fields:
        z_sorted_boxes = sort_boxes_z_pattern(field.boxes)
        is_contiguous = is_contiguous_group(z_sorted_boxes)

        working_boxes = _select_working_boxes(
            field, z_sorted_boxes, all_boxes_in_config
        )

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
    inner_pool: ThreadPoolExecutor | None,
) -> dict[int, np.ndarray]:
    """한 설문의 모든 페이지를 렌더링+정합. inner_pool이 있으면 병렬 처리."""
    if inner_pool and page_count > 1:
        futures = {}
        for local_p in range(page_count):
            global_p = survey_idx * page_count + local_p
            if global_p >= len(doc):
                break
            future = inner_pool.submit(
                _render_aligned_page,
                doc, global_p, local_p, aligners, rot_code, fine_angle, dpi,
            )
            futures[future] = local_p

        result: dict[int, np.ndarray] = {}
        for future in as_completed(futures):
            local_p, aligned = future.result()
            result[local_p] = aligned
        return result

    # 순차 처리 (page_count == 1 또는 pool 미제공)
    result: dict[int, np.ndarray] = {}
    for local_p in range(page_count):
        global_p = survey_idx * page_count + local_p
        if global_p >= len(doc):
            break
        _, aligned = _render_aligned_page(
            doc, global_p, local_p, aligners, rot_code, fine_angle, dpi
        )
        result[local_p] = aligned
    return result


# ── Phase 1 Worker: 파일 1개에서 템플릿 샘플 수집 (스레드 안전) ──
def _collect_template_samples(
    fpath: str,
    config: TemplatePreset,
    aligners: list,
    dpi: int = 300,
    sample_limit: int = 31,
) -> tuple[str, dict[int, list[bytes]]]:
    fname = Path(fpath).stem
    try:
        doc = fitz.open(fpath)
    except Exception as e:
        print(f"파일 로드 실패 ({fname}): {e}")
        return fname, {}

    page_count = config.page_count
    f_pages: dict[int, list[bytes]] = {i: [] for i in range(page_count)}
    survey_count = _survey_count(len(doc), page_count)
    limit = min(survey_count, sample_limit)

    # 설문 내 페이지 병렬 처리를 위한 inner pool
    inner_workers = min(4, page_count)
    inner_pool = ThreadPoolExecutor(max_workers=inner_workers) if inner_workers > 1 else None

    try:
        for survey_idx in range(limit):
            pages = _render_survey_pages(
                doc, survey_idx, page_count, aligners,
                config.rot_code, config.fine_angle, dpi, inner_pool,
            )
            for local_p, aligned in pages.items():
                if len(f_pages[local_p]) < sample_limit:
                    f_pages[local_p].append(cv2.imencode(".png", aligned)[1].tobytes())
    finally:
        if inner_pool:
            inner_pool.shutdown(wait=False)
        doc.close()

    return fname, f_pages


# ── Phase 2 Worker: 파일 1개 전체 분석 (스레드 안전) ──
def _analyze_single_file(
    fpath: str,
    config: TemplatePreset,
    file_templates: dict[str, dict[int, np.ndarray]],
    reference_templates: dict[int, np.ndarray],
    aligners: list,
    review_folder: Path,
    dpi: int = 300,
) -> tuple[str, list[dict], dict[int, np.ndarray]]:
    fname = Path(fpath).stem
    try:
        doc = fitz.open(fpath)
    except Exception as e:
        print(f"파일 로드 실패 ({fname}): {e}")
        return fname, [], {}

    survey_count = _survey_count(len(doc), config.page_count)
    f_template = file_templates.get(fname, reference_templates)
    page_count = config.page_count
    rot_code = config.rot_code
    fine_angle = config.fine_angle

    out_orig = fitz.open()
    out_ink = fitz.open()
    file_results: list[dict] = []
    comment_pages: list[np.ndarray] = []

    # 설문 내 페이지 병렬 처리를 위한 inner pool
    inner_workers = min(4, page_count)
    inner_pool = ThreadPoolExecutor(max_workers=inner_workers) if inner_workers > 1 else None

    try:
        for survey_idx in range(survey_count):
            survey_gray_pages = _render_survey_pages(
                doc, survey_idx, page_count, aligners,
                rot_code, fine_angle, dpi, inner_pool,
            )

            survey_data = {
                "fname": fname,
                "row_title": f"{fname}_{survey_idx + 1}p",
                "gray_pages": survey_gray_pages,
            }

            row_data, debug_base, ink_base, debug_ann, ink_ann, cp = (
                process_survey_data(survey_data, config, f_template)
            )

            field_values = [
                v for k, v in row_data.items() if k not in ("파일명", "페이지")
            ]
            if any(v.strip() for v in field_values):
                file_results.append(row_data)

            for local_p in sorted(debug_base.keys()):
                _build_vector_page(
                    out_orig, debug_base[local_p], debug_ann.get(local_p, [])
                )
            for local_p in sorted(ink_base.keys()):
                _build_vector_page(
                    out_ink, ink_base[local_p], ink_ann.get(local_p, [])
                )
            for local_p in sorted(cp.keys()):
                comment_pages.append(cp[local_p])

            del row_data, debug_base, ink_base, debug_ann, ink_ann, cp
            del survey_gray_pages, survey_data
            gc.collect()
    finally:
        if inner_pool:
            inner_pool.shutdown(wait=False)
        doc.close()

    # 파일별 출력 PDF 저장 + 해제
    if len(out_orig) > 0:
        out_orig.save(review_folder / f"{fname}_원본포함.pdf")
    out_orig.close()
    if len(out_ink) > 0:
        out_ink.save(review_folder / f"{fname}_잉크추출.pdf")
    out_ink.close()

    return fname, file_results, comment_pages


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

    # ── 전체 설문 개수 파악 (진행률 표시용) ──
    total_surveys = 0
    for fpath in file_paths:
        try:
            doc = fitz.open(fpath)
            total_surveys += _survey_count(len(doc), config.page_count)
            doc.close()
        except Exception:
            continue
    if total_surveys <= 0:
        total_surveys = 1

    num_files = len(file_paths)
    if num_files == 0:
        return False

    # ── 페이지 정합기 (모든 스레드에서 읽기 전용으로 공유) ──
    aligners = [
        ImageAligner(apply_rotation(p, config.rot_code, config.fine_angle))
        for p in template_pages[: config.page_count]
    ]

    # ══════════════════════════════════════
    # Phase 1: 템플릿 샘플 병렬 수집
    # ══════════════════════════════════════
    report_progress(0, "템플릿 샘플 수집 중...")
    sample_results: dict[str, dict[int, list[bytes]]] = {}
    max_workers = min(8, num_files)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _collect_template_samples, fpath, config, aligners
            ): fpath
            for fpath in file_paths
        }
        completed = 0
        for future in as_completed(futures):
            fname, f_pages = future.result()
            if f_pages:
                sample_results[fname] = f_pages
            completed += 1
            report_progress(
                int(completed / num_files * 8),
                f"템플릿 샘플 수집 중... ({completed}/{num_files})",
            )

    # ── 템플릿 빌드 + 정렬 (원본 순서 유지) ──
    report_progress(8, "템플릿 생성 중...")
    reference_templates: dict[int, np.ndarray] | None = None
    file_templates: dict[str, dict[int, np.ndarray]] = {}

    for fpath in file_paths:
        fname = Path(fpath).stem
        f_pages = sample_results.get(fname, {})
        if not f_pages:
            file_templates[fname] = {}
            continue

        file_template = generate_dynamic_templates(f_pages)
        if not file_template:
            file_templates[fname] = {}
            continue

        if reference_templates is None:
            reference_templates = file_template
            file_templates[fname] = file_template
        else:
            aligned_tpl: dict[int, np.ndarray] = {}
            for local_p, tpl in file_template.items():
                if local_p in reference_templates:
                    a = ImageAligner(reference_templates[local_p])
                    aligned_tpl[local_p] = a.align(tpl)
                else:
                    aligned_tpl[local_p] = tpl
            file_templates[fname] = aligned_tpl

    del sample_results
    gc.collect()

    if reference_templates is None:
        print("템플릿 생성에 실패했습니다.")
        return False

    for fname in file_templates:
        if not file_templates[fname]:
            file_templates[fname] = reference_templates

    # ── 검토용 템플릿 PDF 저장 ──
    template_pdf = fitz.open()
    for local_p in sorted(reference_templates.keys()):
        _insert_img_into_pdf(template_pdf, reference_templates[local_p], quality=90)
    if len(template_pdf) > 0:
        template_pdf.save(review_folder / "00_추론된_템플릿.pdf")
    template_pdf.close()

    # ══════════════════════════════════════
    # Phase 2: 파일별 분석 병렬 처리
    # ══════════════════════════════════════
    report_progress(10, "분석 시작 중...")
    all_results: list[dict] = []
    all_comment_pages: list[np.ndarray] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _analyze_single_file,
                fpath,
                config,
                file_templates,
                reference_templates,
                aligners,
                review_folder,
            ): fpath
            for fpath in file_paths
        }
        completed = 0
        for future in as_completed(futures):
            try:
                fname, file_results, cp_list = future.result()
                all_results.extend(file_results)
                if cp_list:
                    all_comment_pages.extend(cp_list)
            except Exception as e:
                print(f"분석 실패: {e}")
            completed += 1
            report_progress(
                10 + int(completed / num_files * 88),
                f"분석 중... ({completed}/{num_files})",
            )

    # ── 의견 PDF 병합 저장 ──
    comment_doc = fitz.open()
    for img in all_comment_pages:
        _insert_img_into_pdf(comment_doc, img)
    if len(comment_doc) > 0:
        comment_doc.save("의견.pdf")
    comment_doc.close()
    del all_comment_pages

    # ── 엑셀 저장 ──
    report_progress(98, "엑셀 저장 중...")
    success = export_to_excel(all_results, config)
    report_progress(100, "완료")
    return success
