"""Checkbox alignment, ink extraction, and mark scoring algorithms."""

import copy
from dataclasses import dataclass

import cv2
import numpy as np

from .models import Box, Field, TemplatePreset


_CHECKBOX_MIN_BORDER_CONFIDENCE = 0.35
_CANCEL_MARK_MIN_FILL_RATIO = 0.05
_CANCEL_MARK_MIN_BBOX_DENSITY = 0.22
_CANCEL_MARK_MIN_INK_RATIO = 2.5
_RUNNER_UP_MIN_FILL_RATIO = 0.01
_RUNNER_UP_MAX_BBOX_DENSITY = 0.20
_CHECKBOX_MIN_DIRECT_EVIDENCE_PIXELS = 5
_SHARED_STROKE_MIN_LOCAL_SUPPORT_RATIO = 0.45
_SHARED_STROKE_MIN_SHAPE_SPREAD = 0.18
_FULL_LOCAL_SEARCH_RADIUS = 2
_CHECKBOX_CANCEL_MIN_COMBINED_RATIO = 1.75
_CHECKBOX_CANCEL_MIN_RUNNER_INK = 30
_CHECKBOX_CANCEL_MIN_SHAPE_SPREAD = 0.35
_CHECKBOX_RELIABLE_MARK_STRENGTH = 0.025


def _expand_comment_box(box: Box) -> Box:
    """Cover handwriting that strays around a configured free-text line.

    Comment boxes in older presets usually describe only the printed answer
    line. Respondents commonly start beside the ``답:`` label or continue on
    the whitespace below it, so use a deliberately asymmetric, bounded
    corridor around that line.
    """
    if box.w < 900:
        return Box(box.page_idx, box.x, box.y, box.w, box.h)

    left = min(240, round(box.w * 0.22))
    right = min(90, round(box.w * 0.08))
    top = min(35, round(box.h * 0.25))
    bottom = min(210, round(box.h * 1.80))
    return Box(
        page_idx=box.page_idx,
        x=max(0, box.x - left),
        y=max(0, box.y - top),
        w=box.w + left + right,
        h=box.h + top + bottom,
    )


@dataclass
class _CheckboxInkInfo:
    ink_pixels: int
    area: int
    box: Box
    border_confidence: float
    mask_bounds: tuple[int, int, int, int]
    ink_mask: np.ndarray
    mark_strength: float = 0.0
    residual_pixels: int = 0
    residual_energy: float = 0.0
    stroke_span_ratio: float = 0.0


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
    direct_strengths: list[float]
    direct_results: list[bool]
    halo_infos: list[_CheckboxHaloInfo]


def _resize_binary_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.threshold(resized, 32, 255, cv2.THRESH_BINARY)[1]


def _rotate_binary_mask(mask: np.ndarray, angle: float) -> np.ndarray:
    height, width = mask.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


@dataclass(frozen=True)
class _TemplateAlignmentCache:
    """Immutable template-side preprocessing reused within one analysis."""

    template_mask: np.ndarray
    search_template_mask: np.ndarray
    alignment_mask: np.ndarray | None
    template_pixels: int
    angles: tuple[float, ...]
    search_scale: float
    search_size: tuple[int, int]
    small_template_pixels: int
    coarse_rotations: dict[float, np.ndarray]
    fine_search_scale: float
    fine_search_size: tuple[int, int]
    fine_template: np.ndarray
    fine_template_pixels: int


def _prepare_template_alignment_cache(
    template_mask: np.ndarray,
    image_shape: tuple[int, ...],
    max_angle: float = 0.6,
    alignment_mask: np.ndarray | None = None,
) -> _TemplateAlignmentCache:
    height, width = image_shape[:2]
    if template_mask.shape[:2] != (height, width):
        template_mask = cv2.resize(
            template_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    prepared_alignment_mask = None
    search_template_mask = template_mask
    if alignment_mask is not None and alignment_mask.size > 0:
        prepared_alignment_mask = alignment_mask
        if prepared_alignment_mask.ndim == 3:
            prepared_alignment_mask = cv2.cvtColor(
                prepared_alignment_mask, cv2.COLOR_BGR2GRAY
            )
        if prepared_alignment_mask.shape[:2] != (height, width):
            prepared_alignment_mask = cv2.resize(
                prepared_alignment_mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        prepared_alignment_mask = np.where(
            prepared_alignment_mask > 0, 255, 0
        ).astype(np.uint8)
        if cv2.countNonZero(prepared_alignment_mask) >= height * width * 0.15:
            search_template_mask = cv2.bitwise_and(
                template_mask, prepared_alignment_mask
            )
        else:
            prepared_alignment_mask = None

    angle_count = max(0, round(max_angle / 0.1))
    angles = tuple(step * 0.1 for step in range(-angle_count, angle_count + 1))
    search_scale = min(0.5, 900.0 / max(height, width))
    search_size = (
        max(1, round(width * search_scale)),
        max(1, round(height * search_scale)),
    )
    small_template = _resize_binary_mask(search_template_mask, *search_size)

    fine_search_scale = max(
        search_scale, min(0.75, 2000.0 / max(height, width))
    )
    fine_search_size = (
        max(1, round(width * fine_search_scale)),
        max(1, round(height * fine_search_scale)),
    )
    fine_template = (
        small_template
        if fine_search_scale == search_scale
        else _resize_binary_mask(search_template_mask, *fine_search_size)
    )

    return _TemplateAlignmentCache(
        template_mask=template_mask,
        search_template_mask=search_template_mask,
        alignment_mask=prepared_alignment_mask,
        template_pixels=cv2.countNonZero(search_template_mask),
        angles=angles,
        search_scale=search_scale,
        search_size=search_size,
        small_template_pixels=cv2.countNonZero(small_template),
        coarse_rotations={
            angle: _rotate_binary_mask(small_template, angle)
            for angle in angles
        },
        fine_search_scale=fine_search_scale,
        fine_search_size=fine_search_size,
        fine_template=fine_template,
        fine_template_pixels=cv2.countNonZero(fine_template),
    )


def _best_shift_by_correlation(
    template_mask: np.ndarray,
    padded_target: np.ndarray,
    max_shift: int,
    reference_pixels: int,
) -> tuple[float, int, int, int]:
    """제한된 이동 범위의 모든 겹침을 OpenCV 상관맵 한 번으로 계산합니다."""
    candidate_pixels = cv2.countNonZero(template_mask)
    if candidate_pixels <= 0:
        return float("-inf"), 0, 0, 0

    x, y, width, height = cv2.boundingRect(template_mask)
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


def _best_shift_by_local_binary_overlap(
    template_mask: np.ndarray,
    padded_target: np.ndarray,
    max_shift: int,
    reference_pixels: int,
    predicted_shift: tuple[int, int],
    radius: int = _FULL_LOCAL_SEARCH_RADIUS,
) -> tuple[tuple[float, int, int, int], bool]:
    """Search a small binary window and report whether its winner is safe."""
    candidate_pixels = cv2.countNonZero(template_mask)
    if candidate_pixels <= 0:
        return (float("-inf"), 0, 0, 0), False

    predicted_x = max(-max_shift, min(max_shift, predicted_shift[0]))
    predicted_y = max(-max_shift, min(max_shift, predicted_shift[1]))
    min_dx = max(-max_shift, predicted_x - radius)
    max_dx = min(max_shift, predicted_x + radius)
    min_dy = max(-max_shift, predicted_y - radius)
    max_dy = min(max_shift, predicted_y + radius)

    x, y, width, height = cv2.boundingRect(template_mask)
    template_roi = np.ascontiguousarray(
        template_mask[y : y + height, x : x + width]
    )
    overlap_mask = np.empty_like(template_roi)
    best_overlap = -1
    runner_up_overlap = -1
    best_dx = predicted_x
    best_dy = predicted_y

    for dy in range(min_dy, max_dy + 1):
        target_y = y + max_shift + dy
        for dx in range(min_dx, max_dx + 1):
            target_x = x + max_shift + dx
            target_roi = padded_target[
                target_y : target_y + height,
                target_x : target_x + width,
            ]
            cv2.bitwise_and(template_roi, target_roi, dst=overlap_mask)
            overlap = cv2.countNonZero(overlap_mask)
            if overlap > best_overlap:
                runner_up_overlap = best_overlap
                best_overlap = overlap
                best_dx = dx
                best_dy = dy
            elif overlap > runner_up_overlap:
                runner_up_overlap = overlap

    penalty = abs(candidate_pixels - reference_pixels) * 0.2
    score = float(best_overlap) - penalty
    runner_up_score = (
        float(runner_up_overlap) - penalty
        if runner_up_overlap >= 0
        else float("-inf")
    )
    touches_unsearched_edge = (
        (best_dx == min_dx and min_dx > -max_shift)
        or (best_dx == max_dx and max_dx < max_shift)
        or (best_dy == min_dy and min_dy > -max_shift)
        or (best_dy == max_dy and max_dy < max_shift)
    )
    is_confident = (
        not touches_unsearched_edge
        and not _correlation_choice_is_ambiguous(
            score,
            runner_up_score,
            reference_pixels,
        )
    )
    return (score, best_overlap, best_dx, best_dy), is_confident


def _best_shift_with_safe_local_fallback(
    template_mask: np.ndarray,
    padded_target: np.ndarray,
    max_shift: int,
    reference_pixels: int,
    predicted_shift: tuple[int, int],
) -> tuple[float, int, int, int]:
    local_result, is_confident = _best_shift_by_local_binary_overlap(
        template_mask,
        padded_target,
        max_shift,
        reference_pixels,
        predicted_shift,
    )
    if is_confident:
        return local_result
    return _best_shift_by_correlation(
        template_mask,
        padded_target,
        max_shift,
        reference_pixels,
    )


def _correlation_choice_is_ambiguous(
    best_score: float,
    runner_up_score: float,
    reference_pixels: int,
) -> bool:
    margin = max(4, round(reference_pixels * 0.0002))
    return best_score - runner_up_score < margin


def _scaled_shift_is_consistent(
    full_shift: tuple[int, int],
    small_shift: tuple[int, int],
    search_scale: float,
) -> bool:
    tolerance = max(2, int(np.ceil(0.75 / search_scale)))
    predicted_full = (
        round(small_shift[0] / search_scale),
        round(small_shift[1] / search_scale),
    )
    return (
        abs(full_shift[0] - predicted_full[0]) <= tolerance
        and abs(full_shift[1] - predicted_full[1]) <= tolerance
    )


def _build_stable_region_mask(
    image_shape: tuple[int, ...],
    config: TemplatePreset,
    page_idx: int,
) -> np.ndarray | None:
    """Keep printed form regions while excluding every user response region."""
    height, width = image_shape[:2]
    if height <= 0 or width <= 0:
        return None

    mask = np.full((height, width), 255, np.uint8)
    excluded_any = False
    for field in config.fields:
        for box in field.boxes:
            if box.page_idx != page_idx or box.w <= 0 or box.h <= 0:
                continue
            if _is_checkbox_like(box, image_shape):
                padding = max(12, round(min(box.w, box.h) * 0.9))
            else:
                padding = max(8, round(min(box.w, box.h) * 0.08))
            x1 = max(0, box.x - padding)
            y1 = max(0, box.y - padding)
            x2 = min(width, box.x + box.w + padding)
            y2 = min(height, box.y + box.h + padding)
            if x2 <= x1 or y2 <= y1:
                continue
            mask[y1:y2, x1:x2] = 0
            excluded_any = True

    if not excluded_any:
        return None
    if cv2.countNonZero(mask) < height * width * 0.15:
        return None
    return mask


def _align_template_mask_by_coverage(
    template_mask: np.ndarray,
    target_mask: np.ndarray,
    max_angle: float = 0.6,
    max_shift: int = 8,
    alignment_mask: np.ndarray | None = None,
    template_cache: _TemplateAlignmentCache | None = None,
) -> np.ndarray:
    """템플릿 선이 대상의 어두운 픽셀을 가장 많이 덮도록 미세 정합합니다.

    체크박스 생성용 페이지 정합과는 완전히 분리된 후처리입니다. 이미 ORB/ECC로
    정합된 페이지의 잔여 오차만 보정하므로 탐색 범위를 작게 제한합니다.
    """
    h, w = target_mask.shape[:2]
    if template_cache is None:
        template_cache = _prepare_template_alignment_cache(
            template_mask,
            target_mask.shape,
            max_angle,
            alignment_mask,
        )

    template_mask = template_cache.template_mask
    search_template_mask = template_cache.search_template_mask
    search_target_mask = target_mask
    if template_cache.alignment_mask is not None:
        search_target_mask = cv2.bitwise_and(
            target_mask, template_cache.alignment_mask
        )

    template_pixels = template_cache.template_pixels
    target_pixels = cv2.countNonZero(search_target_mask)
    if template_pixels < 32 or target_pixels < 32:
        return template_mask

    identity_overlap = cv2.countNonZero(
        cv2.bitwise_and(search_template_mask, search_target_mask)
    )
    identity_score = float(identity_overlap)
    # 1~2px 확장은 뒤에서 적용되므로 97% 이상 맞으면 추가 탐색의 이득이 없습니다.
    if identity_overlap >= template_pixels * 0.97:
        return template_mask

    # 긴 변을 최대 900px로 줄여 각도와 대략적인 이동량을 빠르게 찾습니다.
    search_scale = template_cache.search_scale
    search_w, search_h = template_cache.search_size
    # INTER_AREA로 축소한 뒤 낮은 임계값으로 다시 이진화해 가는 선의 소실을 줄입니다.
    small_template_pixels = template_cache.small_template_pixels
    small_target = _resize_binary_mask(search_target_mask, search_w, search_h)
    if small_template_pixels == 0:
        return template_mask

    angles = template_cache.angles
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
    best_coarse_key = (float("-inf"), 0, float("-inf"))
    best_coarse = (0.0, 0, 0)

    for angle in angles:
        rotated = template_cache.coarse_rotations[angle]
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
        search_target_mask,
        max_shift,
        max_shift,
        max_shift,
        max_shift,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    # Resolve the three half-step angles on an intermediate-size mask: 900px is
    # sufficient for the 0.1-degree coarse sweep but can quantize away a
    # 0.05-degree difference.  Only the strongest candidate is then verified at
    # full resolution; ambiguous candidates retain the exhaustive path.
    fine_search_scale = template_cache.fine_search_scale
    if fine_search_scale > search_scale:
        fine_search_w, fine_search_h = template_cache.fine_search_size
        fine_target = _resize_binary_mask(
            search_target_mask, fine_search_w, fine_search_h
        )
    else:
        fine_search_scale = search_scale
        fine_search_w, fine_search_h = search_w, search_h
        fine_target = small_target

    fine_template_pixels = template_cache.fine_template_pixels
    fine_shift = max(0, int(np.ceil(max_shift * fine_search_scale)))
    fine_padded_target = cv2.copyMakeBorder(
        fine_target,
        fine_shift,
        fine_shift,
        fine_shift,
        fine_shift,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    fine_results: list[
        tuple[tuple[float, int, float], float, int, int]
    ] = []
    for angle in sorted(fine_angles):
        rotated = _rotate_binary_mask(template_cache.fine_template, angle)
        score, overlap, dx, dy = _best_shift_by_correlation(
            rotated, fine_padded_target, fine_shift, fine_template_pixels
        )
        motion = abs(angle) + abs(dx) + abs(dy)
        fine_results.append(((score, overlap, -motion), angle, dx, dy))

    ranked_fine = sorted(
        fine_results,
        key=lambda result: result[0],
        reverse=True,
    )
    best_fine = ranked_fine[0]
    needs_full_sweep = (
        len(ranked_fine) > 1
        and _correlation_choice_is_ambiguous(
            best_fine[0][0], ranked_fine[1][0][0], fine_template_pixels
        )
    )
    fine_shifts = {
        result[1]: (result[2], result[3]) for result in fine_results
    }

    full_results: dict[float, tuple[tuple[float, int, float], float, int, int]] = {}

    def evaluate_full(angle: float):
        rotated = _rotate_binary_mask(search_template_mask, angle)
        fine_dx, fine_dy = fine_shifts[angle]
        predicted_shift = (
            round(fine_dx / fine_search_scale),
            round(fine_dy / fine_search_scale),
        )
        score, overlap, dx, dy = _best_shift_with_safe_local_fallback(
            rotated,
            full_padded_target,
            max_shift,
            template_pixels,
            predicted_shift,
        )
        motion = abs(angle) + abs(dx) + abs(dy)
        result = ((score, overlap, -motion), angle, dx, dy)
        full_results[angle] = result
        return result

    if not needs_full_sweep:
        verified = evaluate_full(best_fine[1])
        if not _scaled_shift_is_consistent(
            (verified[2], verified[3]),
            (best_fine[2], best_fine[3]),
            fine_search_scale,
        ):
            needs_full_sweep = True

    if needs_full_sweep:
        for angle in sorted(fine_angles):
            if angle not in full_results:
                evaluate_full(angle)

    if full_results:
        for angle in sorted(fine_angles):
            result = full_results.get(angle)
            if result is not None and result[0] > best_key:
                best_key = result[0]
                best_transform = (result[1], result[2], result[3])

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
    alignment_mask: np.ndarray | None = None,
    template_alignment_cache: _TemplateAlignmentCache | None = None,
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
    template_mask = _align_template_mask_by_coverage(
        template_mask,
        target_mask,
        alignment_mask=alignment_mask,
        template_cache=template_alignment_cache,
    )

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


def _comment_region_has_meaningful_ink(
    pure_ink_mask: np.ndarray,
    box: Box,
) -> bool:
    """Reject isolated scan residue while retaining fragmented handwriting."""
    image_h, image_w = pure_ink_mask.shape[:2]
    x1 = max(0, box.x)
    y1 = max(0, box.y)
    x2 = min(image_w, box.x + box.w)
    y2 = min(image_h, box.y + box.h)
    if x2 <= x1 or y2 <= y1:
        return False

    roi = (pure_ink_mask[y1:y2, x1:x2] > 0).astype(np.uint8)
    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        roi, connectivity=8
    )
    compact_component_count = 0
    compact_ink_pixels = 0
    for component_idx in range(1, component_count):
        width = int(stats[component_idx, cv2.CC_STAT_WIDTH])
        height = int(stats[component_idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_idx, cv2.CC_STAT_AREA])
        span = max(width, height)
        density = area / max(1, width * height)
        aspect = span / max(1, min(width, height))
        if area < 12:
            continue
        if aspect >= 6.0 and density <= 0.20:
            continue
        if span <= 32 and density >= 0.45:
            compact_component_count += 1
            compact_ink_pixels += area
            continue
        return True

    # Template subtraction can split small handwritten Korean glyphs into
    # several dense islands. One or two such islands are commonly scan dust;
    # a sufficiently strong cluster is a real free-text response.
    return compact_component_count >= 3 and compact_ink_pixels >= 300


def _comment_has_overprint_evidence(
    target_gray: np.ndarray,
    template_gray: np.ndarray,
    box: Box,
) -> bool:
    """Detect a localized pen stroke drawn over the printed answer label.

    Binary template subtraction intentionally removes printed glyphs and can
    also hide a response written directly on top of them. For long free-text
    lines, compare darkness in a narrow label window with its taller local
    context. The absolute energy guard prevents ordinary registration residue
    from being promoted to a response.
    """
    if box.w < 900 or target_gray.size == 0 or template_gray.size == 0:
        return False
    if target_gray.ndim == 3:
        target_gray = cv2.cvtColor(target_gray, cv2.COLOR_BGR2GRAY)
    if template_gray.ndim == 3:
        template_gray = cv2.cvtColor(template_gray, cv2.COLOR_BGR2GRAY)
    if template_gray.shape != target_gray.shape:
        template_gray = cv2.resize(
            template_gray,
            (target_gray.shape[1], target_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    def darkness_energy(
        x1: int, y1: int, x2: int, y2: int
    ) -> tuple[int, int]:
        image_h, image_w = target_gray.shape[:2]
        x1 = max(0, min(image_w, x1))
        y1 = max(0, min(image_h, y1))
        x2 = max(x1, min(image_w, x2))
        y2 = max(y1, min(image_h, y2))
        if x2 <= x1 or y2 <= y1:
            return 0, 0
        delta = (
            template_gray[y1:y2, x1:x2].astype(np.int16)
            - target_gray[y1:y2, x1:x2].astype(np.int16)
        )
        darker = delta >= 20
        return int(delta[darker].sum()), int(delta.size)

    context_energy, _ = darkness_energy(
        box.x - round(box.w * 0.018),
        box.y - round(box.h * 0.18),
        box.x + round(box.w * 0.036),
        box.y + round(box.h * 1.18),
    )
    focus_energy, focus_area = darkness_energy(
        box.x - round(box.w * 0.026),
        box.y + round(box.h * 0.20),
        box.x + round(box.w * 0.046),
        box.y + round(box.h * 0.98),
    )
    minimum_focus_energy = round(focus_area * 1.45)
    return (
        context_energy > 0
        and focus_energy >= minimum_focus_energy
        and focus_energy >= context_energy * 0.80
    )


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


def _checkbox_box_key(box: Box) -> tuple[int, int, int, int, int]:
    return (box.page_idx, box.x, box.y, box.w, box.h)


def _prepare_checkbox_template_interiors(
    field_plans: list[tuple[Field, list[Box], list[Box], bool]],
    dynamic_templates: dict[int, np.ndarray],
    page_shapes: dict[int, tuple[int, ...]],
) -> dict[tuple[int, int, int, int, int], np.ndarray]:
    prepared: dict[tuple[int, int, int, int, int], np.ndarray] = {}
    gray_templates: dict[int, np.ndarray] = {}
    for field, scoring_boxes, _working_boxes, _is_contiguous in field_plans:
        if (
            field.is_comment
            or not scoring_boxes
            or not all(
                box.page_idx in page_shapes
                and _is_checkbox_like(box, page_shapes[box.page_idx])
                for box in scoring_boxes
            )
        ):
            continue
        for box in scoring_boxes:
            key = _checkbox_box_key(box)
            if key in prepared or box.page_idx not in dynamic_templates:
                continue

            template_gray = gray_templates.get(box.page_idx)
            if template_gray is None:
                template = dynamic_templates[box.page_idx]
                template_gray = (
                    cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    if template.ndim == 3
                    else template
                )
                target_shape = page_shapes.get(box.page_idx)
                if (
                    target_shape is not None
                    and template_gray.shape[:2] != target_shape[:2]
                ):
                    template_gray = cv2.resize(
                        template_gray,
                        (target_shape[1], target_shape[0]),
                        interpolation=cv2.INTER_AREA,
                    )
                gray_templates[box.page_idx] = template_gray

            template_box, _ = _refine_checkbox_box(template_gray, box)
            short_side = min(template_box.w, template_box.h)
            border_width = max(1, round(short_side * 0.1))
            margin = max(border_width + 1, round(short_side * 0.2))
            x1 = max(0, template_box.x + margin)
            y1 = max(0, template_box.y + margin)
            x2 = min(
                template_gray.shape[1], template_box.x + template_box.w - margin
            )
            y2 = min(
                template_gray.shape[0], template_box.y + template_box.h - margin
            )
            prepared[key] = (
                template_gray[y1:y2, x1:x2]
                if x2 > x1 and y2 > y1
                else template_gray[0:0, 0:0]
            )
    return prepared


def _filter_checkbox_mark_components(
    binary_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Keep pen-like components and reject compact scanner dust."""
    filtered = np.zeros_like(binary_mask, dtype=np.uint8)
    if binary_mask.size == 0:
        return filtered, 0.0

    binary = (binary_mask > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    height, width = binary.shape[:2]
    short_side = max(1, min(height, width))
    min_component_area = max(2, round(binary.size * 0.008))
    min_component_span = max(3, round(short_side * 0.25))
    dense_component_area = max(5, round(binary.size * 0.04))
    max_span_ratio = 0.0
    for component_idx in range(1, component_count):
        component_area = int(stats[component_idx, cv2.CC_STAT_AREA])
        component_width = int(stats[component_idx, cv2.CC_STAT_WIDTH])
        component_height = int(stats[component_idx, cv2.CC_STAT_HEIGHT])
        component_span = max(component_width, component_height)
        if component_area < min_component_area or (
            component_span < min_component_span
            and component_area < dense_component_area
        ):
            continue
        filtered[labels == component_idx] = 255
        max_span_ratio = max(
            max_span_ratio,
            component_width / max(1, width),
            component_height / max(1, height),
        )
    return filtered, max_span_ratio


def _checkbox_difference_features(
    target_interior: np.ndarray,
    template_interior: np.ndarray,
) -> tuple[np.ndarray, int, float, float]:
    """Measure locally normalized grayscale ink added after form printing."""
    common_h = min(target_interior.shape[0], template_interior.shape[0])
    common_w = min(target_interior.shape[1], template_interior.shape[1])
    if common_h <= 0 or common_w <= 0:
        return np.zeros((0, 0), np.uint8), 0, 0.0, 0.0

    target = target_interior[:common_h, :common_w].astype(np.float32)
    template = template_interior[:common_h, :common_w].astype(np.float32)
    target_paper = float(np.percentile(target, 90))
    template_paper = float(np.percentile(template, 90))
    target += template_paper - target_paper
    np.clip(target, 0.0, 255.0, out=target)

    residual = template - target
    baseline = float(np.median(residual))
    deviation = float(np.median(np.abs(residual - baseline)))
    noise_sigma = 1.4826 * deviation
    threshold = max(10.0, baseline + noise_sigma * 4.0)
    candidate = (residual >= threshold).astype(np.uint8) * 255
    filtered, span_ratio = _filter_checkbox_mark_components(candidate)
    residual_pixels = cv2.countNonZero(filtered)
    if residual_pixels <= 0:
        return filtered, 0, 0.0, span_ratio

    excess = np.maximum(residual - threshold, 0.0)
    residual_energy = float(np.sum(excess[filtered > 0]))
    return filtered, residual_pixels, residual_energy, span_ratio


def extract_checkbox_ink_info(
    target_gray: np.ndarray,
    box: Box,
    template_gray: np.ndarray | None = None,
    template_interior: np.ndarray | None = None,
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
    raw_mask = (interior < ink_threshold).astype(np.uint8) * 255
    component_mask, raw_span_ratio = _filter_checkbox_mark_components(raw_mask)

    residual_pixels = 0
    residual_energy = 0.0
    residual_span_ratio = 0.0
    reference_interior = template_interior
    if (
        reference_interior is None
        and template_gray is not None
        and template_gray.size > 0
    ):
        if template_gray.ndim == 3:
            template_gray = cv2.cvtColor(template_gray, cv2.COLOR_BGR2GRAY)
        if template_gray.shape[:2] != target_gray.shape[:2]:
            template_gray = cv2.resize(
                template_gray,
                (target_gray.shape[1], target_gray.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        template_box, _ = _refine_checkbox_box(template_gray, box)
        template_short_side = min(template_box.w, template_box.h)
        template_border_width = max(1, round(template_short_side * 0.1))
        template_margin = max(
            template_border_width + 1, round(template_short_side * 0.2)
        )
        tx1 = max(0, template_box.x + template_margin)
        ty1 = max(0, template_box.y + template_margin)
        tx2 = min(
            template_gray.shape[1], template_box.x + template_box.w - template_margin
        )
        ty2 = min(
            template_gray.shape[0], template_box.y + template_box.h - template_margin
        )
        if tx2 > tx1 and ty2 > ty1:
            reference_interior = template_gray[ty1:ty2, tx1:tx2]

    if reference_interior is not None and reference_interior.size > 0:
        difference_mask, residual_pixels, residual_energy, residual_span_ratio = (
            _checkbox_difference_features(
                interior,
                reference_interior,
            )
        )
        if difference_mask.shape == component_mask.shape:
            component_mask = cv2.bitwise_or(component_mask, difference_mask)

    ink_pixels = cv2.countNonZero(component_mask)
    area = int(raw_mask.size)
    raw_fill = ink_pixels / max(1, area)
    residual_fill = residual_pixels / max(1, area)
    residual_energy_ratio = residual_energy / max(1.0, area * 255.0)
    mark_strength = max(
        raw_fill,
        residual_fill + residual_energy_ratio * 0.75,
    )

    return _CheckboxInkInfo(
        ink_pixels,
        area,
        refined,
        confidence,
        (x1, y1, x2, y2),
        component_mask,
        mark_strength,
        residual_pixels,
        residual_energy,
        max(raw_span_ratio, residual_span_ratio),
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
    component_count, connected_labels, _, _ = cv2.connectedComponentsWithStats(
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
    anchored_labels = set(
        int(value) for value in np.unique(connected_labels[anchor > 0])
    )
    anchored_labels.discard(0)
    anchored = np.zeros_like(candidate)
    for component_idx in anchored_labels:
        anchored[(connected_labels == component_idx) & (candidate > 0)] = 255

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
    interior_connected_labels: set[int] = set()
    if inner_x2 > inner_x1 and inner_y2 > inner_y1:
        border_mask[inner_y1:inner_y2, inner_x1:inner_x2] = 0
        interior_pixels = candidate[inner_y1:inner_y2, inner_x1:inner_x2] > 0
        if np.any(interior_pixels):
            interior_connected_labels = set(
                int(value)
                for value in np.unique(
                    connected_labels[inner_y1:inner_y2, inner_x1:inner_x2][
                        interior_pixels
                    ]
                )
            )
            interior_connected_labels.discard(0)
    anchored[border_mask > 0] = 0
    anchored[
        max(0, local_y) : min(anchored.shape[0], local_y2),
        max(0, local_x) : min(anchored.shape[1], local_x2),
    ] = 0

    filtered = np.zeros_like(anchored)
    component_count, filtered_labels, stats, _ = cv2.connectedComponentsWithStats(
        (anchored > 0).astype(np.uint8), connectivity=8
    )
    min_component_area = max(4, round(short_side * 0.15))
    min_component_span = max(4, round(short_side * 0.2))
    for component_idx in range(1, component_count):
        component_area = stats[component_idx, cv2.CC_STAT_AREA]
        component_width = stats[component_idx, cv2.CC_STAT_WIDTH]
        component_height = stats[component_idx, cv2.CC_STAT_HEIGHT]
        component_left = stats[component_idx, cv2.CC_STAT_LEFT]
        component_top = stats[component_idx, cv2.CC_STAT_TOP]
        component_right = component_left + component_width
        component_bottom = component_top + component_height
        component_fill = component_area / max(
            1, component_width * component_height
        )
        source_component_labels = set(
            int(value)
            for value in np.unique(
                connected_labels[filtered_labels == component_idx]
            )
        )
        source_component_labels.discard(0)
        connected_to_interior = bool(
            source_component_labels & interior_connected_labels
        )
        overlaps_box_x = component_left < local_x2 and component_right > local_x
        overlaps_box_y = component_top < local_y2 and component_bottom > local_y
        diagonally_outside = not overlaps_box_x and not overlaps_box_y
        compact_speck = (
            component_fill >= 0.65
            and max(component_width, component_height) <= short_side * 0.5
            and not connected_to_interior
            and diagonally_outside
        )
        # 대각선 모서리 밖에 고립된 작고 조밀한 얼룩만 제거합니다. 실제 큰
        # 체크의 짧은 꼬리는 박스의 위·아래 또는 좌·우 축과 겹칠 수 있습니다.
        if (
            component_area >= min_component_area
            and component_width >= min_component_span
            and component_height >= min_component_span
            and not compact_speck
        ):
            filtered[filtered_labels == component_idx] = 255

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


def _checkbox_mark_shape_spread(
    direct_info: _CheckboxInkInfo,
    halo_info: _CheckboxHaloInfo,
) -> float:
    """Return the minor/major spread ratio of a checkbox's local mark."""
    point_sets = []
    for info in (direct_info, halo_info):
        ys, xs = np.nonzero(info.ink_mask)
        if xs.size <= 0:
            continue
        point_sets.append(
            np.column_stack(
                (xs + info.mask_bounds[0], ys + info.mask_bounds[1])
            )
        )
    if not point_sets:
        return 0.0
    points = np.unique(np.vstack(point_sets), axis=0).astype(np.float32)
    if len(points) < 3:
        return 0.0
    eigenvalues = np.linalg.eigvalsh(np.cov(points, rowvar=False))
    return float(eigenvalues[0] / max(1e-6, eigenvalues[-1]))


def _resolve_checkbox_halo_ownership(
    pure_ink_masks: dict[int, np.ndarray],
    survey_gray_pages: dict[int, np.ndarray],
    candidates: list[tuple[_CheckboxInkInfo, bool, _CheckboxHaloInfo]],
) -> None:
    """Assign one connected pen stroke to its closest supported checkbox.

    A long check drawn in one box can pass through a nearby blank box.  Both
    local windows then see the same connected stroke.  Resolve that ambiguity
    page-wide: prefer the box with stronger local evidence, but retain another
    box when it has both comparable support and a bent, two-dimensional local
    mark of its own.  Only nearby boxes compete, so unrelated marks connected
    through form lines cannot suppress one another across the page.
    """
    by_page: dict[
        int, list[tuple[_CheckboxInkInfo, bool, _CheckboxHaloInfo]]
    ] = {}
    for candidate in candidates:
        direct_info, direct_is_checked, halo_info = candidate
        has_halo = halo_info.ink_pixels > 0 and halo_info.ink_mask.size > 0
        has_strong_direct = (
            direct_is_checked
            and direct_info.ink_pixels >= max(
                8, round(direct_info.area * 0.05)
            )
            and direct_info.ink_mask.size > 0
        )
        if not has_halo and not has_strong_direct:
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

        component_members: dict[int, dict[int, int]] = {}
        halo_component_labels: set[int] = set()

        def add_component_members(
            candidate_idx: int,
            mask_bounds: tuple[int, int, int, int],
            ink_mask: np.ndarray,
            *,
            is_halo: bool,
        ) -> None:
            if ink_mask.size == 0:
                return
            x1, y1, x2, y2 = mask_bounds
            label_roi = labels[y1:y2, x1:x2]
            if label_roi.shape != ink_mask.shape:
                return
            values, counts = np.unique(
                label_roi[ink_mask > 0], return_counts=True
            )
            for value, count in zip(values.tolist(), counts.tolist()):
                component_label = int(value)
                if component_label == 0:
                    continue
                members = component_members.setdefault(component_label, {})
                members[candidate_idx] = members.get(candidate_idx, 0) + int(count)
                if is_halo:
                    halo_component_labels.add(component_label)

        for candidate_idx, (direct_info, direct_is_checked, halo_info) in enumerate(
            page_candidates
        ):
            if halo_info.ink_pixels > 0:
                add_component_members(
                    candidate_idx,
                    halo_info.mask_bounds,
                    halo_info.ink_mask,
                    is_halo=True,
                )
            if direct_is_checked and direct_info.ink_pixels >= max(
                8, round(direct_info.area * 0.05)
            ):
                add_component_members(
                    candidate_idx,
                    direct_info.mask_bounds,
                    direct_info.ink_mask,
                    is_halo=False,
                )

        labels_to_remove: dict[int, set[int]] = {}
        for component_label, counts_by_candidate in component_members.items():
            if (
                component_label not in halo_component_labels
                or len(counts_by_candidate) <= 1
            ):
                continue

            pending = set(counts_by_candidate)
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
                winner_component_support = counts_by_candidate[winner_idx]
                for loser_idx in cluster:
                    loser_box = page_candidates[loser_idx][0].box
                    loser_key = (
                        loser_box.x,
                        loser_box.y,
                        loser_box.w,
                        loser_box.h,
                    )
                    if loser_key != winner_key:
                        _direct_info, direct_is_checked, _halo_info = (
                            page_candidates[loser_idx]
                        )
                        has_own_mark_shape = (
                            direct_is_checked
                            and counts_by_candidate[loser_idx]
                            >= max(
                                12,
                                round(
                                    winner_component_support
                                    * _SHARED_STROKE_MIN_LOCAL_SUPPORT_RATIO
                                ),
                            )
                            and _checkbox_mark_shape_spread(
                                _direct_info, _halo_info
                            )
                            >= _SHARED_STROKE_MIN_SHAPE_SPREAD
                        )
                        if has_own_mark_shape:
                            continue
                        labels_to_remove.setdefault(loser_idx, set()).add(
                            component_label
                        )

        for candidate_idx, component_labels in labels_to_remove.items():
            direct_info = page_candidates[candidate_idx][0]
            direct_x1, direct_y1, direct_x2, direct_y2 = direct_info.mask_bounds
            direct_label_roi = labels[
                direct_y1:direct_y2, direct_x1:direct_x2
            ]
            if direct_label_roi.shape == direct_info.ink_mask.shape:
                direct_remove = np.isin(
                    direct_label_roi, list(component_labels)
                )
                old_direct_ink = direct_info.ink_pixels
                direct_info.ink_mask[direct_remove] = 0
                direct_info.ink_pixels = cv2.countNonZero(
                    direct_info.ink_mask
                )
                if old_direct_ink > 0:
                    remaining_ratio = direct_info.ink_pixels / old_direct_ink
                    direct_info.mark_strength *= remaining_ratio
                    direct_info.residual_pixels = min(
                        direct_info.ink_pixels,
                        round(direct_info.residual_pixels * remaining_ratio),
                    )
                    direct_info.residual_energy *= remaining_ratio
                if direct_info.ink_pixels <= 0:
                    direct_info.stroke_span_ratio = 0.0
                else:
                    points = cv2.findNonZero(direct_info.ink_mask)
                    if points is not None:
                        _x, _y, width, height = cv2.boundingRect(points)
                        direct_info.stroke_span_ratio = max(
                            width / max(1, direct_info.ink_mask.shape[1]),
                            height / max(1, direct_info.ink_mask.shape[0]),
                        )

            halo_info = page_candidates[candidate_idx][2]
            x1, y1, x2, y2 = halo_info.mask_bounds
            label_roi = labels[y1:y2, x1:x2]
            remove = np.isin(label_roi, list(component_labels))
            halo_info.ink_mask[remove] = 0
            halo_info.ink_pixels = cv2.countNonZero(halo_info.ink_mask)


# Mark scoring


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
    inks: list[int],
    areas: list[int],
    strict: bool = False,
    strengths: list[float] | None = None,
) -> list[bool]:
    """인쇄선이 제외된 체크박스 내부 점수를 평가합니다."""
    if not inks:
        return []

    if strengths is not None and len(strengths) == len(inks):
        normalized = [max(0.0, float(value)) for value in strengths]
        if strict:
            # Multiple responses must not be suppressed merely because another
            # option contains a much darker check.
            threshold = 0.025
        elif len(normalized) > 2:
            ordered = sorted(normalized)
            blank_pool = ordered[: max(1, len(ordered) // 2)]
            blank_level = float(np.median(blank_pool)) if blank_pool else 0.0
            threshold = max(0.025, blank_level + 0.012)
        else:
            threshold = 0.025
        minimum_evidence = [
            max(
                _CHECKBOX_MIN_DIRECT_EVIDENCE_PIXELS,
                round(area * 0.025),
            )
            if area > 0
            else _CHECKBOX_MIN_DIRECT_EVIDENCE_PIXELS
            for area in areas
        ]
        return [
            strength >= threshold and ink >= required
            for strength, ink, required in zip(
                normalized, inks, minimum_evidence
            )
        ]

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


def _mark_bbox_density(mask: np.ndarray, box: Box) -> float:
    image_h, image_w = mask.shape[:2]
    x1 = max(0, box.x)
    y1 = max(0, box.y)
    x2 = min(image_w, box.x + box.w)
    y2 = min(image_h, box.y + box.h)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = mask[y1:y2, x1:x2]
    ink_pixels = cv2.countNonZero(roi)
    if ink_pixels <= 0:
        return 0.0
    _x, _y, width, height = cv2.boundingRect(roi)
    return ink_pixels / max(1, width * height)


def _cancellation_runner_up_index(
    inks: list[int],
    areas: list[int],
    boxes: list[Box],
    pure_ink_masks: dict[int, np.ndarray],
) -> int | None:
    """Return the intended runner-up when a dense correction obscures it."""
    if not inks or not (len(inks) == len(areas) == len(boxes)):
        return None

    fill_ratios = [
        ink / area if area > 0 else 0.0 for ink, area in zip(inks, areas)
    ]
    independent_marks = [
        index
        for index, fill_ratio in enumerate(fill_ratios)
        if fill_ratio >= _RUNNER_UP_MIN_FILL_RATIO
    ]
    if len(independent_marks) != 2:
        return None

    top_idx, runner_up_idx = sorted(
        independent_marks,
        key=lambda index: inks[index],
        reverse=True,
    )
    runner_up_ink = inks[runner_up_idx]
    if (
        fill_ratios[top_idx] < _CANCEL_MARK_MIN_FILL_RATIO
        or runner_up_ink <= 0
        or inks[top_idx] < runner_up_ink * _CANCEL_MARK_MIN_INK_RATIO
    ):
        return None

    top_mask = pure_ink_masks.get(boxes[top_idx].page_idx)
    runner_up_mask = pure_ink_masks.get(boxes[runner_up_idx].page_idx)
    if top_mask is None or runner_up_mask is None:
        return None
    top_density = _mark_bbox_density(top_mask, boxes[top_idx])
    runner_up_density = _mark_bbox_density(
        runner_up_mask, boxes[runner_up_idx]
    )
    if (
        top_density < _CANCEL_MARK_MIN_BBOX_DENSITY
        or runner_up_density > _RUNNER_UP_MAX_BBOX_DENSITY
    ):
        return None
    return runner_up_idx


def _local_mark_bbox_density(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    ink_pixels = cv2.countNonZero(mask)
    if ink_pixels <= 0:
        return 0.0
    _x, _y, width, height = cv2.boundingRect(mask)
    return ink_pixels / max(1, width * height)


def _checkbox_cancellation_runner_up_index(
    checkbox_infos: list[_CheckboxInkInfo],
    halo_infos: list[_CheckboxHaloInfo],
    check_results: list[bool],
) -> int | None:
    """Return the intended mark beside a strong two-dimensional cancellation."""
    if not (
        len(checkbox_infos) == len(halo_infos) == len(check_results)
    ):
        return None
    candidates = [
        index for index, is_checked in enumerate(check_results) if is_checked
    ]
    if len(candidates) != 2:
        return None

    combined_inks = [
        checkbox_infos[index].ink_pixels + halo_infos[index].ink_pixels
        for index in candidates
    ]
    ordered = sorted(
        zip(candidates, combined_inks), key=lambda item: item[1], reverse=True
    )
    (top_idx, top_ink), (runner_idx, runner_ink) = ordered
    if (
        runner_ink < _CHECKBOX_CANCEL_MIN_RUNNER_INK
        or top_ink < runner_ink * _CHECKBOX_CANCEL_MIN_COMBINED_RATIO
    ):
        return None

    top_info = checkbox_infos[top_idx]
    runner_info = checkbox_infos[runner_idx]
    runner_has_stroke = (
        runner_info.stroke_span_ratio >= 0.35
        or halo_infos[runner_idx].ink_pixels >= 12
    )
    shape_cancel = (
        _checkbox_mark_shape_spread(top_info, halo_infos[top_idx])
        >= _CHECKBOX_CANCEL_MIN_SHAPE_SPREAD
    )
    dense_cancel = (
        top_ink >= runner_ink * _CANCEL_MARK_MIN_INK_RATIO
        and top_info.ink_pixels / max(1, top_info.area) >= 0.18
        and _local_mark_bbox_density(top_info.ink_mask) >= 0.28
        and runner_info.ink_pixels / max(1, runner_info.area) <= 0.22
    )
    if (
        top_info.mark_strength < _CHECKBOX_RELIABLE_MARK_STRENGTH
        or runner_info.mark_strength < _CHECKBOX_RELIABLE_MARK_STRENGTH
        or not runner_has_stroke
        or not (shape_cancel or dense_cancel)
    ):
        return None
    return runner_idx


def _checkbox_ambiguous_indices(
    checkbox_infos: list[_CheckboxInkInfo],
    halo_infos: list[_CheckboxHaloInfo],
    check_results: list[bool],
) -> list[int]:
    """Return two independently credible marks that should not be guessed."""
    if not (
        len(checkbox_infos) == len(halo_infos) == len(check_results)
    ):
        return []
    candidates = [
        index for index, is_checked in enumerate(check_results) if is_checked
    ]
    if len(candidates) != 2:
        return []
    for index in candidates:
        direct = checkbox_infos[index]
        halo = halo_infos[index]
        if (
            direct.ink_pixels + halo.ink_pixels
            < _CHECKBOX_CANCEL_MIN_RUNNER_INK
            or direct.mark_strength < _CHECKBOX_RELIABLE_MARK_STRENGTH
            or direct.stroke_span_ratio < 0.35
            or halo.ink_pixels < 12
        ):
            return []
    return candidates


def _suppress_isolated_weak_checkbox_marks(
    checkbox_infos: list[_CheckboxInkInfo],
    halo_infos: list[_CheckboxHaloInfo],
    check_results: list[bool],
) -> list[bool]:
    """Drop a tiny isolated speck only when a strong sibling mark exists."""
    if not (
        len(checkbox_infos) == len(halo_infos) == len(check_results)
    ):
        return list(check_results)
    selected = [index for index, value in enumerate(check_results) if value]
    if len(selected) < 2:
        return list(check_results)
    strong = {
        index
        for index in selected
        if (
            checkbox_infos[index].ink_pixels + halo_infos[index].ink_pixels >= 50
            and checkbox_infos[index].mark_strength >= 0.25
        )
    }
    if not strong:
        return list(check_results)

    filtered = list(check_results)
    for index in selected:
        if index in strong:
            continue
        direct = checkbox_infos[index]
        halo = halo_infos[index]
        if (
            direct.ink_pixels + halo.ink_pixels <= 6
            and halo.ink_pixels == 0
            and direct.stroke_span_ratio < 0.30
        ):
            filtered[index] = False
    return filtered
