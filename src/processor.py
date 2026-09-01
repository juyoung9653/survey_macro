import copy
import gc
import hashlib
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import fitz
import numpy as np

from .export import export_to_excel
from .mark_analysis import (
    _CANCEL_MARK_MIN_BBOX_DENSITY,
    _CANCEL_MARK_MIN_FILL_RATIO,
    _CANCEL_MARK_MIN_INK_RATIO,
    _CHECKBOX_MIN_BORDER_CONFIDENCE,
    _CHECKBOX_MIN_DIRECT_EVIDENCE_PIXELS,
    _RUNNER_UP_MAX_BBOX_DENSITY,
    _RUNNER_UP_MIN_FILL_RATIO,
    _SHARED_STROKE_MIN_LOCAL_SUPPORT_RATIO,
    _SHARED_STROKE_MIN_SHAPE_SPREAD,
    _CheckboxFieldAnalysis,
    _CheckboxHaloInfo,
    _CheckboxInkInfo,
    _TemplateAlignmentCache,
    _align_template_mask_by_coverage,
    _best_shift_by_correlation,
    _build_stable_region_mask,
    _cancellation_runner_up_index,
    _checkbox_box_key,
    _checkbox_ambiguous_indices,
    _checkbox_cancellation_runner_up_index,
    _checkbox_difference_features,
    _checkbox_mark_shape_spread,
    _comment_has_overprint_evidence,
    _comment_region_has_meaningful_ink,
    _correlation_choice_is_ambiguous,
    _expand_comment_box,
    _extract_checkbox_halo_info,
    _filter_checkbox_mark_components,
    _is_checkbox_like,
    _local_mark_bbox_density,
    _mark_bbox_density,
    _prepare_checkbox_template_interiors,
    _prepare_template_alignment_cache,
    _refine_checkbox_box,
    _resolve_checkbox_halo_ownership,
    _scaled_shift_is_consistent,
    _suppress_isolated_weak_checkbox_marks,
    enforce_single_choice,
    evaluate_checkbox_halo_marks,
    evaluate_checkbox_marks,
    evaluate_marks,
    extract_checkbox_halo_ink_info,
    extract_checkbox_ink_info,
    extract_ink_info_from_mask,
    extract_pure_ink_mask,
)
from .models import Box, Field, TemplatePreset, validate_field_names
from .resources import AdaptiveResourceController, ResourceUnavailableError
from .vision import (
    ImageAligner,
    apply_rotation,
    auto_detect_checkboxes,
    load_pdf_pages,
)


_UI_TEMPLATE_SAMPLE_LIMIT = 31
_UI_DETECTION_SAMPLE_LIMIT = 7
_UI_TEMPLATE_CACHE_VERSION = 8
_MIB = 1024 * 1024
_ANALYSIS_SAMPLE_WORK = 2.0
_ANALYSIS_TEMPLATE_WORK = 1.0
_ANALYSIS_REUSED_SURVEY_WORK = 1.0
_ANALYSIS_RENDERED_SURVEY_WORK = 3.0
_ANALYSIS_PROGRESS_START = 2.0
_ANALYSIS_PROGRESS_SPAN = 95.0
_ANALYSIS_PLAN_EPOCH_WINDOWS = 4
_TEMPLATE_ALIGNMENT_STATE_LANES = 2
_RESULT_RUN_RETENTION_COUNT = 30
_RESULT_RUN_NAME_PATTERN = re.compile(
    r"^설문결과_(\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2})(?:_(\d+))?$"
)

_SamplePage = bytes | np.ndarray


@dataclass(frozen=True)
class _AnalysisOutputPaths:
    result_folder: Path
    run_folder: Path
    review_folder: Path
    comment_path: Path
    excel_path: Path


def _runtime_directory() -> Path:
    """Return the source launcher or packaged executable directory."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _prepare_analysis_output_paths(
    base_directory: str | Path | None = None,
    now: datetime | None = None,
) -> _AnalysisOutputPaths:
    base_path = (
        Path(base_directory).resolve()
        if base_directory is not None
        else _runtime_directory()
    )
    result_folder = base_path / "결과"
    result_folder.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now()).strftime("%Y.%m.%d.%H.%M.%S")
    base_stem = f"설문결과_{timestamp}"
    suffix = 1
    while True:
        run_stem = base_stem if suffix == 1 else f"{base_stem}_{suffix}"
        run_folder = result_folder / run_stem
        try:
            # The run folder acts as an atomic claim if two analyses start together.
            run_folder.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            suffix += 1
            continue
        review_folder = run_folder / "검토용"
        try:
            review_folder.mkdir()
        except Exception:
            run_folder.rmdir()
            raise
        return _AnalysisOutputPaths(
            result_folder=result_folder,
            run_folder=run_folder,
            review_folder=review_folder,
            comment_path=run_folder / "자유기입.pdf",
            excel_path=run_folder / "설문결과.xlsx",
        )


def _result_run_sort_key(run_folder: Path) -> tuple[datetime, int]:
    match = _RESULT_RUN_NAME_PATTERN.fullmatch(run_folder.name)
    if match is None:
        raise ValueError(f"결과 폴더 이름 형식이 올바르지 않습니다: {run_folder.name}")
    timestamp = datetime.strptime(match.group(1), "%Y.%m.%d.%H.%M.%S")
    return timestamp, int(match.group(2) or 1)


def _is_link_or_junction(path: Path) -> bool:
    try:
        is_junction = getattr(path, "is_junction", None)
        return path.is_symlink() or (is_junction is not None and is_junction())
    except OSError:
        return True


def _is_complete_result_run(run_folder: Path) -> bool:
    return (
        run_folder.is_dir()
        and not _is_link_or_junction(run_folder)
        and _RESULT_RUN_NAME_PATTERN.fullmatch(run_folder.name) is not None
        and (run_folder / "검토용").is_dir()
        and (run_folder / "설문결과.xlsx").is_file()
    )


def _delete_result_run(run_folder: Path, result_folder: Path) -> Path:
    """Delete one verified application result without following links."""
    run_folder = Path(run_folder)
    result_folder = Path(result_folder).resolve()
    if not _is_complete_result_run(run_folder):
        raise ValueError(f"완료된 분석 결과 폴더가 아닙니다: {run_folder.name}")
    resolved_run_folder = run_folder.resolve()
    if resolved_run_folder.parent != result_folder:
        raise ValueError(f"결과 폴더 밖의 경로는 삭제하지 않습니다: {run_folder}")
    if any(_is_link_or_junction(entry) for entry in run_folder.rglob("*")):
        raise ValueError(f"링크가 포함된 결과 폴더는 삭제하지 않습니다: {run_folder.name}")
    shutil.rmtree(resolved_run_folder)
    return run_folder


def _prune_old_result_runs(
    result_folder: Path,
    keep_count: int = _RESULT_RUN_RETENTION_COUNT,
) -> tuple[list[Path], list[str]]:
    """Keep only the newest completed application result folders."""
    result_folder = Path(result_folder).resolve()
    if not result_folder.is_dir():
        return [], []
    keep_count = max(0, int(keep_count))
    run_folders = sorted(
        (
            path
            for path in result_folder.iterdir()
            if _is_complete_result_run(path)
        ),
        key=_result_run_sort_key,
        reverse=True,
    )

    deleted: list[Path] = []
    errors: list[str] = []
    for run_folder in run_folders[keep_count:]:
        try:
            deleted.append(_delete_result_run(run_folder, result_folder))
        except Exception as exc:
            errors.append(f"{run_folder.name}: {exc}")
    return deleted, errors


@dataclass
class PresetLayoutRemapResult:
    config: TemplatePreset
    auxiliary_boxes: list[Box]
    page_transforms: dict[int, np.ndarray]
    matched_boxes: int
    expected_boxes: int
    accepted: bool
    compatible: bool = True
    matched_box_keys: frozenset[tuple[int, int, int, int, int]] = frozenset()
    supplied_box_keys: frozenset[tuple[int, int, int, int, int]] = frozenset()


def _fine_angle_for_page(
    fine_angle: float,
    page_fine_angles: list[float] | None,
    page_idx: int,
) -> float:
    page_angle = 0.0
    if page_fine_angles and 0 <= page_idx < len(page_fine_angles):
        page_angle = float(page_fine_angles[page_idx])
    return float(fine_angle) + page_angle


def _ui_template_cache_path(
    pdf_paths: list[str],
    page_count: int,
    rot_code: int,
    fine_angle: float,
    mode: str,
    page_fine_angles: list[float] | None = None,
) -> Path | None:
    parts = [
        f"v{_UI_TEMPLATE_CACHE_VERSION}",
        mode,
        str(page_count),
        str(rot_code),
        str(fine_angle),
        str(page_fine_angles or []),
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
    totals = Counter(normalized)
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


@dataclass(frozen=True)
class _EncodedPageImage:
    width: int
    height: int
    image_bytes: bytes | None


def _encode_page_image(
    img: np.ndarray, quality: int = 85
) -> _EncodedPageImage:
    height, width = img.shape[:2]
    return _EncodedPageImage(
        width=width,
        height=height,
        image_bytes=_encode_jpeg(img, quality),
    )


def _build_encoded_vector_page(
    target_doc, encoded: _EncodedPageImage, annotations: list
) -> None:
    page = target_doc.new_page(width=encoded.width, height=encoded.height)
    if encoded.image_bytes:
        page.insert_image(page.rect, stream=encoded.image_bytes)
    if not annotations:
        return

    shape = page.new_shape()
    for bx, by, bw, bh, label, is_ticked in annotations:
        color = (0, 1, 0) if is_ticked else (1, 0, 0)
        shape.draw_rect(fitz.Rect(bx, by, bx + bw, by + bh))
        shape.finish(color=color, width=2)
        shape.insert_text(
            fitz.Point(bx, max(0, by - 5)),
            label,
            fontsize=8,
            color=color,
        )
    shape.commit()


def _build_vector_page(
    target_doc, base_img: np.ndarray, annotations: list, img_quality: int = 85
) -> None:
    """벡터 PDF 페이지 생성: 배경 이미지(JPEG) + 벡터 사각형/텍스트 오버레이."""
    _build_encoded_vector_page(
        target_doc,
        _encode_page_image(base_img, img_quality),
        annotations,
    )


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


# Template generation


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


def _clean_response_regions(
    stack: np.ndarray,
    template: np.ndarray,
    config: TemplatePreset,
    page_idx: int,
) -> np.ndarray:
    """Use the brightest aligned samples inside response ROIs.

    Printed form strokes occur in every sample and remain dark. Handwritten ink
    occurs only in answered samples, so a high percentile keeps a clean local
    reference even when the same option is selected by more than half of the
    respondents. Checkbox frame pixels are restored from the median template to
    avoid thinning their borders through tiny residual registration errors.
    """
    if stack.ndim != 3 or stack.shape[0] < 2:
        return template

    height, width = template.shape[:2]
    percentile_index = int(np.ceil((stack.shape[0] - 1) * 0.90))
    response_boxes = [
        box
        for field in config.fields
        if not field.is_comment
        for box in field.boxes
        if box.page_idx == page_idx and box.w > 0 and box.h > 0
    ]
    for box in response_boxes:
        checkbox_like = _is_checkbox_like(box, template.shape)
        padding = max(4, round(min(box.w, box.h) * 0.7)) if checkbox_like else 0
        x1 = max(0, box.x - padding)
        y1 = max(0, box.y - padding)
        x2 = min(width, box.x + box.w + padding)
        y2 = min(height, box.y + box.h + padding)
        if x2 <= x1 or y2 <= y1:
            continue

        samples = stack[:, y1:y2, x1:x2]
        bright_patch = np.partition(
            samples, percentile_index, axis=0
        )[percentile_index]
        median_patch = template[y1:y2, x1:x2].copy()
        output_patch = template[y1:y2, x1:x2]
        if checkbox_like:
            output_patch[:] = bright_patch

        border_mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
        local_x1 = max(0, box.x - x1)
        local_y1 = max(0, box.y - y1)
        local_x2 = min(x2 - x1 - 1, box.x + box.w - x1 - 1)
        local_y2 = min(y2 - y1 - 1, box.y + box.h - y1 - 1)
        if local_x2 > local_x1 and local_y2 > local_y1:
            short_side = min(box.w, box.h)
            border_width = (
                max(2, round(short_side * 0.18))
                if checkbox_like
                else max(8, round(short_side * 0.10))
            )
            cv2.rectangle(
                border_mask,
                (local_x1, local_y1),
                (local_x2, local_y2),
                255,
                thickness=border_width,
            )
            if not checkbox_like:
                # Some presets cover several similar grid layouts whose cell
                # borders differ by more than a few pixels. Preserve long form
                # rules wherever they actually occur instead of treating their
                # registration residue as a handwritten response.
                printed = (median_patch < 210).astype(np.uint8) * 255
                vertical_length = max(9, round(printed.shape[0] * 0.30))
                horizontal_length = max(9, round(printed.shape[1] * 0.30))
                vertical_rules = cv2.morphologyEx(
                    printed,
                    cv2.MORPH_OPEN,
                    np.ones((vertical_length, 1), np.uint8),
                )
                horizontal_rules = cv2.morphologyEx(
                    printed,
                    cv2.MORPH_OPEN,
                    np.ones((1, horizontal_length), np.uint8),
                )
                printed_rules = cv2.bitwise_or(vertical_rules, horizontal_rules)
                printed_rules = cv2.dilate(
                    printed_rules, np.ones((5, 5), np.uint8), iterations=1
                )
                border_mask = cv2.bitwise_or(border_mask, printed_rules)
            if checkbox_like:
                output_patch[border_mask > 0] = median_patch[border_mask > 0]
            else:
                # Change only pen-like consensus strokes that are dark in the
                # median but absent from the brightest samples. Replacing the
                # whole cell makes small layout differences look like answers.
                lift = bright_patch.astype(np.int16) - median_patch.astype(np.int16)
                candidates = (lift >= 20).astype(np.uint8) * 255
                candidates[border_mask > 0] = 0
                candidates = cv2.morphologyEx(
                    candidates,
                    cv2.MORPH_CLOSE,
                    np.ones((3, 3), np.uint8),
                )
                component_count, labels, stats, _ = (
                    cv2.connectedComponentsWithStats(
                        (candidates > 0).astype(np.uint8), connectivity=8
                    )
                )
                pen_mask = np.zeros_like(candidates)
                min_area = max(5, round(candidates.size * 0.0003))
                min_span = max(6, round(short_side * 0.05))
                for component_idx in range(1, component_count):
                    component_area = int(stats[component_idx, cv2.CC_STAT_AREA])
                    component_span = max(
                        int(stats[component_idx, cv2.CC_STAT_WIDTH]),
                        int(stats[component_idx, cv2.CC_STAT_HEIGHT]),
                    )
                    if component_area >= min_area and component_span >= min_span:
                        pen_mask[labels == component_idx] = 255
                pen_mask = cv2.dilate(
                    pen_mask, np.ones((3, 3), np.uint8), iterations=1
                )
                pen_mask[border_mask > 0] = 0
                output_patch[pen_mask > 0] = bright_patch[pen_mask > 0]

    return template


def generate_dynamic_templates(
    pages_by_local_idx: dict[int, list],
    config: TemplatePreset | None = None,
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
        template = _median_uint8_inplace(stack)
        if config is not None:
            template = _clean_response_regions(stack, template, config, local_p)
        templates[local_p] = template
        del stack

    return templates


def _select_ui_detection_templates(
    pages_by_local_idx: dict[int, list],
    median_templates: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """Prefer a cleaner real sample only when it exposes more valid boxes."""
    selected = dict(median_templates)
    for local_page, template in median_templates.items():
        best_template = template
        best_count = len(
            auto_detect_checkboxes(cv2.cvtColor(template, cv2.COLOR_GRAY2BGR))
        )

        samples = pages_by_local_idx.get(local_page, [])
        if len(samples) <= _UI_DETECTION_SAMPLE_LIMIT:
            sample_indices = list(range(len(samples)))
        else:
            head_count = min(3, _UI_DETECTION_SAMPLE_LIMIT)
            sample_indices = list(range(head_count))
            remaining = _UI_DETECTION_SAMPLE_LIMIT - head_count
            spread = np.linspace(
                head_count,
                len(samples) - 1,
                num=remaining,
                dtype=int,
            )
            sample_indices.extend(int(index) for index in spread)

        for sample_index in sample_indices:
            sample = samples[sample_index]
            if isinstance(sample, bytes):
                candidate = cv2.imdecode(
                    np.frombuffer(sample, np.uint8), cv2.IMREAD_GRAYSCALE
                )
            else:
                candidate = sample
            if candidate is None:
                continue
            if candidate.ndim == 3:
                candidate = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)

            count = len(
                auto_detect_checkboxes(
                    cv2.cvtColor(candidate, cv2.COLOR_GRAY2BGR)
                )
            )
            if count > best_count:
                best_template = candidate
                best_count = count

        selected[local_page] = best_template
    return selected


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
    page_fine_angles: list[float] | None = None,
) -> dict[int, np.ndarray]:
    """UI에서 자동 탐지를 수행하기 전, PDF 전체를 읽어 깔끔한 빈 템플릿을 생성해 반환합니다."""
    if page_count <= 0:
        return {}

    cache_path = _ui_template_cache_path(
        [pdf_path],
        page_count,
        rot_code,
        fine_angle,
        "single",
        page_fine_angles,
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
            apply_rotation(
                p,
                rot_code,
                _fine_angle_for_page(fine_angle, page_fine_angles, local_p),
            ),
            refine_ecc=False,
        )
        for local_p, p in enumerate(pages[:page_count])
    ]

    survey_count = _survey_count(len(pages), page_count)

    pages_by_local_idx = {i: [] for i in range(page_count)}

    for survey_idx in range(survey_count):
        for local_p in range(page_count):
            global_p = survey_idx * page_count + local_p
            if global_p >= len(pages):
                break

            orig = apply_rotation(
                pages[global_p],
                rot_code,
                _fine_angle_for_page(fine_angle, page_fine_angles, local_p),
            )
            aligner = aligners[local_p] if local_p < len(aligners) else aligners[-1]

            aligned = aligner.align(orig)
            if len(pages_by_local_idx[local_p]) < _UI_TEMPLATE_SAMPLE_LIMIT:
                success, encoded = cv2.imencode(".png", aligned)
                if success:
                    pages_by_local_idx[local_p].append(encoded.tobytes())

    dynamic_templates = generate_dynamic_templates(pages_by_local_idx)
    dynamic_templates = _select_ui_detection_templates(
        pages_by_local_idx, dynamic_templates
    )
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
    page_fine_angles: list[float] | None = None,
) -> dict[int, np.ndarray]:
    """여러 PDF에서 템플릿을 생성하고 병합하여 더 정확한 템플릿을 만듭니다."""
    if not pdf_paths or page_count <= 0:
        return {}

    cache_path = _ui_template_cache_path(
        pdf_paths,
        page_count,
        rot_code,
        fine_angle,
        "multi",
        page_fine_angles,
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

                orig = apply_rotation(
                    pages[global_p],
                    rot_code,
                    _fine_angle_for_page(
                        fine_angle, page_fine_angles, local_p
                    ),
                )
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
    dynamic_templates = _select_ui_detection_templates(
        all_by_local_idx, dynamic_templates
    )
    _save_ui_template_cache(cache_path, dynamic_templates)

    bgr_templates = {}
    for k, v in dynamic_templates.items():
        bgr_templates[k] = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

    return bgr_templates


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
    if field.is_comment:
        return [_expand_comment_box(box) for box in z_sorted_boxes]

    if field.allow_duplicates:
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


def _config_checkbox_refs(
    config: TemplatePreset,
    page_idx: int,
    image_shape: tuple[int, ...],
) -> list[tuple[int, int, Box]]:
    return [
        (field_idx, box_idx, box)
        for field_idx, field in enumerate(config.fields)
        if not field.is_comment
        for box_idx, box in enumerate(field.boxes)
        if box.page_idx == page_idx and _is_checkbox_like(box, image_shape)
    ]


def _config_framed_refs(
    config: TemplatePreset,
    page_idx: int,
    source_template: np.ndarray,
) -> list[tuple[int, int, Box]]:
    """Return configured answer cells backed by visible frame edges."""
    source_mask = _layout_line_mask(source_template)
    refs = []
    for field_idx, field in enumerate(config.fields):
        if field.is_comment:
            continue
        for box_idx, box in enumerate(field.boxes):
            if box.page_idx != page_idx or _is_checkbox_like(
                box, source_template.shape
            ):
                continue
            edge_scores = _box_frame_edge_scores(source_mask, box)
            if sum(score >= 0.55 for score in edge_scores) >= 3:
                refs.append((field_idx, box_idx, box))
    return refs


def _layout_line_mask(image: np.ndarray) -> np.ndarray:
    gray = image
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    _threshold, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    return mask


def _box_frame_edge_scores(mask: np.ndarray, box: Box) -> tuple[float, ...]:
    """Measure continuous dark-line support along the four box edges."""
    image_h, image_w = mask.shape[:2]
    x1 = max(0, min(image_w - 1, int(box.x)))
    y1 = max(0, min(image_h - 1, int(box.y)))
    x2 = max(0, min(image_w - 1, int(box.x + box.w - 1)))
    y2 = max(0, min(image_h - 1, int(box.y + box.h - 1)))
    if x2 <= x1 or y2 <= y1:
        return (0.0, 0.0, 0.0, 0.0)

    trim_x = max(2, round((x2 - x1 + 1) * 0.08))
    trim_y = max(2, round((y2 - y1 + 1) * 0.08))
    short_side = min(x2 - x1 + 1, y2 - y1 + 1)
    band = max(2, min(12, round(short_side * 0.06)))

    def horizontal_score(center_y: int) -> float:
        start_x = x1 + trim_x
        end_x = x2 - trim_x + 1
        start_y = max(0, center_y - band)
        end_y = min(image_h, center_y + band + 1)
        if end_x <= start_x or end_y <= start_y:
            return 0.0
        rows = mask[start_y:end_y, start_x:end_x] > 0
        return float(np.max(np.mean(rows, axis=1)))

    def vertical_score(center_x: int) -> float:
        start_y = y1 + trim_y
        end_y = y2 - trim_y + 1
        start_x = max(0, center_x - band)
        end_x = min(image_w, center_x + band + 1)
        if end_y <= start_y or end_x <= start_x:
            return 0.0
        columns = mask[start_y:end_y, start_x:end_x] > 0
        return float(np.max(np.mean(columns, axis=0)))

    return (
        horizontal_score(y1),
        horizontal_score(y2),
        vertical_score(x1),
        vertical_score(x2),
    )


def _transform_box_in_place(
    box: Box,
    matrix: np.ndarray,
    target_shape: tuple[int, ...],
) -> None:
    height, width = target_shape[:2]
    corners = np.float32(
        [
            [box.x, box.y],
            [box.x + box.w, box.y],
            [box.x, box.y + box.h],
            [box.x + box.w, box.y + box.h],
        ]
    ).reshape(-1, 1, 2)
    transformed = cv2.transform(corners, matrix).reshape(-1, 2)
    x1 = max(0, min(width - 1, int(np.floor(transformed[:, 0].min()))))
    y1 = max(0, min(height - 1, int(np.floor(transformed[:, 1].min()))))
    x2 = max(x1 + 1, min(width, int(np.ceil(transformed[:, 0].max()))))
    y2 = max(y1 + 1, min(height, int(np.ceil(transformed[:, 1].max()))))
    box.x = x1
    box.y = y1
    box.w = x2 - x1
    box.h = y2 - y1


def _fit_checkbox_anchor_transform(
    source_boxes: list[Box],
    target_boxes: list[Box],
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> np.ndarray | None:
    if not source_boxes or len(source_boxes) != len(target_boxes):
        return None

    source_points = np.float32(
        [[box.x + box.w / 2, box.y + box.h / 2] for box in source_boxes]
    )
    target_points = np.float32(
        [[box.x + box.w / 2, box.y + box.h / 2] for box in target_boxes]
    )
    source_h, source_w = source_shape[:2]
    target_h, target_w = target_shape[:2]
    expected_scale = float(
        np.mean([target_w / max(1, source_w), target_h / max(1, source_h)])
    )
    median_target_side = float(
        np.median([min(box.w, box.h) for box in target_boxes])
    )

    matrix = None
    inliers = None
    if len(source_boxes) >= 3:
        matrix, inliers = cv2.estimateAffinePartial2D(
            source_points,
            target_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=max(8.0, median_target_side * 1.5),
            maxIters=5000,
            confidence=0.999,
            refineIters=20,
        )

    if matrix is None:
        scale_x = target_w / max(1, source_w)
        scale_y = target_h / max(1, source_h)
        offsets = target_points - source_points * np.float32([scale_x, scale_y])
        offset_x, offset_y = np.median(offsets, axis=0)
        matrix = np.float64(
            [[scale_x, 0.0, offset_x], [0.0, scale_y, offset_y]]
        )

    if not np.isfinite(matrix).all():
        return None
    affine_scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    if expected_scale <= 0 or not 0.75 <= affine_scale / expected_scale <= 1.25:
        return None
    rotation = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    if abs(rotation) > 5.0:
        return None

    projected = cv2.transform(source_points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    residuals = np.linalg.norm(projected - target_points, axis=1)
    if float(np.median(residuals)) > max(8.0, median_target_side):
        return None
    if float(np.percentile(residuals, 90)) > max(15.0, median_target_side * 2.0):
        return None
    if inliers is not None and float(np.mean(inliers)) < 0.7:
        return None
    return np.asarray(matrix, dtype=np.float64)


def _group_layout_ref_rows(
    refs: list[tuple[int, int, Box]], tolerance: float
) -> list[list[tuple[int, int, Box]]]:
    rows: list[list[tuple[int, int, Box]]] = []
    row_centers: list[float] = []
    for ref in sorted(refs, key=lambda item: (item[2].y + item[2].h / 2, item[2].x)):
        box = ref[2]
        center_y = box.y + box.h / 2
        if not rows or abs(center_y - row_centers[-1]) > tolerance:
            rows.append([ref])
            row_centers.append(center_y)
            continue
        rows[-1].append(ref)
        row_centers[-1] = float(
            np.mean([item[2].y + item[2].h / 2 for item in rows[-1]])
        )

    for row in rows:
        row.sort(key=lambda item: item[2].x + item[2].w / 2)
    return rows


def _match_layout_row_boxes(
    expected_row: list[tuple[int, int, Box]],
    detected_row: list[Box],
    max_x_distance: float,
) -> tuple[list[tuple[tuple[int, int, Box], Box]], float]:
    """Match an ordered row while allowing obscured or spurious frames."""
    expected_count = len(expected_row)
    detected_count = len(detected_row)
    states: list[
        list[
            tuple[
                int,
                float,
                list[tuple[tuple[int, int, Box], Box]],
            ]
            | None
        ]
    ] = [[None] * (detected_count + 1) for _ in range(expected_count + 1)]
    states[0][0] = (0, 0.0, [])

    def update(
        expected_pos: int,
        detected_pos: int,
        candidate: tuple[
            int,
            float,
            list[tuple[tuple[int, int, Box], Box]],
        ],
    ) -> None:
        current = states[expected_pos][detected_pos]
        if current is None or candidate[0] > current[0] or (
            candidate[0] == current[0] and candidate[1] < current[1]
        ):
            states[expected_pos][detected_pos] = candidate

    for expected_pos in range(expected_count + 1):
        for detected_pos in range(detected_count + 1):
            state = states[expected_pos][detected_pos]
            if state is None:
                continue
            if expected_pos < expected_count:
                update(expected_pos + 1, detected_pos, state)
            if detected_pos < detected_count:
                update(expected_pos, detected_pos + 1, state)
            if expected_pos >= expected_count or detected_pos >= detected_count:
                continue

            expected_ref = expected_row[expected_pos]
            expected_box = expected_ref[2]
            detected_box = detected_row[detected_pos]
            x_distance = abs(
                expected_box.x
                + expected_box.w / 2
                - detected_box.x
                - detected_box.w / 2
            )
            if x_distance > max_x_distance:
                continue
            size_cost = (
                abs(expected_box.w - detected_box.w)
                + abs(expected_box.h - detected_box.h)
            ) / max(1.0, expected_box.w + expected_box.h)
            update(
                expected_pos + 1,
                detected_pos + 1,
                (
                    state[0] + 1,
                    state[1] + x_distance / max_x_distance + size_cost,
                    state[2] + [(expected_ref, detected_box)],
                ),
            )

    best = states[expected_count][detected_count]
    if best is None:
        return [], 0.0
    return best[2], best[1]


def _match_layout_anchor_rows(
    expected_refs: list[tuple[int, int, Box]],
    detected: list[Box],
    target_shape: tuple[int, ...],
    *,
    tight: bool = False,
) -> tuple[list[tuple[tuple[int, int, Box], Box]], int, int]:
    """Match layout rows by order, then boxes within each matched row."""
    if not expected_refs or not detected:
        return [], 0, 0

    target_h, target_w = target_shape[:2]
    median_w = float(np.median([ref[2].w for ref in expected_refs]))
    median_h = float(np.median([ref[2].h for ref in expected_refs]))
    row_tolerance = max(4.0, median_h * 0.6)
    expected_rows = _group_layout_ref_rows(expected_refs, row_tolerance)
    detected_rows = _group_checkbox_rows(detected, row_tolerance)
    if tight:
        max_y_distance = max(median_h * 2.0, target_h * 0.012)
        max_x_distance = max(median_w * 2.5, target_w * 0.02)
    else:
        max_y_distance = max(median_h * 6.0, target_h * 0.06)
        max_x_distance = max(median_w * 6.0, target_w * 0.08)

    pair_options: dict[
        tuple[int, int],
        tuple[list[tuple[tuple[int, int, Box], Box]], float],
    ] = {}
    for expected_idx, expected_row in enumerate(expected_rows):
        expected_y = float(
            np.mean([ref[2].y + ref[2].h / 2 for ref in expected_row])
        )
        for detected_idx, detected_row in enumerate(detected_rows):
            detected_y = float(
                np.mean([box.y + box.h / 2 for box in detected_row])
            )
            y_distance = abs(detected_y - expected_y)
            if y_distance > max_y_distance:
                continue
            matches, box_cost = _match_layout_row_boxes(
                expected_row, detected_row, max_x_distance
            )
            minimum_matches = max(1, int(np.ceil(len(expected_row) * 0.5)))
            if len(matches) < minimum_matches:
                continue
            pair_options[(expected_idx, detected_idx)] = (
                matches,
                box_cost
                + y_distance / max_y_distance
                + abs(len(expected_row) - len(detected_row)) * 0.25,
            )

    row_count = len(expected_rows)
    detected_row_count = len(detected_rows)
    states: list[
        list[
            tuple[
                int,
                int,
                float,
                list[tuple[tuple[int, int, Box], Box]],
            ]
            | None
        ]
    ] = [[None] * (detected_row_count + 1) for _ in range(row_count + 1)]
    states[0][0] = (0, 0, 0.0, [])

    def update(
        expected_pos: int,
        detected_pos: int,
        candidate: tuple[
            int,
            int,
            float,
            list[tuple[tuple[int, int, Box], Box]],
        ],
    ) -> None:
        current = states[expected_pos][detected_pos]
        if current is None:
            states[expected_pos][detected_pos] = candidate
            return
        candidate_rank = (candidate[0], candidate[1], -candidate[2])
        current_rank = (current[0], current[1], -current[2])
        if candidate_rank > current_rank:
            states[expected_pos][detected_pos] = candidate

    for expected_pos in range(row_count + 1):
        for detected_pos in range(detected_row_count + 1):
            state = states[expected_pos][detected_pos]
            if state is None:
                continue
            if expected_pos < row_count:
                update(expected_pos + 1, detected_pos, state)
            if detected_pos < detected_row_count:
                update(expected_pos, detected_pos + 1, state)
            option = pair_options.get((expected_pos, detected_pos))
            if option is None:
                continue
            matches, pair_cost = option
            update(
                expected_pos + 1,
                detected_pos + 1,
                (
                    state[0] + len(matches),
                    state[1] + 1,
                    state[2] + pair_cost,
                    state[3] + matches,
                ),
            )

    best = states[row_count][detected_row_count]
    if best is None:
        return [], 0, row_count
    return best[3], best[1], row_count


def _detect_preset_checkbox_anchors(
    template: np.ndarray,
    expected_refs: list[tuple[int, int, Box]],
    page_idx: int,
) -> list[Box]:
    if not expected_refs:
        return []
    median_w = float(np.median([ref[2].w for ref in expected_refs]))
    median_h = float(np.median([ref[2].h for ref in expected_refs]))
    detection_image = (
        cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
        if template.ndim == 2
        else template
    )
    return [
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


def _layout_boxes_share_region(first: Box, second: Box) -> bool:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.w, second.x + second.w)
    bottom = min(first.y + first.h, second.y + second.h)
    overlap = max(0, right - left) * max(0, bottom - top)
    smaller_area = min(first.w * first.h, second.w * second.h)
    return smaller_area > 0 and overlap / smaller_area >= 0.8


def _layout_box_size_is_plausible(expected: Box, detected: Box) -> bool:
    width_ratio = max(expected.w, detected.w) / max(
        1, min(expected.w, detected.w)
    )
    height_ratio = max(expected.h, detected.h) / max(
        1, min(expected.h, detected.h)
    )
    return width_ratio <= 1.6 and height_ratio <= 1.6


def _detect_preset_framed_anchors(
    template: np.ndarray,
    expected_refs: list[tuple[int, int, Box]],
    page_idx: int,
) -> list[Box]:
    """Detect repeated framed cells using sizes supplied by the preset itself."""
    if not expected_refs:
        return []

    expected_boxes = [ref[2] for ref in expected_refs]
    widths = [box.w for box in expected_boxes]
    heights = [box.h for box in expected_boxes]
    detection_image = (
        cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
        if template.ndim == 2
        else template
    )
    detected = [
        Box(page_idx=page_idx, x=x, y=y, w=w, h=h)
        for x, y, w, h in auto_detect_checkboxes(
            detection_image,
            min_w=max(8, round(min(widths) * 0.55)),
            max_w=max(9, round(max(widths) * 1.8)),
            min_h=max(8, round(min(heights) * 0.55)),
            max_h=max(9, round(max(heights) * 1.8)),
        )
    ]
    return [
        box
        for box in detected
        if any(
            _layout_box_size_is_plausible(expected, box)
            for expected in expected_boxes
        )
    ]


def _merge_layout_candidates(*groups: list[Box]) -> list[Box]:
    merged: list[Box] = []
    for group in groups:
        for candidate in group:
            if any(
                _layout_boxes_share_region(candidate, existing)
                for existing in merged
            ):
                continue
            merged.append(candidate)
    return merged


def _layout_matches_have_plausible_geometry(
    matches: list[tuple[tuple[int, int, Box], Box]],
) -> bool:
    if not matches:
        return False
    plausible = sum(
        _layout_box_size_is_plausible(expected_ref[2], detected)
        for expected_ref, detected in matches
    )
    return plausible / len(matches) >= 0.8


def remap_preset_to_detected_layout(
    config: TemplatePreset,
    templates: dict[int, np.ndarray],
    source_templates: dict[int, np.ndarray] | None = None,
    auxiliary_boxes: list[Box] | None = None,
    detected_boxes_by_page: dict[int, list[Box]] | None = None,
) -> PresetLayoutRemapResult:
    """Move a saved preset into the current document's detected layout.

    Small checkbox rows establish a conservative page transform. Repeated framed
    answer cells are then matched independently by reading order, so a local
    table-width change does not invalidate an otherwise identical form. Row and
    column counts come from the preset at runtime; inserted target rows and a few
    missing frames are handled by the ordered matcher. Matched cells reuse exact
    detector geometry while comments and auxiliary boxes follow the validated
    page transform.
    """
    original_config = copy.deepcopy(config)
    original_auxiliary = copy.deepcopy(auxiliary_boxes or [])
    if not templates:
        return PresetLayoutRemapResult(
            original_config, original_auxiliary, {}, 0, 0, False
        )

    source_templates = source_templates or {}
    layout_pages = set(range(max(0, int(config.page_count))))
    if not layout_pages or any(
        page_idx not in source_templates for page_idx in layout_pages
    ):
        return PresetLayoutRemapResult(
            original_config, original_auxiliary, {}, 0, 0, False
        )

    source_checkbox_refs_by_page = {
        page_idx: _config_checkbox_refs(
            config, page_idx, source_templates[page_idx].shape
        )
        for page_idx in layout_pages
    }
    source_framed_refs_by_page = {
        page_idx: _config_framed_refs(
            config, page_idx, source_templates[page_idx]
        )
        for page_idx in layout_pages
    }
    required_pages = {
        page_idx
        for page_idx in layout_pages
        if source_checkbox_refs_by_page[page_idx]
        or source_framed_refs_by_page[page_idx]
    }
    expected_total = sum(
        len(source_checkbox_refs_by_page[page_idx])
        + len(source_framed_refs_by_page[page_idx])
        for page_idx in required_pages
    )
    if not required_pages:
        return PresetLayoutRemapResult(
            original_config, original_auxiliary, {}, 0, expected_total, False
        )
    if any(page_idx not in templates for page_idx in layout_pages):
        return PresetLayoutRemapResult(
            original_config, original_auxiliary, {}, 0, expected_total, False
        )

    scaled_config = copy.deepcopy(config)
    source_shapes: dict[int, tuple[int, ...]] = {}
    scale_transforms: dict[int, np.ndarray] = {}

    for page_idx, target in templates.items():
        source = source_templates.get(page_idx)
        source_shape = source.shape if source is not None else target.shape
        source_shapes[page_idx] = source_shape
        source_h, source_w = source_shape[:2]
        target_h, target_w = target.shape[:2]
        scale_matrix = np.float64(
            [
                [target_w / max(1, source_w), 0.0, 0.0],
                [0.0, target_h / max(1, source_h), 0.0],
            ]
        )
        scale_transforms[page_idx] = scale_matrix
        for field in scaled_config.fields:
            for box in field.boxes:
                if box.page_idx == page_idx:
                    _transform_box_in_place(box, scale_matrix, target.shape)

    page_transforms: dict[int, np.ndarray] = {}
    snapped_refs_by_page: dict[int, list[tuple[int, int, Box]]] = {}
    supplied_box_keys: set[tuple[int, int, int, int, int]] = set()
    matched_total = 0
    compatible = True
    for page_idx in required_pages:
        source_shape = source_shapes[page_idx]
        source_checkbox_refs = source_checkbox_refs_by_page[page_idx]
        source_framed_refs = source_framed_refs_by_page[page_idx]
        scaled_checkbox_refs = [
            (
                field_idx,
                box_idx,
                scaled_config.fields[field_idx].boxes[box_idx],
            )
            for field_idx, box_idx, _box in source_checkbox_refs
        ]
        scaled_framed_refs = [
            (
                field_idx,
                box_idx,
                scaled_config.fields[field_idx].boxes[box_idx],
            )
            for field_idx, box_idx, _box in source_framed_refs
        ]
        supplied_detected = (
            detected_boxes_by_page.get(page_idx)
            if detected_boxes_by_page is not None
            else None
        )
        valid_supplied = [
            copy.copy(box)
            for box in supplied_detected or []
            if box.page_idx == page_idx and box.w > 0 and box.h > 0
        ]
        if valid_supplied:
            supplied_box_keys.update(
                _checkbox_box_key(box) for box in valid_supplied
            )

        detected_checkboxes = _merge_layout_candidates(
            [
                box
                for box in valid_supplied
                if _is_checkbox_like(box, templates[page_idx].shape)
            ],
            _detect_preset_checkbox_anchors(
                templates[page_idx], scaled_checkbox_refs, page_idx
            ),
        )
        supplied_framed = [
            box
            for box in valid_supplied
            if not _is_checkbox_like(box, templates[page_idx].shape)
            and any(
                _layout_box_size_is_plausible(ref[2], box)
                for ref in scaled_framed_refs
            )
        ]
        detected_framed = _merge_layout_candidates(
            supplied_framed,
            _detect_preset_framed_anchors(
                templates[page_idx], scaled_framed_refs, page_idx
            ),
        )

        initial_matches, initial_row_matches, expected_row_count = (
            _match_layout_anchor_rows(
                scaled_checkbox_refs,
                detected_checkboxes,
                templates[page_idx].shape,
            )
        )
        if source_checkbox_refs:
            checkbox_initial_coverage = len(initial_matches) / len(
                source_checkbox_refs
            )
            checkbox_initial_row_coverage = initial_row_matches / max(
                1, expected_row_count
            )
            page_compatible = (
                checkbox_initial_coverage >= 0.45
                and checkbox_initial_row_coverage >= 0.5
            )
        else:
            page_compatible = True

        if len(initial_matches) < 3 and scaled_framed_refs:
            initial_matches, initial_row_matches, expected_row_count = (
                _match_layout_anchor_rows(
                    scaled_framed_refs,
                    detected_framed,
                    templates[page_idx].shape,
                )
            )
            framed_initial_coverage = len(initial_matches) / max(
                1, len(source_framed_refs)
            )
            framed_initial_row_coverage = initial_row_matches / max(
                1, expected_row_count
            )
            page_compatible = page_compatible and (
                framed_initial_coverage >= 0.45
                and framed_initial_row_coverage >= 0.5
            )
        compatible = compatible and page_compatible
        if len(initial_matches) < 3:
            continue

        initial_source = [
            config.fields[field_idx].boxes[box_idx]
            for (field_idx, box_idx, _scaled_box), _target_box in initial_matches
        ]
        initial_target = [target_box for _source_ref, target_box in initial_matches]
        initial_matrix = _fit_checkbox_anchor_transform(
            initial_source,
            initial_target,
            source_shape,
            templates[page_idx].shape,
        )
        if initial_matrix is None:
            continue

        projected_checkbox_refs = []
        for field_idx, box_idx, source_box in source_checkbox_refs:
            projected = copy.copy(source_box)
            _transform_box_in_place(
                projected, initial_matrix, templates[page_idx].shape
            )
            projected_checkbox_refs.append((field_idx, box_idx, projected))
        checkbox_matches, matched_checkbox_rows, checkbox_row_count = (
            _match_layout_anchor_rows(
                projected_checkbox_refs,
                detected_checkboxes,
                templates[page_idx].shape,
                tight=True,
            )
        )

        projected_framed_refs = []
        for field_idx, box_idx, source_box in source_framed_refs:
            projected = copy.copy(source_box)
            _transform_box_in_place(
                projected, initial_matrix, templates[page_idx].shape
            )
            projected_framed_refs.append((field_idx, box_idx, projected))
        framed_matches, matched_framed_rows, framed_row_count = (
            _match_layout_anchor_rows(
                projected_framed_refs,
                detected_framed,
                templates[page_idx].shape,
                tight=True,
            )
        )

        page_matched = len(checkbox_matches) + len(framed_matches)
        matched_total += page_matched
        checkbox_coverage = len(checkbox_matches) / max(
            1, len(source_checkbox_refs)
        )
        checkbox_row_coverage = matched_checkbox_rows / max(
            1, checkbox_row_count
        )
        framed_coverage = len(framed_matches) / max(1, len(source_framed_refs))
        framed_row_coverage = matched_framed_rows / max(1, framed_row_count)

        transform_matches = checkbox_matches or framed_matches
        matched_source = [
            config.fields[field_idx].boxes[box_idx]
            for (field_idx, box_idx, _projected), _target in transform_matches
        ]
        matched_target = [target for _source, target in transform_matches]
        matrix = _fit_checkbox_anchor_transform(
            matched_source,
            matched_target,
            source_shape,
            templates[page_idx].shape,
        )
        if matrix is None:
            continue

        if source_checkbox_refs:
            all_y = np.asarray(
                [
                    box.y + box.h / 2
                    for _field_idx, _box_idx, box in source_checkbox_refs
                ],
                dtype=np.float64,
            )
            matched_checkbox_source = [
                config.fields[field_idx].boxes[box_idx]
                for (field_idx, box_idx, _projected), _target in checkbox_matches
            ]
            matched_y = np.asarray(
                [box.y + box.h / 2 for box in matched_checkbox_source],
                dtype=np.float64,
            )
        else:
            all_y = np.empty(0, dtype=np.float64)
            matched_y = np.empty(0, dtype=np.float64)
        if len(all_y) <= 1 or float(np.ptp(all_y)) <= 0:
            vertical_span = 1.0
        else:
            vertical_span = float(np.ptp(matched_y) / np.ptp(all_y))
        checkbox_layout_accepted = not source_checkbox_refs or (
            checkbox_coverage >= 0.85
            and checkbox_row_coverage >= 0.8
            and vertical_span >= 0.7
        )
        framed_layout_accepted = not source_framed_refs or (
            framed_coverage >= 0.8
            and framed_row_coverage >= 0.8
            and _layout_matches_have_plausible_geometry(framed_matches)
        )
        page_accepted = checkbox_layout_accepted and framed_layout_accepted
        if not page_accepted:
            if source_framed_refs and not framed_layout_accepted:
                compatible = False
            continue

        page_transforms[page_idx] = matrix
        snapped_refs_by_page[page_idx] = [
            (field_idx, box_idx, target)
            for (field_idx, box_idx, _projected), target in [
                *checkbox_matches,
                *framed_matches,
            ]
        ]

    if set(page_transforms) != required_pages:
        return PresetLayoutRemapResult(
            original_config,
            original_auxiliary,
            {},
            matched_total,
            expected_total,
            False,
            compatible,
        )

    for page_idx, matrix in scale_transforms.items():
        page_transforms.setdefault(page_idx, matrix)

    adjusted_config = copy.deepcopy(config)
    adjusted_auxiliary = copy.deepcopy(auxiliary_boxes or [])
    matched_box_keys: set[tuple[int, int, int, int, int]] = set()
    for field in adjusted_config.fields:
        for box in field.boxes:
            matrix = page_transforms.get(box.page_idx)
            target = templates.get(box.page_idx)
            if matrix is not None and target is not None:
                _transform_box_in_place(box, matrix, target.shape)
    for box in adjusted_auxiliary:
        matrix = page_transforms.get(box.page_idx)
        target = templates.get(box.page_idx)
        if matrix is not None and target is not None:
            _transform_box_in_place(box, matrix, target.shape)

    # Global anchors move comments and manually drawn regions. Checkbox frames
    # themselves use the exact detector result so their borders remain precise.
    for page_idx, snapped_refs in snapped_refs_by_page.items():
        for field_idx, box_idx, target_box in snapped_refs:
            matched_box_keys.add(_checkbox_box_key(target_box))
            output_box = adjusted_config.fields[field_idx].boxes[box_idx]
            output_box.x = target_box.x
            output_box.y = target_box.y
            output_box.w = target_box.w
            output_box.h = target_box.h

    return PresetLayoutRemapResult(
        adjusted_config,
        adjusted_auxiliary,
        page_transforms,
        matched_total,
        expected_total,
        True,
        True,
        frozenset(matched_box_keys),
        frozenset(supplied_box_keys),
    )


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


def _encode_review_crop(
    gray_page: np.ndarray,
    boxes: list[Box],
) -> bytes | None:
    """Encode a compact original-image crop around one ambiguous field."""
    if gray_page.size == 0 or not boxes:
        return None
    image_h, image_w = gray_page.shape[:2]
    x1 = max(0, min(box.x for box in boxes) - 70)
    y1 = max(0, min(box.y for box in boxes) - 55)
    x2 = min(image_w, max(box.x + box.w for box in boxes) + 70)
    y2 = min(image_h, max(box.y + box.h for box in boxes) + 55)
    if x2 <= x1 or y2 <= y1:
        return None
    success, encoded = cv2.imencode(".png", gray_page[y1:y2, x1:x2])
    return encoded.tobytes() if success else None


# Survey processing and pipeline orchestration


def process_survey_data(
    survey_data: dict,
    config: TemplatePreset,
    dynamic_templates: dict[int, np.ndarray],
    template_masks: dict[int, np.ndarray] | None = None,
    field_plans: list[tuple[Field, list[Box], list[Box], bool]] | None = None,
    trust_checkbox_layout: bool = False,
    checkbox_template_interiors: dict[
        tuple[int, int, int, int, int], np.ndarray
    ]
    | None = None,
    prepared_stable_region_masks: dict[int, np.ndarray | None] | None = None,
    template_alignment_caches: dict[int, _TemplateAlignmentCache] | None = None,
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
    stable_region_masks = prepared_stable_region_masks
    if stable_region_masks is None:
        stable_region_masks = {
            local_p: _build_stable_region_mask(gray_img.shape, config, local_p)
            for local_p, gray_img in survey_gray_pages.items()
        }
    for local_p, gray_img in survey_gray_pages.items():
        if local_p in dynamic_templates:
            pure_ink_masks[local_p] = extract_pure_ink_mask(
                gray_img,
                dynamic_templates[local_p],
                config.template_dilate_pct,
                template_masks.get(local_p) if template_masks else None,
                stable_region_masks.get(local_p),
                template_alignment_caches.get(local_p)
                if template_alignment_caches
                else None,
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

        checkbox_infos = []
        for box in scoring_boxes:
            box_key = _checkbox_box_key(box)
            has_prepared_interior = (
                checkbox_template_interiors is not None
                and box_key in checkbox_template_interiors
            )
            checkbox_infos.append(
                extract_checkbox_ink_info(
                    survey_gray_pages[box.page_idx],
                    box,
                    None
                    if has_prepared_interior
                    else dynamic_templates.get(box.page_idx),
                    checkbox_template_interiors[box_key]
                    if has_prepared_interior
                    else None,
                )
            )
        direct_inks = [info.ink_pixels for info in checkbox_infos]
        direct_areas = [info.area for info in checkbox_infos]
        direct_strengths = [info.mark_strength for info in checkbox_infos]
        reliable_direct = [
            trust_checkbox_layout
            or info.border_confidence >= _CHECKBOX_MIN_BORDER_CONFIDENCE
            for info in checkbox_infos
        ]
        direct_results = evaluate_checkbox_marks(
            direct_inks,
            direct_areas,
            strict=field.allow_duplicates,
            strengths=direct_strengths,
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
            direct_strengths,
            direct_results,
            halo_infos,
        )
        ownership_candidates.extend(zip(checkbox_infos, direct_results, halo_infos))

    _resolve_checkbox_halo_ownership(
        pure_ink_masks, survey_gray_pages, ownership_candidates
    )
    for plan_idx, checkbox_analysis in checkbox_analyses.items():
        field = plans[plan_idx][0]
        checkbox_analysis.direct_inks = [
            info.ink_pixels for info in checkbox_analysis.checkbox_infos
        ]
        checkbox_analysis.direct_strengths = [
            info.mark_strength for info in checkbox_analysis.checkbox_infos
        ]
        refreshed_results = evaluate_checkbox_marks(
            checkbox_analysis.direct_inks,
            checkbox_analysis.direct_areas,
            strict=field.allow_duplicates,
            strengths=checkbox_analysis.direct_strengths,
        )
        checkbox_analysis.direct_results = [
            is_checked and is_reliable
            for is_checked, is_reliable in zip(
                refreshed_results, checkbox_analysis.reliable_direct
            )
        ]

    for plan_idx, (field, scoring_boxes, working_boxes, is_contiguous) in enumerate(
        plans
    ):
        checkbox_analysis = checkbox_analyses.get(plan_idx)
        checkbox_mode = checkbox_analysis is not None

        direct_inks: list[int] = []
        direct_areas: list[int] = []
        direct_results: list[bool] = []
        direct_strengths: list[float] = []
        halo_inks: list[int] = []
        halo_areas: list[int] = []
        reliable_direct: list[bool] = []

        if checkbox_analysis is not None:
            checkbox_infos = checkbox_analysis.checkbox_infos
            reliable_direct = checkbox_analysis.reliable_direct
            direct_inks = checkbox_analysis.direct_inks
            direct_areas = checkbox_analysis.direct_areas
            direct_strengths = checkbox_analysis.direct_strengths
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
            if field.allow_duplicates:
                check_results = _suppress_isolated_weak_checkbox_marks(
                    checkbox_infos,
                    halo_infos,
                    check_results,
                )
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
                page_idx = info.box.page_idx
                addition = checkbox_ink_additions.get(page_idx)
                if addition is None:
                    addition = np.zeros_like(
                        survey_gray_pages[page_idx], dtype=np.uint8
                    )
                    checkbox_ink_additions[page_idx] = addition
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
            current_inks, current_areas, current_boxes = _collect_ink_data(
                working_boxes, pure_ink_masks
            )
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
            check_results = []
            for source_box, working_box in zip(scoring_boxes, valid_boxes):
                page_idx = working_box.page_idx
                has_ink = page_idx in pure_ink_masks and (
                    _comment_region_has_meaningful_ink(
                        pure_ink_masks[page_idx], working_box
                    )
                )
                template = dynamic_templates.get(page_idx)
                has_overprint = (
                    not has_ink
                    and page_idx in survey_gray_pages
                    and template is not None
                    and _comment_has_overprint_evidence(
                        survey_gray_pages[page_idx], template, source_box
                    )
                )
                check_results.append(has_ink or has_overprint)

        if field.is_comment:
            has_comment = False
            total_boxes = len(scoring_boxes)

            for idx, (box, is_ticked) in enumerate(
                zip(scoring_boxes, check_results), start=1
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

        ambiguous_indices: list[int] = []
        if not field.allow_duplicates:
            if checkbox_mode and sum(check_results) > 1:
                correction_idx = _checkbox_cancellation_runner_up_index(
                    checkbox_infos,
                    halo_infos,
                    check_results,
                )
                if correction_idx is not None:
                    check_results = [
                        index == correction_idx
                        for index in range(len(check_results))
                    ]
                else:
                    ambiguous_indices = _checkbox_ambiguous_indices(
                        checkbox_infos,
                        halo_infos,
                        check_results,
                    )
                    if not ambiguous_indices:
                        checked_indices = [
                            index
                            for index, is_checked in enumerate(check_results)
                            if is_checked
                        ]
                        best_idx = max(
                            checked_indices,
                            key=lambda index: (
                                direct_strengths[index],
                                direct_inks[index] * 3 + halo_inks[index],
                                direct_inks[index] / max(1, direct_areas[index]),
                                halo_inks[index] / max(1, halo_areas[index]),
                            ),
                        )
                        check_results = [
                            index == best_idx
                            for index in range(len(check_results))
                        ]
            else:
                correction_idx = None
                if not checkbox_mode:
                    correction_idx = _cancellation_runner_up_index(
                        inks,
                        areas,
                        valid_boxes,
                        pure_ink_masks,
                    )
                if correction_idx is None:
                    check_results = enforce_single_choice(
                        check_results, inks, areas
                    )
                else:
                    check_results = [
                        index == correction_idx
                        for index in range(len(check_results))
                    ]
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

        if ambiguous_indices:
            candidate_values = [
                mapped_value if mapped_value else label
                for label, mapped_value in checked_labels
            ]
            candidate_text = "/".join(candidate_values)
            row_data[field.name] = f"검수필요({candidate_text})"
            page_indices = {
                valid_boxes[index].page_idx for index in ambiguous_indices
            }
            review_image = None
            if len(page_indices) == 1:
                page_idx = next(iter(page_indices))
                if page_idx in survey_gray_pages:
                    review_image = _encode_review_crop(
                        survey_gray_pages[page_idx], valid_boxes
                    )
            row_data.setdefault("__review_items__", []).append(
                {
                    "파일명": fname,
                    "페이지": survey_label,
                    "문항": field.name,
                    "후보": candidate_text,
                    "사유": "두 선택지의 표식이 모두 뚜렷해 자동 확정하지 않음",
                    "이미지": review_image,
                }
            )
        elif checked_labels:
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
def _render_pdf_page(doc, global_p: int, dpi: int) -> np.ndarray:
    page = doc[global_p]
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)


def _render_aligned_page(
    doc,
    global_p: int,
    local_p: int,
    aligners: list,
    rot_code: int,
    fine_angle: float,
    dpi: int,
    page_fine_angles: list[float] | None = None,
) -> tuple[int, np.ndarray]:
    page_img = _render_pdf_page(doc, global_p, dpi)
    orig = apply_rotation(
        page_img,
        rot_code,
        _fine_angle_for_page(fine_angle, page_fine_angles, local_p),
    )
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
    page_fine_angles: list[float] | None = None,
) -> dict[int, np.ndarray]:
    """한 설문의 페이지를 순차 렌더링·정합합니다."""
    sequential_result: dict[int, np.ndarray] = {}
    for local_p in range(page_count):
        global_p = survey_idx * page_count + local_p
        if global_p >= len(doc):
            break
        _, aligned = _render_aligned_page(
            doc,
            global_p,
            local_p,
            aligners,
            rot_code,
            fine_angle,
            dpi,
            page_fine_angles,
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
    sample_pages: dict[int, list[_SamplePage]],
    alignment_references: list[np.ndarray],
) -> int:
    pixels = _reference_page_pixels(alignment_references)
    peak_bytes = 0
    for local_p, samples in sample_pages.items():
        page_pixels = (
            pixels[local_p] if local_p < len(pixels) else max(pixels, default=0)
        )
        sample_count = max(1, len(samples))
        samples_are_raw = bool(samples) and isinstance(samples[0], np.ndarray)
        # Raw samples only need the partition stack. PNG samples additionally
        # need decoded arrays while that stack is being created.
        multiplier = 1 if samples_are_raw else 2
        peak_bytes = max(peak_bytes, page_pixels * sample_count * multiplier)
    return peak_bytes + (max(pixels, default=0) * 4) + 64 * _MIB


def _estimate_raw_sample_pipeline_peak_bytes(
    alignment_references: list[np.ndarray],
    sample_count: int,
    worker_count: int,
) -> int:
    pixels = _reference_page_pixels(alignment_references)
    raw_cache_bytes = sum(pixels) * max(0, sample_count)
    largest_page_bytes = max(pixels, default=0)
    template_peak = (
        raw_cache_bytes
        + largest_page_bytes * max(1, sample_count)
        + largest_page_bytes * 4
        + 64 * _MIB
    )
    collection_peak = (
        raw_cache_bytes
        + _estimate_render_memory_bytes(alignment_references)
        * max(1, worker_count)
        + largest_page_bytes * 2
    )
    return max(template_peak, collection_peak)


def _build_page_aligners(
    alignment_references: list[np.ndarray],
    config: TemplatePreset,
) -> list[ImageAligner]:
    return [
        ImageAligner(
            reference,
            stable_mask=_build_stable_region_mask(reference.shape, config, page_idx),
            sparse_lk=True,
        )
        for page_idx, reference in enumerate(alignment_references)
    ]


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
) -> tuple[str, dict[int, list[_SamplePage]]]:
    fname = Path(fpath).stem
    try:
        doc = fitz.open(fpath)
    except Exception as e:
        print(f"파일 로드 실패 ({fname}): {e}")
        return _file_key(fpath), {}

    page_count = config.page_count
    if not alignment_references:
        doc.close()
        raise RuntimeError("페이지 정합 기준 이미지가 없습니다.")

    f_pages: dict[int, list[_SamplePage]] = {i: [] for i in range(page_count)}
    survey_count = _survey_count(len(doc), page_count)
    limit = min(survey_count, sample_limit)

    jobs = [
        (survey_idx, local_p, survey_idx * page_count + local_p)
        for survey_idx in range(limit)
        for local_p in range(page_count)
        if survey_idx * page_count + local_p < len(doc)
    ]
    supports_parallel_plans = (
        resource_controller is not None
        and callable(getattr(resource_controller, "parallel_checkpoint", None))
        and len(jobs) > 1
    )

    try:
        if not supports_parallel_plans:
            aligners = _build_page_aligners(alignment_references, config)
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
                    config.page_fine_angles,
                )
                for local_p, aligned in pages.items():
                    success, encoded = cv2.imencode(".png", aligned)
                    f_pages[local_p].append(
                        encoded.tobytes() if success else b""
                    )
                if progress_cb:
                    progress_cb(survey_idx + 1, limit)
        else:
            alignment_lanes: list[list[ImageAligner] | None] = []
            retain_raw_samples: bool | None = None

            def align_and_encode(
                lane_idx: int, local_p: int, page_img: np.ndarray
            ) -> tuple[int, _SamplePage]:
                lane_aligners = alignment_lanes[lane_idx]
                if lane_aligners is None:
                    lane_aligners = _build_page_aligners(
                        alignment_references, config
                    )
                    alignment_lanes[lane_idx] = lane_aligners
                rotated = apply_rotation(
                    page_img,
                    config.rot_code,
                    config.fine_angle_for_page(local_p),
                )
                aligner = (
                    lane_aligners[local_p]
                    if local_p < len(lane_aligners)
                    else lane_aligners[-1]
                )
                aligned = aligner.align(rotated)
                if retain_raw_samples:
                    return local_p, aligned
                success, encoded = cv2.imencode(".png", aligned)
                return local_p, encoded.tobytes() if success else b""

            max_workers = max(
                1,
                min(
                    len(jobs),
                    int(getattr(resource_controller, "cpu_count", 1)),
                ),
            )
            alignment_lanes.extend(
                [None] * min(max_workers, _TEMPLATE_ALIGNMENT_STATE_LANES)
            )
            render_memory = _estimate_render_memory_bytes(
                alignment_references
            )
            coordinator_memory = max(
                _reference_page_pixels(alignment_references), default=0
            ) * 2
            executor = ThreadPoolExecutor(max_workers=max_workers)
            job_idx = 0
            try:
                while job_idx < len(jobs):
                    plan = resource_controller.parallel_checkpoint(
                        render_memory,
                        pending_tasks=len(jobs) - job_idx,
                        stage=f"{fname} 템플릿 표본 처리",
                        status_cb=resource_status_cb,
                        coordinator_threads=1,
                        coordinator_memory_bytes=coordinator_memory,
                    )
                    if retain_raw_samples is None:
                        available_headroom = max(
                            0,
                            int(getattr(plan, "available_memory_bytes", 0))
                            - int(getattr(plan, "reserve_memory_bytes", 0))
                            - int(getattr(plan, "safety_memory_bytes", 0)),
                        )
                        raw_pipeline_peak = (
                            _estimate_raw_sample_pipeline_peak_bytes(
                                alignment_references,
                                limit,
                                plan.worker_count,
                            )
                        )
                        retain_raw_samples = (
                            raw_pipeline_peak <= available_headroom
                        )
                    window_size = min(
                        len(jobs) - job_idx,
                        int(plan.worker_count),
                        len(alignment_lanes),
                    )
                    epoch_end = min(
                        len(jobs),
                        job_idx
                        + window_size * _ANALYSIS_PLAN_EPOCH_WINDOWS,
                    )
                    next_submit = job_idx
                    pending = deque()

                    def submit_job(job_position: int):
                        job = jobs[job_position]
                        _survey_idx, local_p, global_p = job
                        page_img = _render_pdf_page(doc, global_p, dpi)
                        # Bind warm alignment history to a stable lane instead
                        # of whichever executor thread happens to run the job.
                        lane_idx = job_position % len(alignment_lanes)
                        return executor.submit(
                            align_and_encode, lane_idx, local_p, page_img
                        )

                    while (
                        next_submit < epoch_end
                        and len(pending) < window_size
                    ):
                        pending.append(
                            (jobs[next_submit], submit_job(next_submit))
                        )
                        next_submit += 1

                    while pending:
                        job, future = pending.popleft()
                        local_p, encoded = future.result()
                        if next_submit < epoch_end:
                            pending.append(
                                (
                                    jobs[next_submit],
                                    submit_job(next_submit),
                                )
                            )
                            next_submit += 1
                        f_pages[local_p].append(encoded)
                        job_idx += 1
                        survey_idx, job_local_p, _global_p = job
                        is_last_survey_page = (
                            job_local_p == page_count - 1
                            or job_idx == len(jobs)
                            or jobs[job_idx][0] != survey_idx
                        )
                        if is_last_survey_page and progress_cb:
                            progress_cb(survey_idx + 1, limit)

            finally:
                executor.shutdown(wait=True)
    finally:
        doc.close()

    return _file_key(fpath), f_pages


def _decode_sampled_survey(
    sample_pages: dict[int, list[_SamplePage]] | None,
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
        if isinstance(data, np.ndarray):
            image = data
        else:
            if not data:
                return None
            image = cv2.imdecode(
                np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE
            )
        if image is None:
            return None
        decoded[local_p] = image
    return decoded


def _sampled_survey_is_available(
    sample_pages: dict[int, list[_SamplePage]] | None,
    survey_idx: int,
    expected_pages: int,
) -> bool:
    if not sample_pages or expected_pages <= 0:
        return False
    for local_p in range(expected_pages):
        samples = sample_pages.get(local_p, [])
        if survey_idx >= len(samples):
            return False
        sample = samples[survey_idx]
        if isinstance(sample, np.ndarray):
            if sample.size == 0:
                return False
        elif not sample:
            return False
    return True


def _build_file_templates(
    file_paths: list[str],
    sample_results: dict[str, dict[int, list[_SamplePage]]],
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
    sample_pages: dict[int, list[_SamplePage]] | None = None,
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
        aligners = _build_page_aligners(alignment_references, config)
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
        page_shapes = {
            page_idx: reference.shape
            for page_idx, reference in enumerate(alignment_references)
        }
        checkbox_template_interiors = _prepare_checkbox_template_interiors(
            field_plans, f_template, page_shapes
        )
        stable_region_masks = {
            page_idx: _build_stable_region_mask(
                reference.shape, analysis_config, page_idx
            )
            for page_idx, reference in enumerate(alignment_references)
        }
        template_alignment_caches = {
            page_idx: _prepare_template_alignment_cache(
                template_mask,
                alignment_references[page_idx].shape,
                alignment_mask=stable_region_masks.get(page_idx),
            )
            for page_idx, template_mask in template_masks.items()
            if page_idx < len(alignment_references)
        }

        page_count = config.page_count
        rot_code = config.rot_code
        fine_angle = config.fine_angle
        file_results: list[dict] = []
        comment_pages: list[bytes] = []
        survey_memory_bytes = _estimate_survey_memory_bytes(alignment_references)

        def analyze_survey(
            survey_idx: int,
            survey_gray_pages: dict[int, np.ndarray],
        ):
            survey_data = {
                "fname": file_label,
                "row_title": f"{file_label}_{survey_idx + 1}p",
                "gray_pages": survey_gray_pages,
            }
            (
                row_data,
                debug_base,
                ink_base,
                debug_ann,
                ink_ann,
                comment_images,
            ) = process_survey_data(
                survey_data,
                analysis_config,
                f_template,
                template_masks,
                field_plans,
                trust_checkbox_layout=trust_checkbox_layout,
                checkbox_template_interiors=checkbox_template_interiors,
                prepared_stable_region_masks=stable_region_masks,
                template_alignment_caches=template_alignment_caches,
            )
            encoded_debug = {
                local_p: _encode_page_image(image)
                for local_p, image in debug_base.items()
            }
            encoded_ink = {
                local_p: _encode_page_image(image)
                for local_p, image in ink_base.items()
            }
            encoded_comments = {
                local_p: encoded_debug[local_p].image_bytes
                for local_p in comment_images
                if local_p in encoded_debug
                and encoded_debug[local_p].image_bytes is not None
            }
            return (
                row_data,
                encoded_debug,
                encoded_ink,
                debug_ann,
                ink_ann,
                encoded_comments,
            )

        def consume_survey_result(result, completed_surveys: int) -> None:
            row_data, debug_base, ink_base, debug_ann, ink_ann, cp = result
            field_values = [
                v
                for k, v in row_data.items()
                if k not in ("파일명", "페이지") and not k.startswith("__")
            ]
            if any(str(v).strip() for v in field_values):
                file_results.append(row_data)

            for local_p in sorted(debug_base):
                _build_encoded_vector_page(
                    out_orig, debug_base[local_p], debug_ann.get(local_p, [])
                )
            for local_p in sorted(ink_base):
                _build_encoded_vector_page(
                    out_ink, ink_base[local_p], ink_ann.get(local_p, [])
                )
            for local_p in sorted(cp):
                comment_pages.append(cp[local_p])

            if progress_cb:
                progress_cb(completed_surveys, survey_count)

        supports_parallel_plans = (
            resource_controller is not None
            and callable(
                getattr(resource_controller, "parallel_checkpoint", None)
            )
        )
        executor = None
        if supports_parallel_plans and survey_count > 1:
            executor = ThreadPoolExecutor(
                max_workers=max(
                    1,
                    min(
                        survey_count,
                        int(getattr(resource_controller, "cpu_count", 1)),
                    ),
                )
            )

        def submit_cached_survey(candidate_idx: int):
            if executor is None:
                return None
            candidate_pages = min(
                page_count,
                max(0, len(doc) - candidate_idx * page_count),
            )
            candidate_images = _decode_sampled_survey(
                sample_pages, candidate_idx, candidate_pages
            )
            if candidate_images is None:
                return None
            return executor.submit(
                analyze_survey, candidate_idx, candidate_images
            )

        survey_idx = 0
        try:
            while survey_idx < survey_count:
                expected_pages = min(
                    page_count, max(0, len(doc) - survey_idx * page_count)
                )
                cached_count = 0
                if executor is not None:
                    for candidate_idx in range(survey_idx, survey_count):
                        candidate_pages = min(
                            page_count,
                            max(0, len(doc) - candidate_idx * page_count),
                        )
                        if not _sampled_survey_is_available(
                            sample_pages, candidate_idx, candidate_pages
                        ):
                            break
                        cached_count += 1

                if cached_count > 0:
                    plan = resource_controller.parallel_checkpoint(
                        survey_memory_bytes,
                        pending_tasks=cached_count,
                        stage=f"{file_label} 설문 분석",
                        status_cb=resource_status_cb,
                        coordinator_threads=1,
                    )
                    window_size = min(cached_count, plan.worker_count)
                    epoch_size = min(
                        cached_count,
                        window_size * _ANALYSIS_PLAN_EPOCH_WINDOWS,
                    )
                else:
                    if resource_controller is not None:
                        resource_controller.checkpoint(
                            survey_memory_bytes,
                            stage=f"{file_label} 설문 분석",
                            status_cb=resource_status_cb,
                        )
                    window_size = 0
                    epoch_size = 0

                pending = deque()
                if executor is not None and window_size > 0:
                    epoch_end = survey_idx + epoch_size
                    next_submit = survey_idx
                    while (
                        next_submit < epoch_end
                        and len(pending) < window_size
                    ):
                        future = submit_cached_survey(next_submit)
                        if future is None:
                            epoch_end = next_submit
                            break
                        pending.append((next_submit, future))
                        next_submit += 1

                if pending:
                    overlap_output = (
                        int(getattr(plan, "total_cpu_threads", 2)) > 1
                    )
                    while pending:
                        completed_idx, future = pending.popleft()
                        result = future.result()
                        if overlap_output and next_submit < epoch_end:
                            next_future = submit_cached_survey(next_submit)
                            if next_future is None:
                                epoch_end = next_submit
                            else:
                                pending.append((next_submit, next_future))
                                next_submit += 1

                        consume_survey_result(result, completed_idx + 1)
                        survey_idx = completed_idx + 1

                        if not overlap_output and next_submit < epoch_end:
                            next_future = submit_cached_survey(next_submit)
                            if next_future is None:
                                epoch_end = next_submit
                            else:
                                pending.append((next_submit, next_future))
                                next_submit += 1
                    continue

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
                        config.page_fine_angles,
                    )
                consume_survey_result(
                    analyze_survey(survey_idx, survey_gray_pages),
                    survey_idx + 1,
                )
                survey_idx += 1
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

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


def _prepare_alignment_references(
    template_pages: list,
    config: TemplatePreset,
    template_pages_preprocessed: bool = False,
) -> list:
    if template_pages_preprocessed:
        return list(template_pages[: config.page_count])
    return [
        apply_rotation(
            page,
            config.rot_code,
            config.fine_angle_for_page(page_idx),
        )
        for page_idx, page in enumerate(template_pages[: config.page_count])
    ]


_PAGE_ASPECT_RATIO_TOLERANCE = 0.04


def _validate_analysis_page_geometry(
    file_paths: list[str],
    alignment_references: list[np.ndarray],
    config: TemplatePreset,
) -> None:
    """Reject incompatible page geometry while allowing ordinary DPI changes."""
    page_count = int(config.page_count)
    if page_count <= 0 or len(alignment_references) < page_count:
        raise ValueError(
            "분석에 필요한 페이지별 기준 템플릿이 모두 준비되지 않았습니다."
        )

    reference_ratios: list[float] = []
    for page_idx, reference in enumerate(alignment_references[:page_count]):
        if not isinstance(reference, np.ndarray) or reference.size == 0:
            raise ValueError(
                f"{page_idx + 1}쪽 기준 템플릿이 비어 있어 분석할 수 없습니다."
            )
        height, width = reference.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError(
                f"{page_idx + 1}쪽 기준 템플릿 크기를 확인할 수 없습니다."
            )
        reference_ratios.append(width / height)

    swaps_axes = config.rot_code in (
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
    )
    for file_path in file_paths:
        try:
            doc = fitz.open(file_path)
        except Exception as exc:
            raise ValueError(
                f"'{Path(file_path).name}' 파일을 확인할 수 없습니다: {exc}"
            ) from exc
        try:
            if len(doc) == 0:
                raise ValueError(
                    f"'{Path(file_path).name}'에 분석할 페이지가 없습니다."
                )
            for global_page_idx, page in enumerate(doc):
                local_page_idx = global_page_idx % page_count
                width = float(page.rect.width)
                height = float(page.rect.height)
                if swaps_axes:
                    width, height = height, width
                if width <= 0 or height <= 0:
                    raise ValueError(
                        f"'{Path(file_path).name}' {global_page_idx + 1}쪽의 "
                        "페이지 크기를 확인할 수 없습니다."
                    )
                actual_ratio = width / height
                expected_ratio = reference_ratios[local_page_idx]
                relative_error = abs(actual_ratio / expected_ratio - 1.0)
                if relative_error > _PAGE_ASPECT_RATIO_TOLERANCE:
                    raise ValueError(
                        "분석을 중단했습니다. "
                        f"'{Path(file_path).name}' {global_page_idx + 1}쪽의 "
                        f"페이지 비율({width:.0f}×{height:.0f})이 "
                        f"{local_page_idx + 1}쪽 기준 템플릿과 다릅니다. "
                        "같은 설문지와 방향인지 확인해주세요."
                    )
        finally:
            doc.close()


def _validate_analysis_template_layout(
    config: TemplatePreset,
    templates: dict[int, np.ndarray],
    alignment_references: list[np.ndarray],
    file_label: str,
) -> None:
    """Block answer extraction when a file template is missing or incompatible."""
    required_pages = set(range(max(0, int(config.page_count))))
    missing_pages = sorted(required_pages.difference(templates))
    if missing_pages:
        page_text = ", ".join(f"{page_idx + 1}쪽" for page_idx in missing_pages)
        raise ValueError(
            f"{file_label}: 자동 생성 템플릿에서 {page_text}을(를) 만들지 "
            "못해 분석을 중단했습니다."
        )

    source_templates = {
        page_idx: reference
        for page_idx, reference in enumerate(
            alignment_references[: config.page_count]
        )
    }
    layout_check = remap_preset_to_detected_layout(
        config,
        templates,
        source_templates=source_templates,
    )
    if layout_check.expected_boxes > 0 and not layout_check.compatible:
        raise ValueError(
            f"{file_label}: 체크박스 배치 정합 신뢰도가 부족해 분석을 "
            "중단했습니다. 다른 설문지가 섞였는지 확인해주세요."
        )


def run_analysis(
    file_paths: list[str],
    template_pages: list,
    config: TemplatePreset,
    progress_cb=None,
    resource_controller: AdaptiveResourceController | None = None,
    template_pages_preprocessed: bool = False,
    output_base_dir: str | Path | None = None,
) -> bool:
    def report_progress(value: float, message: str = ""):
        if progress_cb:
            progress_cb(max(0, min(100, value)), message)

    report_progress(0, "분석 준비 중...")

    num_files = len(file_paths)
    if num_files == 0:
        return False
    validate_field_names(field.name for field in config.fields)
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
    # Saved preset templates and pages aligned to them are already in the
    # configured rotation coordinate system. Applying the configured angle
    # again would rotate them twice while survey pages are rotated once.
    alignment_references = _prepare_alignment_references(
        template_pages,
        config,
        template_pages_preprocessed,
    )
    if not alignment_references:
        raise ValueError("페이지 정합 기준 이미지가 없어 분석할 수 없습니다.")
    _validate_analysis_page_geometry(file_paths, alignment_references, config)

    output_paths = _prepare_analysis_output_paths(output_base_dir)
    review_folder = output_paths.review_folder

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
    all_results: list[dict] = []
    analysis_failures: list[str] = []
    completed = 0
    comment_path = output_paths.comment_path
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
        sample_pages: dict[int, list[_SamplePage]] | None,
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
            sample_pages: dict[int, list[_SamplePage]] = {}
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
                    file_template = generate_dynamic_templates(
                        sample_pages, config=config
                    )
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

            _validate_analysis_template_layout(
                config,
                file_template,
                alignment_references,
                file_label,
            )

            if reference_templates is None:
                reference_templates = file_template
                save_reference_template()

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

        # 자유기입 이미지는 파일 분석 직후 PDF 문서로 옮겼으므로 JPEG 목록을 따로
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
    success = export_to_excel(all_results, config, str(output_paths.excel_path))
    if success and not analysis_failures:
        report_progress(99, "오래된 결과 정리 중...")
        _deleted, cleanup_errors = _prune_old_result_runs(
            output_paths.result_folder
        )
        for cleanup_error in cleanup_errors:
            print(f"결과 자동 정리 보류: {cleanup_error}")
    report_progress(100, "완료")
    return success and not analysis_failures
