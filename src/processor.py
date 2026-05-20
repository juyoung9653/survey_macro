import copy
import gc
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
                    max_dw = 0

            if abs(cx1 - cx2) < (box.w + other.w) / 2 + 15:
                if dist_y > 0:
                    max_dh = min(max_dh, dist_y / 2.1)
                else:
                    max_dh = 0

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
        sample_data = pages[:31]
        # PNG 압축된 bytes면 디코딩, raw array면 그대로 사용
        if isinstance(sample_data[0], bytes):
            decoded = [
                cv2.imdecode(np.frombuffer(d, np.uint8), cv2.IMREAD_GRAYSCALE)
                for d in sample_data
            ]
            stack = np.stack(decoded, axis=0)
        else:
            stack = np.stack(sample_data, axis=0)
        templates[local_p] = np.median(stack, axis=0).astype(np.uint8)
    return templates


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
            if len(pages_by_local_idx[local_p]) < 31:
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
                if len(all_by_local_idx[local_p]) < 31:
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


def extract_pure_ink_mask(
    target_gray: np.ndarray, template_gray: np.ndarray
) -> np.ndarray:

    blur_target = cv2.GaussianBlur(target_gray, (3, 3), 0)
    blur_template = cv2.GaussianBlur(template_gray, (3, 3), 0)

    diff = cv2.absdiff(blur_template, blur_target)

    _, pure_ink_mask = cv2.threshold(diff, 70, 255, cv2.THRESH_BINARY)

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
    field, z_sorted_boxes: list[Box], is_contiguous: bool, all_boxes: list[Box]
) -> list[Box]:
    if field.is_comment or is_contiguous or field.allow_duplicates:
        return [copy.copy(b) for b in z_sorted_boxes]

    return expand_isolated_boxes(z_sorted_boxes, all_boxes, scale_factor=3.0)


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
    row_data = {"페이지": survey_data["row_title"]}
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
            mask = extract_pure_ink_mask(gray_img, dynamic_templates[local_p])
            pure_ink_masks[local_p] = mask

            ink_display = cv2.bitwise_not(mask)
            survey_ink_only_images[local_p] = ink_display
            ink_annotations[local_p] = []

    all_boxes_in_config = [b for f in config.fields for b in f.boxes]

    for field in config.fields:
        z_sorted_boxes = sort_boxes_z_pattern(field.boxes)
        is_contiguous = is_contiguous_group(z_sorted_boxes)

        working_boxes = _select_working_boxes(
            field, z_sorted_boxes, is_contiguous, all_boxes_in_config
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
            row_data[field.name] = ", ".join(output_values)
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

    total_surveys = 0
    for fpath in file_paths:
        try:
            doc = fitz.open(fpath)
            total_pages = len(doc)
            doc.close()
        except Exception:
            continue

        total_surveys += _survey_count(total_pages, config.page_count)

    if total_surveys <= 0:
        total_surveys = 1

    aligners = [
        ImageAligner(apply_rotation(p, config.rot_code, config.fine_angle))
        for p in template_pages[: config.page_count]
    ]

    surveys_data = []
    pages_by_local_idx = {i: [] for i in range(config.page_count)}
    prepared_surveys = 0

    for fpath in file_paths:
        fname = Path(fpath).stem
        try:
            doc = fitz.open(fpath)
        except Exception as e:
            print(f"파일 로드 실패 ({fname}): {e}")
            continue

        survey_count = _survey_count(len(doc), config.page_count)

        for survey_idx in range(survey_count):
            survey_gray_pages = {}

            for local_p in range(config.page_count):
                global_p = survey_idx * config.page_count + local_p
                if global_p >= len(doc):
                    break

                # 단일 페이지 렌더링 (직접 gray로 → BGR 대비 1/3 메모리)
                page = doc[global_p]
                pix = page.get_pixmap(dpi=200, colorspace=fitz.csGRAY)
                page_img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.h, pix.w
                )

                orig = apply_rotation(page_img, config.rot_code, config.fine_angle)
                aligner = aligners[local_p] if local_p < len(aligners) else aligners[-1]
                aligned = aligner.align(orig)

                survey_gray_pages[local_p] = aligned

                if len(pages_by_local_idx[local_p]) < 31:
                    pages_by_local_idx[local_p].append(
                        cv2.imencode(".png", aligned)[1].tobytes()
                    )

            surveys_data.append(
                {
                    "fname": fname,
                    "row_title": f"Page{survey_idx + 1}",
                    "gray_pages": {
                        k: cv2.imencode(".png", v)[1].tobytes()
                        for k, v in survey_gray_pages.items()
                    },
                }
            )
            prepared_surveys += 1
            report_progress(
                int(prepared_surveys / total_surveys * 40),
                f"분석 준비 중... ({prepared_surveys}/{total_surveys})",
            )

        doc.close()

    # pages_by_local_idx는 generate_dynamic_templates에 넘긴 후 바로 해제
    dynamic_templates = generate_dynamic_templates(pages_by_local_idx)
    pages_by_local_idx.clear()
    report_progress(40, "분석 시작 중...")

    template_pdf = fitz.open()
    for local_p in sorted(dynamic_templates.keys()):
        _insert_img_into_pdf(template_pdf, dynamic_templates[local_p], quality=90)

    if len(template_pdf) > 0:
        template_pdf.save(review_folder / "00_추론된_템플릿.pdf")
    template_pdf.close()

    all_results = []
    out_pdfs_original = {}
    out_pdfs_ink_only = {}
    comment_doc = fitz.open()
    processed_surveys = 0

    for survey in surveys_data:
        fname = survey["fname"]
        if fname not in out_pdfs_original:
            out_pdfs_original[fname] = fitz.open()
            out_pdfs_ink_only[fname] = fitz.open()

        row_data, debug_base, ink_base, debug_ann, ink_ann, comment_pages = (
            process_survey_data(survey, config, dynamic_templates)
        )
        all_results.append(row_data)

        for local_p in sorted(debug_base.keys()):
            _build_vector_page(
                out_pdfs_original[fname],
                debug_base[local_p],
                debug_ann.get(local_p, []),
            )

        for local_p in sorted(ink_base.keys()):
            _build_vector_page(
                out_pdfs_ink_only[fname],
                ink_base[local_p],
                ink_ann.get(local_p, []),
            )

        for local_p in sorted(comment_pages.keys()):
            _insert_img_into_pdf(comment_doc, comment_pages[local_p])

        del row_data, debug_base, ink_base, debug_ann, ink_ann, comment_pages
        gc.collect()

        processed_surveys += 1
        report_progress(
            40 + int(processed_surveys / total_surveys * 50),
            f"분석 중... ({processed_surveys}/{total_surveys})",
        )

    report_progress(95, "결과 저장 중...")

    # 파일별로 바로바로 저장 (fitz.Document가 메모리에 페이지를 계속 쌓지 않도록)
    for fname in out_pdfs_original.keys():
        doc_orig = out_pdfs_original[fname]
        if len(doc_orig) > 0:
            doc_orig.save(review_folder / f"{fname}_원본포함.pdf")
        doc_orig.close()

        doc_ink = out_pdfs_ink_only[fname]
        if len(doc_ink) > 0:
            doc_ink.save(review_folder / f"{fname}_잉크추출.pdf")
        doc_ink.close()

    if len(comment_doc) > 0:
        comment_doc.save(review_folder / "의견.pdf")
    comment_doc.close()

    success = export_to_excel(all_results, config)
    report_progress(100, "완료")
    return success
