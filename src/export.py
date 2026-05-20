from typing import Optional

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from .models import TemplatePreset


def _try_number(s: str):
    """문자열이 순수 숫자면 int/float로 변환, 아니면 원본 반환"""
    s = s.strip()
    if not s:
        return s
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _compute_field_stats(results: list[dict], field, config: TemplatePreset):
    """field 하나에 대한 카운트·미응답 수를 반환"""
    n_opts = len(field.boxes)
    raw_labels = []
    for i in range(n_opts):
        if i < len(field.value_map) and field.value_map[i].strip():
            v = field.value_map[i].strip()
            raw_labels.append(_try_number(v))
        else:
            raw_labels.append(i + 1)
    option_labels = raw_labels

    if config.reverse_numbering:
        option_labels = list(reversed(option_labels))

    label_keys = [str(lbl) for lbl in option_labels]
    counts = {k: 0 for k in label_keys}
    non_resp = 0

    for item in results:
        raw = str(item.get(field.name, "")).strip()
        if not raw:
            non_resp += 1
        else:
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            for p in parts:
                if p in counts:
                    counts[p] += 1

    return option_labels, label_keys, counts, non_resp


# ── 스타일 상수 ──
_header_font = Font(bold=True, size=11, color="FFFFFF")
_header_fill = PatternFill("solid", fgColor="4472C4")
_group_font = Font(bold=True, size=11, color="1F4E79")
_group_fill = PatternFill("solid", fgColor="D6E4F0")
_count_fill = PatternFill("solid", fgColor="F2F2F2")
_pct_fill = PatternFill("solid", fgColor="E2EFDA")
_non_resp_fill = PatternFill("solid", fgColor="FCE4D6")
_red_font = Font(bold=True, size=11, color="FFFFFF")
_red_fill = PatternFill("solid", fgColor="C00000")
_thin_border = Border(
    left=Side(style="thin"),
    right=Side(style="thin"),
    top=Side(style="thin"),
    bottom=Side(style="thin"),
)
_center_align = Alignment(horizontal="center", vertical="center")


def _write_stats_rows(
    ws,
    config: TemplatePreset,
    results: list[dict],
    start_row: int = 1,
) -> int:
    """통계 행을 쓰고 마지막 사용 행 번호를 반환"""
    total_responses = len(results)
    row = start_row

    for field in config.fields:
        if field.is_comment:
            continue

        option_labels, label_keys, counts, non_resp = _compute_field_stats(
            results, field, config
        )
        last_col = len(option_labels) + 2

        # ── 첫 번째 행: 그룹명 + 옵션 레이블 + 미응답 ──
        cell = ws.cell(row=row, column=1, value=field.name)
        cell.font = _group_font
        cell.fill = _group_fill
        cell.border = _thin_border
        cell.alignment = _center_align

        for ci, label in enumerate(option_labels):
            c = ci + 2
            cell = ws.cell(row=row, column=c, value=label)
            cell.font = _header_font
            cell.fill = _header_fill
            cell.border = _thin_border
            cell.alignment = _center_align

        cell = ws.cell(row=row, column=last_col, value="미응답")
        cell.font = _red_font
        cell.fill = _red_fill
        cell.border = _thin_border
        cell.alignment = _center_align

        # ── 두 번째 행: 갯수 ──
        row += 1
        cell = ws.cell(row=row, column=1, value="갯수")
        cell.font = Font(bold=True, size=10)
        cell.fill = _count_fill
        cell.border = _thin_border
        cell.alignment = _center_align

        for ci, key in enumerate(label_keys):
            c = ci + 2
            cell = ws.cell(row=row, column=c, value=counts[key])
            cell.fill = _count_fill
            cell.border = _thin_border
            cell.alignment = _center_align

        cell = ws.cell(row=row, column=last_col, value=non_resp)
        cell.fill = _non_resp_fill
        cell.border = _thin_border
        cell.alignment = _center_align

        # ── 세 번째 행: 응답률 ──
        row += 1
        cell = ws.cell(row=row, column=1, value="응답률")
        cell.font = Font(bold=True, size=10)
        cell.fill = _pct_fill
        cell.border = _thin_border
        cell.alignment = _center_align

        for ci, key in enumerate(label_keys):
            c = ci + 2
            val = counts[key] / total_responses if total_responses else 0
            cell = ws.cell(row=row, column=c, value=val)
            cell.number_format = "0.0%"
            cell.fill = _pct_fill
            cell.border = _thin_border
            cell.alignment = _center_align

        val_non = non_resp / total_responses if total_responses else 0
        cell = ws.cell(row=row, column=last_col, value=val_non)
        cell.number_format = "0.0%"
        cell.fill = _non_resp_fill
        cell.border = _thin_border
        cell.alignment = _center_align

        # 빈 구분 행
        row += 2

    # 열 너비
    ws.column_dimensions["A"].width = 18
    for ci in range(2, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(ci)].width = 12

    return row


def export_to_excel(
    results: list[dict],
    config: Optional[TemplatePreset] = None,
    out_path: str = "설문결과.xlsx",
) -> bool:
    if not results:
        return False

    try:
        wb = openpyxl.Workbook()

        # ── 1. 결과 시트 (첫 번째) ──
        ws_result = wb.active
        ws_result.title = "결과"

        headers = list(results[0].keys())
        ws_result.append(headers)

        for item in results:
            row = [_try_number(str(item.get(header, ""))) for header in headers]
            ws_result.append(row)

        # ── 2. 전체 통계 시트 (두 번째) ──
        if config and config.fields:
            ws_overall = wb.create_sheet("전체 통계")
            _write_stats_rows(ws_overall, config, results)

            # ── 3. 파일별 통계 시트 ──
            # 파일명별로 결과 묶기
            fname_to_results: dict[str, list[dict]] = {}
            for item in results:
                fn = str(item.get("파일명", "")).strip()
                if not fn:
                    fn = "기타"
                fname_to_results.setdefault(fn, []).append(item)

            for fname, f_results in fname_to_results.items():
                sheet_name = fname[:31]  # 엑셀 시트명 31자 제한
                ws_f = wb.create_sheet(sheet_name)
                _write_stats_rows(ws_f, config, f_results)

        wb.save(out_path)
        return True

    except Exception as e:
        print(f"엑셀 저장 실패: {e}")
        return False
