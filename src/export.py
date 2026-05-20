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


def export_to_excel(
    results: list[dict],
    config: Optional[TemplatePreset] = None,
    out_path: str = "설문결과.xlsx",
) -> bool:
    if not results:
        return False

    try:
        wb = openpyxl.Workbook()

        # ── 결과 시트 ──
        ws_result = wb.active
        ws_result.title = "결과"

        headers = list(results[0].keys())
        ws_result.append(headers)

        for item in results:
            row = [_try_number(str(item.get(header, ""))) for header in headers]
            ws_result.append(row)

        # ── 통계 시트 ──
        if config and config.fields:
            _build_stats_sheet(wb, results, config)

        wb.save(out_path)
        return True

    except Exception as e:
        print(f"엑셀 저장 실패: {e}")
        return False


def _build_stats_sheet(
    wb: openpyxl.Workbook, results: list[dict], config: "TemplatePreset"
) -> None:
    ws = wb.create_sheet("통계")

    # 스타일 정의
    header_font = Font(bold=True, size=11, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="4472C4")
    group_font = Font(bold=True, size=11, color="1F4E79")
    group_fill = PatternFill("solid", fgColor="D6E4F0")
    count_fill = PatternFill("solid", fgColor="F2F2F2")
    pct_fill = PatternFill("solid", fgColor="E2EFDA")
    thin_border = Border(
        left=Side(style="thin"),
        right=Side(style="thin"),
        top=Side(style="thin"),
        bottom=Side(style="thin"),
    )
    center_align = Alignment(horizontal="center", vertical="center")

    total_responses = len(results)
    row = 1

    for field in config.fields:
        if field.is_comment:
            continue

        # 옵션 레이블: position별로 value_map에 실값이 있으면 쓰고, 없으면 숫자
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

        # key는 문자열 (데이터 매칭용)
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

        # ── 첫 번째 행: 그룹명 + 옵션 레이블 + 미응답 ──
        ws.cell(row=row, column=1, value=field.name).font = group_font
        ws.cell(row=row, column=1).fill = group_fill
        ws.cell(row=row, column=1).border = thin_border
        ws.cell(row=row, column=1).alignment = center_align

        for ci, label in enumerate(option_labels):
            c = ci + 2
            cell = ws.cell(row=row, column=c, value=label)
            cell.font = header_font
            cell.fill = header_fill
            cell.border = thin_border
            cell.alignment = center_align

        last_col = len(option_labels) + 2
        cell = ws.cell(row=row, column=last_col, value="미응답")
        cell.font = Font(bold=True, size=11, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="C00000")
        cell.border = thin_border
        cell.alignment = center_align

        # ── 두 번째 행: 갯수 ──
        row += 1
        ws.cell(row=row, column=1, value="갯수").font = Font(bold=True, size=10)
        ws.cell(row=row, column=1).fill = count_fill
        ws.cell(row=row, column=1).border = thin_border
        ws.cell(row=row, column=1).alignment = center_align

        for ci, key in enumerate(label_keys):
            c = ci + 2
            cell = ws.cell(row=row, column=c, value=counts[key])
            cell.fill = count_fill
            cell.border = thin_border
            cell.alignment = center_align

        cell = ws.cell(row=row, column=last_col, value=non_resp)
        cell.fill = PatternFill("solid", fgColor="FCE4D6")
        cell.border = thin_border
        cell.alignment = center_align

        # ── 세 번째 행: 응답률 ──
        row += 1
        ws.cell(row=row, column=1, value="응답률").font = Font(bold=True, size=10)
        ws.cell(row=row, column=1).fill = pct_fill
        ws.cell(row=row, column=1).border = thin_border
        ws.cell(row=row, column=1).alignment = center_align

        for ci, key in enumerate(label_keys):
            c = ci + 2
            val = counts[key] / total_responses if total_responses else 0
            cell = ws.cell(row=row, column=c, value=val)
            cell.number_format = "0.0%"
            cell.fill = pct_fill
            cell.border = thin_border
            cell.alignment = center_align

        val_non = non_resp / total_responses if total_responses else 0
        cell = ws.cell(row=row, column=last_col, value=val_non)
        cell.number_format = "0.0%"
        cell.fill = PatternFill("solid", fgColor="FCE4D6")
        cell.border = thin_border
        cell.alignment = center_align

        # 빈 구분 행
        row += 2

    # 열 너비
    ws.column_dimensions["A"].width = 18
    for ci in range(2, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(ci)].width = 12
