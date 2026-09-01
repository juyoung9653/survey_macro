import os
import unittest
from pathlib import Path

import numpy as np

from src.processor import generate_ui_templates, generate_ui_templates_multi
from src.vision import (
    auto_detect_checkboxes,
    estimate_deskew_angle,
    load_pdf_pages,
)


_RUN_CORPUS_TESTS = os.getenv("RUN_SCAN_CORPUS_TESTS") == "1"
_SCAN_DIR = Path(
    os.getenv("SURVEY_SCAN_CORPUS", r"C:\Users\Public\scan")
)

_ONE_PAGE_PDFS = (
    "보드게임.pdf",
    "색깔.pdf",
    "운영실무.pdf",
    "웰다잉.pdf",
    "자활창업.pdf",
)
_TWO_PAGE_PDFS = (
    "거점.pdf",
    "급식.pdf",
    "도서관.pdf",
    "라라.pdf",
    "매점.pdf",
    "물류.pdf",
    "세차분식.pdf",
    "청소.pdf",
    "청수.pdf",
    "카페주거.pdf",
    "편의점.pdf",
    "헤이클린.pdf",
)

_TWO_PAGE_ROW_SIGNATURES = (
    (2, 5, 4, 4, 2, 2, 4, 2, 2, 3, 3, 2, 2),
    (2, 1, 4, 1, 1, 1, 1, 1, 1, 2, 2, 1, 4, 4),
)


def _group_boxes_by_row(boxes):
    rows = []
    current = []
    for box in sorted(boxes, key=lambda item: item[1]):
        if not current:
            current.append(box)
            continue
        previous = current[-1]
        tolerance = max(box[3], previous[3]) * 0.5
        if abs(box[1] - previous[1]) <= tolerance:
            current.append(box)
        else:
            rows.append(current)
            current = [box]
    if current:
        rows.append(current)
    return rows


def _representative_groups(paths):
    groups = []
    for size in (2, 3):
        for start in range(0, len(paths), size):
            group = tuple(paths[start : start + size])
            if len(group) < 2:
                group = tuple(paths[-size:])
            if group not in groups:
                groups.append(group)
    all_paths = tuple(paths)
    if all_paths not in groups:
        groups.append(all_paths)
    return groups


@unittest.skipUnless(
    _RUN_CORPUS_TESTS,
    "set RUN_SCAN_CORPUS_TESTS=1 to run the external PDF corpus",
)
class ScanCorpusDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _SCAN_DIR.is_dir():
            raise AssertionError(f"scan corpus directory is missing: {_SCAN_DIR}")
        expected = set(_ONE_PAGE_PDFS) | set(_TWO_PAGE_PDFS)
        actual = {path.name for path in _SCAN_DIR.glob("*.pdf")}
        if actual != expected:
            missing = sorted(expected - actual)
            unclassified = sorted(actual - expected)
            raise AssertionError(
                f"scan corpus changed; missing={missing}, unclassified={unclassified}"
            )

    @staticmethod
    def _page_angles(path: Path, page_count: int):
        pages = load_pdf_pages(
            str(path),
            page_indices=list(range(page_count)),
        )
        return [float(estimate_deskew_angle(page) or 0.0) for page in pages]

    def _assert_layout(self, templates, page_count: int):
        self.assertEqual(set(templates), set(range(page_count)))
        detected_by_page = {
            page: auto_detect_checkboxes(template)
            for page, template in templates.items()
        }

        if page_count == 1:
            rows = _group_boxes_by_row(detected_by_page[0])
            small_rows = [
                row
                for row in rows
                if np.median([box[2] for box in row]) <= 50
                and np.median([box[3] for box in row]) <= 50
            ]
            grid_rows = [
                row
                for row in rows
                if len(row) >= 3
                and np.median([box[2] for box in row]) >= 100
                and np.median([box[3] for box in row]) >= 100
            ]
            self.assertEqual([len(row) for row in small_rows], [2, 6, 4])
            self.assertEqual([len(row) for row in grid_rows], [5] * 6)
            return

        signatures = tuple(
            tuple(len(row) for row in _group_boxes_by_row(detected_by_page[page]))
            for page in range(page_count)
        )
        self.assertEqual(signatures, _TWO_PAGE_ROW_SIGNATURES)

    def test_every_pdf_individually(self):
        for page_count, names in (
            (1, _ONE_PAGE_PDFS),
            (2, _TWO_PAGE_PDFS),
        ):
            for name in names:
                with self.subTest(pdf=name, page_count=page_count):
                    path = _SCAN_DIR / name
                    angles = self._page_angles(path, page_count)
                    templates = generate_ui_templates(
                        str(path),
                        page_count,
                        -1,
                        0.0,
                        page_fine_angles=angles,
                    )
                    self._assert_layout(templates, page_count)

    def test_compatible_pdfs_in_pairs_triples_and_all(self):
        for page_count, names in (
            (1, _ONE_PAGE_PDFS),
            (2, _TWO_PAGE_PDFS),
        ):
            paths = [_SCAN_DIR / name for name in names]
            for group in _representative_groups(paths):
                labels = tuple(path.name for path in group)
                with self.subTest(pdfs=labels, page_count=page_count):
                    angles = self._page_angles(group[0], page_count)
                    templates = generate_ui_templates_multi(
                        [str(path) for path in group],
                        page_count,
                        -1,
                        0.0,
                        page_fine_angles=angles,
                    )
                    self._assert_layout(templates, page_count)


if __name__ == "__main__":
    unittest.main()
