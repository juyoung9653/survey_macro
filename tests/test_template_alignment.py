import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.models import Box, Field, TemplatePreset
from src.processor import (
    _CheckboxHaloInfo,
    _CheckboxInkInfo,
    _align_template_mask_by_coverage,
    _build_stable_region_mask,
    _checkbox_layout_is_trustworthy,
    _extract_checkbox_halo_info,
    _refine_checkbox_box,
    _remap_checkbox_layout,
    _resolve_checkbox_halo_ownership,
    evaluate_checkbox_marks,
    evaluate_marks,
    extract_checkbox_halo_ink_info,
    extract_checkbox_ink_info,
    extract_pure_ink_mask,
    process_survey_data,
    remap_preset_to_detected_layout,
)
from src.vision import ImageAligner, auto_detect_checkboxes


def _make_form(height: int = 700, width: int = 500) -> np.ndarray:
    image = np.full((height, width), 255, np.uint8)
    for y in range(90, height - 70, 70):
        cv2.line(image, (50, y), (width - 50, y), 0, 2)
        for x in range(70, width - 60, 70):
            cv2.rectangle(image, (x, y + 12), (x + 34, y + 46), 0, 2)
    return image


def _dark_mask(image: np.ndarray) -> np.ndarray:
    return cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)[1]


def _overlap(first: np.ndarray, second: np.ndarray) -> int:
    return cv2.countNonZero(cv2.bitwise_and(first, second))


class TemplateAlignmentTests(unittest.TestCase):
    def test_identity_alignment_is_unchanged(self):
        template = _dark_mask(_make_form())

        aligned = _align_template_mask_by_coverage(template, template.copy())

        self.assertTrue(np.array_equal(aligned, template))

    def test_coverage_alignment_improves_rotated_shifted_template(self):
        form = _make_form()
        height, width = form.shape
        template = _dark_mask(form)
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0.35, 1.0)
        matrix[0, 2] += 5
        matrix[1, 2] -= 4
        target = cv2.warpAffine(form, matrix, (width, height), borderValue=255)
        target_mask = _dark_mask(target)

        aligned = _align_template_mask_by_coverage(template, target_mask)

        self.assertGreater(_overlap(aligned, target_mask), _overlap(template, target_mask))

    def test_image_aligner_uses_one_coordinate_system_for_resized_pages(self):
        reference = _make_form()
        target = cv2.resize(reference, (750, 1050), interpolation=cv2.INTER_LINEAR)
        expected = cv2.resize(target, (500, 700), interpolation=cv2.INTER_AREA)
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)

        with (
            patch(
                "src.vision.cv2.estimateAffine2D",
                return_value=(identity.copy(), None),
            ) as estimate_affine,
            patch(
                "src.vision.cv2.findTransformECC",
                return_value=(0.999, identity.copy()),
            ) as find_ecc,
        ):
            aligned = ImageAligner(reference).align(target)

        estimate_affine.assert_called_once()
        find_ecc.assert_called_once()
        self.assertTrue(np.array_equal(aligned, expected))

    def test_image_aligner_inverts_ecc_warp_and_fills_white_border(self):
        reference = _make_form()
        ref_to_target = np.array(
            [[1.0, 0.0, 5.0], [0.0, 1.0, -4.0]], dtype=np.float32
        )
        target_to_ref = cv2.invertAffineTransform(ref_to_target)
        target = cv2.warpAffine(
            reference,
            ref_to_target,
            (reference.shape[1], reference.shape[0]),
            borderValue=255,
        )
        expected = cv2.warpAffine(
            target,
            target_to_ref,
            (reference.shape[1], reference.shape[0]),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

        with (
            patch(
                "src.vision.cv2.estimateAffine2D",
                return_value=(target_to_ref.copy(), None),
            ),
            patch(
                "src.vision.cv2.findTransformECC",
                return_value=(0.999, ref_to_target.copy()),
            ) as find_ecc,
        ):
            aligned = ImageAligner(reference).align(target)

        ecc_initial = find_ecc.call_args.args[2]
        self.assertTrue(np.allclose(ecc_initial, ref_to_target))
        self.assertTrue(np.array_equal(aligned, expected))
        self.assertEqual(int(aligned[0, 0]), 255)

    def test_image_aligner_can_skip_ecc_for_checkbox_template(self):
        reference = _make_form()
        identity = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )

        with (
            patch(
                "src.vision.cv2.estimateAffine2D",
                return_value=(identity.copy(), None),
            ),
            patch("src.vision.cv2.findTransformECC") as find_ecc,
        ):
            aligned = ImageAligner(reference, refine_ecc=False).align(reference)

        find_ecc.assert_not_called()
        self.assertTrue(np.array_equal(aligned, reference))

    def test_image_aligner_rejects_catastrophic_affine_matches(self):
        aligner = ImageAligner(_make_form(), refine_ecc=False)
        identity = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )
        mirrored = np.array(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )
        oversized = np.array(
            [[1.6, 0.0, 0.0], [0.0, 1.6, 0.0]], dtype=np.float32
        )
        far_away = np.array(
            [[1.0, 0.0, 100.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )

        self.assertTrue(aligner._is_plausible_affine(identity))
        self.assertFalse(aligner._is_plausible_affine(mirrored))
        self.assertFalse(aligner._is_plausible_affine(oversized))
        self.assertFalse(aligner._is_plausible_affine(far_away))

    def test_image_aligner_scales_large_ecc_inputs_and_restores_translation(self):
        reference = cv2.resize(_make_form(), (1000, 2000), interpolation=cv2.INTER_LINEAR)
        target_to_ref = np.array(
            [[1.0, 0.0, -8.0], [0.0, 1.0, 6.0]], dtype=np.float32
        )
        ref_to_target = cv2.invertAffineTransform(target_to_ref)

        with (
            patch(
                "src.vision.cv2.estimateAffine2D",
                return_value=(target_to_ref.copy(), None),
            ),
            patch(
                "src.vision.cv2.findTransformECC",
                side_effect=lambda _ref, _target, initial, *_args: (0.999, initial),
            ) as find_ecc,
        ):
            ImageAligner(reference).align(reference)

        ecc_reference, ecc_target, ecc_initial = find_ecc.call_args.args[:3]
        self.assertEqual(max(ecc_reference.shape), 1200)
        self.assertEqual(ecc_reference.shape, ecc_target.shape)
        self.assertAlmostEqual(ecc_initial[0, 2], ref_to_target[0, 2] * 0.6)
        self.assertAlmostEqual(ecc_initial[1, 2], ref_to_target[1, 2] * 0.6)

    def test_ink_is_preserved_and_checkbox_detection_still_works(self):
        form = _make_form()
        ink = np.zeros_like(form)
        cv2.line(ink, (82, 120), (110, 150), 255, 6)
        cv2.line(ink, (110, 150), (135, 105), 255, 6)
        target = form.copy()
        target[ink > 0] = 0

        extracted = extract_pure_ink_mask(target, form, template_dilate_pct=0.0)
        boxes = auto_detect_checkboxes(cv2.cvtColor(form, cv2.COLOR_GRAY2BGR))

        self.assertGreater(_overlap(extracted, ink), 50)
        self.assertGreater(len(boxes), 0)

    def test_checkbox_inner_ink_refines_shift_and_ignores_empty_border(self):
        expected = Box(page_idx=0, x=70, y=60, w=24, h=24)
        actual_x, actual_y = 75, 57
        empty = np.full((170, 250), 255, np.uint8)
        cv2.rectangle(
            empty,
            (actual_x, actual_y),
            (actual_x + 23, actual_y + 23),
            80,
            2,
        )
        marked = empty.copy()
        cv2.line(
            marked,
            (actual_x + 5, actual_y + 12),
            (actual_x + 10, actual_y + 17),
            35,
            2,
        )
        cv2.line(
            marked,
            (actual_x + 10, actual_y + 17),
            (actual_x + 19, actual_y + 5),
            35,
            2,
        )

        empty_info = extract_checkbox_ink_info(empty, expected)
        marked_info = extract_checkbox_ink_info(marked, expected)

        self.assertAlmostEqual(marked_info.box.x, actual_x, delta=1)
        self.assertAlmostEqual(marked_info.box.y, actual_y, delta=1)
        self.assertEqual(empty_info.ink_pixels, 0)
        self.assertGreater(marked_info.ink_pixels, 5)
        self.assertEqual(
            evaluate_checkbox_marks(
                [empty_info.ink_pixels, marked_info.ink_pixels],
                [empty_info.area, marked_info.area],
            ),
            [False, True],
        )

        border_residue = np.zeros_like(empty)
        cv2.rectangle(
            border_residue,
            (actual_x, actual_y),
            (actual_x + 23, actual_y + 23),
            255,
            2,
        )
        external_mark = border_residue.copy()
        cv2.line(
            external_mark,
            (actual_x + 20, actual_y + 21),
            (actual_x + 34, actual_y + 7),
            255,
            3,
        )

        empty_halo, _ = extract_checkbox_halo_ink_info(
            border_residue, marked_info.box
        )
        marked_halo, _ = extract_checkbox_halo_ink_info(
            external_mark, marked_info.box
        )
        self.assertEqual(empty_halo, 0)
        self.assertGreater(marked_halo, 5)

    def test_checkbox_direct_scoring_rejects_compact_four_pixel_dust(self):
        box = Box(page_idx=0, x=70, y=60, w=24, h=24)
        template = np.full((170, 250), 255, np.uint8)
        cv2.rectangle(template, (70, 60), (94, 84), 80, 2)
        dusty = template.copy()
        dusty[70:72, 80:82] = 20
        checked = template.copy()
        cv2.line(checked, (76, 72), (81, 77), 25, 2)
        cv2.line(checked, (81, 77), (89, 66), 25, 2)

        dust_info = extract_checkbox_ink_info(dusty, box, template)
        check_info = extract_checkbox_ink_info(checked, box, template)

        self.assertEqual(dust_info.ink_pixels, 0)
        self.assertEqual(dust_info.mark_strength, 0.0)
        self.assertGreater(check_info.ink_pixels, 5)
        self.assertGreater(check_info.mark_strength, 0.025)

    def test_alignment_stable_mask_excludes_response_boxes_and_halo(self):
        box = Box(page_idx=0, x=70, y=60, w=24, h=24)
        config = TemplatePreset(
            page_count=1,
            fields=[Field(name="Q", boxes=[box])],
        )

        mask = _build_stable_region_mask((170, 250), config, 0)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(int(mask[72, 82]), 0)
        self.assertEqual(int(mask[35, 35]), 255)

    def test_refined_checkbox_is_reused_for_halo_scoring(self):
        expected = Box(page_idx=0, x=70, y=60, w=24, h=24)
        page = np.full((150, 180), 255, np.uint8)
        cv2.rectangle(page, (74, 57), (97, 80), 70, 2)
        cv2.line(page, (80, 70), (91, 60), 20, 3)
        pure_ink = _dark_mask(page)

        with patch(
            "src.processor._refine_checkbox_box",
            wraps=_refine_checkbox_box,
        ) as refine:
            direct_info = extract_checkbox_ink_info(page, expected)
            _extract_checkbox_halo_info(
                pure_ink,
                direct_info.box,
                target_gray=page,
                box_is_refined=True,
            )

        self.assertEqual(refine.call_count, 1)

    def test_file_template_remaps_reflowed_checkbox_row_by_order(self):
        template = np.full((300, 420), 255, np.uint8)
        expected_positions = [
            (60, 70),
            (160, 70),
            (260, 70),
            (60, 160),
            (160, 160),
            (260, 160),
        ]
        actual_positions = [
            (60, 70),
            (130, 70),
            (310, 70),
            (60, 160),
            (160, 160),
            (260, 160),
        ]
        for x, y in actual_positions:
            cv2.rectangle(template, (x, y), (x + 23, y + 23), 0, 2)

        config = TemplatePreset(
            page_count=1,
            fields=[
                Field(
                    name="Q",
                    boxes=[Box(0, x, y, 24, 24) for x, y in expected_positions],
                    allow_duplicates=True,
                )
            ],
        )

        remapped = _remap_checkbox_layout(config, {0: template})
        actual_centers = [(x + 12, y + 12) for x, y in actual_positions]
        remapped_centers = [
            (box.x + box.w / 2, box.y + box.h / 2)
            for box in remapped.fields[0].boxes
        ]

        self.assertEqual(config.fields[0].boxes[1].x, 160)
        for remapped_center, actual_center in zip(remapped_centers, actual_centers):
            self.assertAlmostEqual(remapped_center[0], actual_center[0], delta=4)
            self.assertAlmostEqual(remapped_center[1], actual_center[1], delta=4)

    def test_preset_layout_uses_detected_boxes_across_render_dpi_changes(self):
        source = np.full((400, 300), 255, np.uint8)
        target = np.full((600, 450), 255, np.uint8)
        source_positions = [
            (50, 90),
            (120, 90),
            (190, 90),
            (50, 190),
            (120, 190),
            (190, 190),
        ]
        target_positions = [
            (round(x * 1.5 + 10), round(y * 1.5 - 12))
            for x, y in source_positions
        ]
        for x, y in target_positions:
            cv2.rectangle(target, (x, y), (x + 29, y + 29), 0, 2)

        config = TemplatePreset(
            page_count=1,
            fields=[
                Field(
                    name="Q",
                    boxes=[Box(0, x, y, 20, 20) for x, y in source_positions],
                ),
                Field(
                    name="의견",
                    boxes=[Box(0, 40, 280, 200, 60)],
                    is_comment=True,
                ),
            ],
        )
        auxiliary = [Box(0, 245, 300, 25, 25)]

        result = remap_preset_to_detected_layout(
            config,
            {0: target},
            source_templates={0: source},
            auxiliary_boxes=auxiliary,
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.matched_boxes, 6)
        for box, (target_x, target_y) in zip(
            result.config.fields[0].boxes, target_positions
        ):
            self.assertAlmostEqual(box.x, target_x, delta=4)
            self.assertAlmostEqual(box.y, target_y, delta=4)
        comment = result.config.fields[1].boxes[0]
        self.assertAlmostEqual(comment.x, 70, delta=3)
        self.assertAlmostEqual(comment.y, 408, delta=3)
        self.assertAlmostEqual(comment.w, 300, delta=4)
        pending = result.auxiliary_boxes[0]
        self.assertAlmostEqual(pending.x, 378, delta=3)
        self.assertAlmostEqual(pending.y, 438, delta=3)

    def test_preset_layout_accepts_matching_table_below_header_checkboxes(self):
        source = np.full((500, 400), 255, np.uint8)
        target = np.full_like(source, 255)
        checkbox_positions = [
            (50 + column * 70, 60 + row * 70)
            for row in range(2)
            for column in range(3)
        ]
        table_boxes = [
            Box(0, 100 + column * 70, 250 + row * 55, 70, 55)
            for row in range(3)
            for column in range(3)
        ]
        for image in (source, target):
            for x, y in checkbox_positions:
                cv2.rectangle(image, (x, y), (x + 19, y + 19), 0, 2)
            for box in table_boxes:
                cv2.rectangle(
                    image,
                    (box.x, box.y),
                    (box.x + box.w - 1, box.y + box.h - 1),
                    0,
                    2,
                )
        config = TemplatePreset(
            page_count=1,
            fields=[
                Field(
                    name="header",
                    boxes=[Box(0, x, y, 20, 20) for x, y in checkbox_positions],
                ),
                Field(name="table", boxes=table_boxes),
            ],
        )

        result = remap_preset_to_detected_layout(
            config,
            {0: target},
            source_templates={0: source},
        )

        self.assertTrue(result.accepted)
        self.assertTrue(result.compatible)

    def test_preset_layout_rejects_locally_shifted_table(self):
        source = np.full((500, 400), 255, np.uint8)
        target = np.full_like(source, 255)
        checkbox_positions = [
            (50 + column * 70, 60 + row * 70)
            for row in range(2)
            for column in range(3)
        ]
        table_boxes = [
            Box(0, 100 + column * 70, 250 + row * 55, 70, 55)
            for row in range(3)
            for column in range(3)
        ]
        for image in (source, target):
            for x, y in checkbox_positions:
                cv2.rectangle(image, (x, y), (x + 19, y + 19), 0, 2)
        for box in table_boxes:
            cv2.rectangle(
                source,
                (box.x, box.y),
                (box.x + box.w - 1, box.y + box.h - 1),
                0,
                2,
            )
            cv2.rectangle(
                target,
                (box.x + 28, box.y),
                (box.x + box.w + 27, box.y + box.h - 1),
                0,
                2,
            )
        config = TemplatePreset(
            page_count=1,
            fields=[
                Field(
                    name="header",
                    boxes=[Box(0, x, y, 20, 20) for x, y in checkbox_positions],
                ),
                Field(name="table", boxes=table_boxes),
            ],
        )

        result = remap_preset_to_detected_layout(
            config,
            {0: target},
            source_templates={0: source},
        )

        self.assertFalse(result.accepted)
        self.assertFalse(result.compatible)

    def test_preset_layout_rejects_incomplete_checkbox_detection(self):
        source = np.full((400, 300), 255, np.uint8)
        target = np.full((600, 450), 255, np.uint8)
        source_positions = [
            (50, 90),
            (120, 90),
            (190, 90),
            (50, 190),
            (120, 190),
            (190, 190),
        ]
        for x, y in source_positions[:-1]:
            target_x = round(x * 1.5 + 10)
            target_y = round(y * 1.5 - 12)
            cv2.rectangle(
                target,
                (target_x, target_y),
                (target_x + 29, target_y + 29),
                0,
                2,
            )
        config = TemplatePreset(
            page_count=1,
            fields=[
                Field(
                    name="Q",
                    boxes=[Box(0, x, y, 20, 20) for x, y in source_positions],
                )
            ],
        )
        auxiliary = [Box(0, 245, 300, 25, 25)]

        result = remap_preset_to_detected_layout(
            config,
            {0: target},
            source_templates={0: source},
            auxiliary_boxes=auxiliary,
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.config.fields[0].boxes[0].x, 50)
        self.assertEqual(result.auxiliary_boxes[0].x, 245)

    def test_preset_layout_rejects_a_missing_required_template_page(self):
        source_pages = {
            0: np.full((300, 220), 255, np.uint8),
            1: np.full((300, 220), 255, np.uint8),
        }
        target_page = np.full((450, 330), 255, np.uint8)
        fields = []
        positions = [
            (40, 70),
            (100, 70),
            (160, 70),
            (40, 160),
            (100, 160),
            (160, 160),
        ]
        for page_idx in range(2):
            boxes = []
            for x, y in positions:
                boxes.append(Box(page_idx, x, y, 20, 20))
                if page_idx == 0:
                    target_x = round(x * 1.5)
                    target_y = round(y * 1.5)
                    cv2.rectangle(
                        target_page,
                        (target_x, target_y),
                        (target_x + 29, target_y + 29),
                        0,
                        2,
                    )
            fields.append(Field(name=f"Q{page_idx + 1}", boxes=boxes))
        config = TemplatePreset(page_count=2, fields=fields)

        result = remap_preset_to_detected_layout(
            config,
            {0: target_page},
            source_templates=source_pages,
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.expected_boxes, 12)
        self.assertEqual(result.page_transforms, {})

    def test_preset_layout_interpolates_a_few_obscured_checkbox_frames(self):
        source = np.full((400, 300), 255, np.uint8)
        target = np.full((600, 450), 255, np.uint8)
        source_positions = [
            (50 + column * 70, 60 + row * 70)
            for row in range(4)
            for column in range(3)
        ]
        missing_index = 7
        target_positions = [
            (round(x * 1.4 + 18), round(y * 1.4 - 6))
            for x, y in source_positions
        ]
        for index, (x, y) in enumerate(target_positions):
            if index == missing_index:
                continue
            cv2.rectangle(target, (x, y), (x + 27, y + 27), 0, 2)

        config = TemplatePreset(
            page_count=1,
            fields=[
                Field(
                    name="Q",
                    boxes=[Box(0, x, y, 20, 20) for x, y in source_positions],
                ),
                Field(
                    name="comment",
                    boxes=[Box(0, 40, 340, 200, 40)],
                    is_comment=True,
                ),
            ],
        )

        result = remap_preset_to_detected_layout(
            config,
            {0: target},
            source_templates={0: source},
        )

        self.assertTrue(result.accepted)
        self.assertTrue(result.compatible)
        self.assertEqual(result.matched_boxes, 11)
        for index, (box, (target_x, target_y)) in enumerate(
            zip(result.config.fields[0].boxes, target_positions)
        ):
            tolerance = 5 if index == missing_index else 4
            self.assertAlmostEqual(box.x, target_x, delta=tolerance)
            self.assertAlmostEqual(box.y, target_y, delta=tolerance)

    def test_preset_layout_marks_a_different_form_as_incompatible(self):
        source = np.full((400, 300), 255, np.uint8)
        target = np.full((600, 450), 255, np.uint8)
        source_positions = [
            (50 + column * 70, 60 + row * 70)
            for row in range(4)
            for column in range(3)
        ]
        for x, y in [(80, 420), (160, 420), (240, 420)]:
            cv2.rectangle(target, (x, y), (x + 27, y + 27), 0, 2)
        config = TemplatePreset(
            page_count=1,
            fields=[
                Field(
                    name="Q",
                    boxes=[Box(0, x, y, 20, 20) for x, y in source_positions],
                )
            ],
        )

        result = remap_preset_to_detected_layout(
            config,
            {0: target},
            source_templates={0: source},
        )

        self.assertFalse(result.accepted)
        self.assertFalse(result.compatible)
        self.assertLess(result.matched_boxes, len(source_positions) * 0.45)

    def test_checkbox_layout_is_trusted_only_when_every_frame_is_detected(self):
        template = np.full((300, 420), 255, np.uint8)
        positions = [
            (60, 70),
            (160, 70),
            (260, 70),
            (60, 160),
            (160, 160),
            (260, 160),
        ]
        for x, y in positions:
            cv2.rectangle(template, (x, y), (x + 23, y + 23), 0, 2)
        config = TemplatePreset(
            page_count=1,
            fields=[
                Field(
                    name="Q",
                    boxes=[Box(0, x, y, 24, 24) for x, y in positions],
                    allow_duplicates=True,
                )
            ],
        )

        remapped = _remap_checkbox_layout(config, {0: template})
        incomplete = template.copy()
        incomplete[65:100, 255:295] = 255

        self.assertTrue(_checkbox_layout_is_trustworthy(remapped, {0: template}))
        self.assertFalse(
            _checkbox_layout_is_trustworthy(remapped, {0: incomplete})
        )

    def test_repeated_mark_is_detected_when_median_template_contains_it(self):
        page = np.full((220, 320), 255, np.uint8)
        first_box = Box(page_idx=0, x=70, y=90, w=24, h=24)
        second_box = Box(page_idx=0, x=160, y=90, w=24, h=24)
        for box in (first_box, second_box):
            cv2.rectangle(
                page,
                (box.x, box.y),
                (box.x + box.w - 1, box.y + box.h - 1),
                80,
                2,
            )
        cv2.line(page, (75, 102), (81, 108), 30, 2)
        cv2.line(page, (81, 108), (89, 96), 30, 2)

        # 모든 표본에 같은 체크가 있으면 중앙값 템플릿도 체크를 포함합니다.
        contaminated_template = page.copy()
        old_mask = extract_pure_ink_mask(
            page, contaminated_template, template_dilate_pct=0.0
        )
        config = TemplatePreset(
            page_count=1,
            reverse_numbering=False,
            template_dilate_pct=0.0,
            fields=[
                Field(
                    name="Q",
                    boxes=[first_box, second_box],
                    value_map=["첫째", "둘째"],
                )
            ],
        )

        row, _, ink_images, _, annotations, _ = process_survey_data(
            {
                "fname": "sample",
                "row_title": "sample_1p",
                "gray_pages": {0: page},
            },
            config,
            {0: contaminated_template},
        )

        self.assertEqual(cv2.countNonZero(old_mask), 0)
        self.assertEqual(row["Q"], "첫째")
        self.assertGreater(cv2.countNonZero(cv2.bitwise_not(ink_images[0])), 0)
        self.assertEqual([item[-1] for item in annotations[0]], [True, False])

    def test_checkbox_row_remap_uses_global_one_to_one_matching_when_neighbor_missing(
        self,
    ):
        template = np.full((220, 240), 255, np.uint8)
        original_first_row = [(50, 70), (110, 70)]
        original_second_row = [(50, 100), (110, 100)]
        detected_second_row = [(60, 100), (130, 100)]
        for x, y in original_first_row + detected_second_row:
            cv2.rectangle(template, (x, y), (x + 23, y + 23), 80, 2)

        config = TemplatePreset(
            page_count=1,
            fields=[
                Field(
                    name="Q1",
                    boxes=[Box(0, x, y, 24, 24) for x, y in original_first_row],
                    value_map=["A1", "A2"],
                ),
                Field(
                    name="Q2",
                    boxes=[Box(0, x, y, 24, 24) for x, y in original_second_row],
                    value_map=["B1", "B2"],
                ),
            ],
        )

        with patch(
            "src.processor.auto_detect_checkboxes",
            return_value=[(x, y, 24, 24) for x, y in detected_second_row],
        ):
            remapped = _remap_checkbox_layout(config, {0: template})

        remapped_positions = {
            field.name: [(box.x, box.y) for box in field.boxes]
            for field in remapped.fields
        }
        self.assertEqual(
            remapped_positions,
            {
                "Q1": original_first_row,
                "Q2": detected_second_row,
            },
        )
        self.assertEqual(remapped.fields[0].value_map, ["A1", "A2"])
        self.assertEqual(remapped.fields[1].value_map, ["B1", "B2"])

    def test_checkbox_halo_rejects_shifted_empty_border_and_accepts_connected_check(
        self,
    ):
        expected = Box(page_idx=0, x=70, y=60, w=24, h=24)
        actual_x, actual_y = 75, 57
        empty_page = np.full((150, 220), 255, np.uint8)
        cv2.rectangle(
            empty_page,
            (actual_x, actual_y),
            (actual_x + 23, actual_y + 23),
            80,
            2,
        )
        empty_residue = np.zeros_like(empty_page)
        cv2.rectangle(
            empty_residue,
            (actual_x, actual_y),
            (actual_x + 23, actual_y + 23),
            255,
            2,
        )

        marked_page = empty_page.copy()
        marked_residue = empty_residue.copy()
        check_start = (actual_x + 20, actual_y + 21)
        check_end = (actual_x + 34, actual_y + 7)
        cv2.line(marked_page, check_start, check_end, 30, 3)
        cv2.line(marked_residue, check_start, check_end, 255, 3)

        empty_ink, empty_area = extract_checkbox_halo_ink_info(
            empty_residue,
            expected,
            target_gray=empty_page,
        )
        marked_ink, marked_area = extract_checkbox_halo_ink_info(
            marked_residue,
            expected,
            target_gray=marked_page,
        )

        self.assertEqual(
            evaluate_marks([empty_ink], [empty_area], is_contiguous=False),
            [False],
        )
        self.assertEqual(
            evaluate_marks([marked_ink], [marked_area], is_contiguous=False),
            [True],
        )

    def test_checkbox_halo_rejects_a_compact_scan_speck_near_frame(self):
        box = Box(page_idx=0, x=90, y=80, w=19, h=19)
        page = np.full((180, 240), 255, np.uint8)
        residue = np.zeros_like(page)
        cv2.rectangle(
            page,
            (box.x, box.y),
            (box.x + box.w - 1, box.y + box.h - 1),
            80,
            2,
        )
        cv2.rectangle(
            page,
            (box.x - 7, box.y - 11),
            (box.x - 3, box.y - 3),
            20,
            -1,
        )
        cv2.rectangle(
            residue,
            (box.x - 7, box.y - 11),
            (box.x - 3, box.y - 3),
            255,
            -1,
        )

        ink, _area = extract_checkbox_halo_ink_info(
            residue,
            box,
            target_gray=page,
        )

        self.assertEqual(ink, 0)

    def test_checkbox_halo_keeps_a_compact_tail_connected_to_interior_ink(self):
        box = Box(page_idx=0, x=90, y=80, w=19, h=19)
        page = np.full((180, 240), 255, np.uint8)
        residue = np.zeros_like(page)
        cv2.rectangle(
            page,
            (box.x, box.y),
            (box.x + box.w - 1, box.y + box.h - 1),
            80,
            2,
        )
        cv2.rectangle(
            page,
            (box.x - 3, box.y - 9),
            (box.x + 2, box.y - 2),
            20,
            -1,
        )
        cv2.line(
            page,
            (box.x, box.y - 2),
            (box.x + 9, box.y + 10),
            20,
            3,
        )
        residue[page < 60] = 255

        ink, _area = extract_checkbox_halo_ink_info(
            residue,
            box,
            target_gray=page,
        )

        self.assertGreater(ink, 0)

    def test_connected_halo_stroke_is_owned_by_the_box_it_marks(self):
        boxes = [
            Box(page_idx=0, x=70, y=60, w=24, h=24),
            Box(page_idx=0, x=70, y=110, w=24, h=24),
            Box(page_idx=0, x=160, y=110, w=24, h=24),
        ]
        template = np.full((190, 240), 255, np.uint8)
        for box in boxes:
            cv2.rectangle(
                template,
                (box.x, box.y),
                (box.x + box.w - 1, box.y + box.h - 1),
                80,
                2,
            )

        page = template.copy()
        cv2.line(page, (91, 131), (104, 119), 30, 3)
        cv2.line(page, (104, 119), (91, 82), 30, 3)
        config = TemplatePreset(
            page_count=1,
            template_dilate_pct=0.0,
            fields=[
                Field(
                    name="Q",
                    boxes=boxes,
                    value_map=["grazed-blank", "actual", "empty"],
                    allow_duplicates=True,
                )
            ],
        )

        row, _, _, _, _, _ = process_survey_data(
            {
                "fname": "sample",
                "row_title": "sample_1p",
                "gray_pages": {0: page},
            },
            config,
            {0: template},
        )

        self.assertEqual(row["Q"], "actual")

    def test_direct_stroke_crossing_a_blank_box_is_owned_by_the_actual_box(self):
        upper_box = Box(page_idx=0, x=70, y=60, w=24, h=24)
        lower_box = Box(page_idx=0, x=70, y=110, w=24, h=24)
        template = np.full((180, 180), 255, np.uint8)
        for box in (upper_box, lower_box):
            cv2.rectangle(
                template,
                (box.x, box.y),
                (box.x + box.w - 1, box.y + box.h - 1),
                80,
                2,
            )

        page = template.copy()
        cv2.line(page, (75, 123), (81, 130), 30, 3)
        cv2.line(page, (81, 130), (89, 72), 30, 3)
        config = TemplatePreset(
            page_count=1,
            reverse_numbering=False,
            template_dilate_pct=0.0,
            fields=[
                Field(
                    name="Q",
                    boxes=[upper_box, lower_box],
                    value_map=["crossed-blank", "actual"],
                    allow_duplicates=True,
                )
            ],
        )

        row, _, _, _, _, _ = process_survey_data(
            {
                "fname": "sample",
                "row_title": "sample_1p",
                "gray_pages": {0: page},
            },
            config,
            {0: template},
            trust_checkbox_layout=True,
        )

        self.assertEqual(row["Q"], "actual")

    def test_connected_checks_with_substantial_local_marks_keep_both_boxes(self):
        upper_box = Box(page_idx=0, x=70, y=60, w=24, h=24)
        lower_box = Box(page_idx=0, x=70, y=110, w=24, h=24)
        template = np.full((180, 180), 255, np.uint8)
        for box in (upper_box, lower_box):
            cv2.rectangle(
                template,
                (box.x, box.y),
                (box.x + box.w - 1, box.y + box.h - 1),
                80,
                2,
            )

        page = template.copy()
        cv2.line(page, (62, 62), (80, 78), 30, 3)
        cv2.line(page, (80, 78), (100, 53), 30, 3)
        cv2.line(page, (100, 53), (100, 103), 30, 3)
        cv2.line(page, (62, 112), (80, 128), 30, 3)
        cv2.line(page, (80, 128), (100, 103), 30, 3)
        config = TemplatePreset(
            page_count=1,
            reverse_numbering=False,
            template_dilate_pct=0.0,
            fields=[
                Field(
                    name="Q",
                    boxes=[upper_box, lower_box],
                    value_map=["upper", "lower"],
                    allow_duplicates=True,
                )
            ],
        )

        row, _, _, _, _, _ = process_survey_data(
            {
                "fname": "sample",
                "row_title": "sample_1p",
                "gray_pages": {0: page},
            },
            config,
            {0: template},
            trust_checkbox_layout=True,
        )

        self.assertEqual(row["Q"], "upper,lower")

    def test_halo_stroke_can_be_owned_by_a_direct_only_neighbor(self):
        upper_box = Box(page_idx=0, x=70, y=60, w=24, h=24)
        lower_box = Box(page_idx=0, x=70, y=110, w=24, h=24)
        page = np.full((180, 180), 255, np.uint8)
        cv2.line(page, (82, 75), (82, 125), 30, 3)
        pure_ink = _dark_mask(page)

        upper_direct_mask = np.zeros((12, 12), np.uint8)
        upper_direct = _CheckboxInkInfo(
            0,
            upper_direct_mask.size,
            upper_box,
            0.8,
            (76, 66, 88, 78),
            upper_direct_mask,
        )
        upper_halo_mask = np.zeros((50, 35), np.uint8)
        cv2.line(upper_halo_mask, (17, 20), (17, 49), 255, 3)
        upper_halo = _CheckboxHaloInfo(
            cv2.countNonZero(upper_halo_mask),
            upper_halo_mask.size,
            upper_box,
            (65, 55, 100, 105),
            upper_halo_mask,
        )

        lower_direct_mask = np.zeros((12, 12), np.uint8)
        cv2.line(lower_direct_mask, (6, 0), (6, 11), 255, 3)
        lower_direct = _CheckboxInkInfo(
            cv2.countNonZero(lower_direct_mask),
            lower_direct_mask.size,
            lower_box,
            0.8,
            (76, 116, 88, 128),
            lower_direct_mask,
        )
        lower_halo_mask = np.zeros((50, 35), np.uint8)
        lower_halo = _CheckboxHaloInfo(
            0,
            lower_halo_mask.size,
            lower_box,
            (65, 105, 100, 155),
            lower_halo_mask,
        )

        _resolve_checkbox_halo_ownership(
            {0: pure_ink},
            {0: page},
            [
                (upper_direct, False, upper_halo),
                (lower_direct, True, lower_halo),
            ],
        )

        self.assertEqual(upper_halo.ink_pixels, 0)
        self.assertGreater(lower_direct.ink_pixels, 0)

    def test_separate_external_checks_keep_separate_checkbox_owners(self):
        boxes = [
            Box(page_idx=0, x=70, y=60, w=24, h=24),
            Box(page_idx=0, x=70, y=110, w=24, h=24),
            Box(page_idx=0, x=160, y=110, w=24, h=24),
        ]
        template = np.full((190, 240), 255, np.uint8)
        for box in boxes:
            cv2.rectangle(
                template,
                (box.x, box.y),
                (box.x + box.w - 1, box.y + box.h - 1),
                80,
                2,
            )

        page = template.copy()
        cv2.line(page, (91, 81), (104, 68), 30, 3)
        cv2.line(page, (91, 131), (104, 118), 30, 3)
        config = TemplatePreset(
            page_count=1,
            reverse_numbering=False,
            template_dilate_pct=0.0,
            fields=[
                Field(
                    name="Q",
                    boxes=boxes,
                    value_map=["first", "second", "empty"],
                    allow_duplicates=True,
                )
            ],
        )

        row, _, _, _, _, _ = process_survey_data(
            {
                "fname": "sample",
                "row_title": "sample_1p",
                "gray_pages": {0: page},
            },
            config,
            {0: template},
        )

        self.assertEqual(row["Q"], "first,second")

    def test_neighbor_field_check_does_not_mark_grazed_blank_box(self):
        marked_box = Box(page_idx=0, x=70, y=60, w=24, h=24)
        blank_box = Box(page_idx=0, x=70, y=104, w=24, h=24)
        template = np.full((170, 220), 255, np.uint8)
        for box in (marked_box, blank_box):
            cv2.rectangle(
                template,
                (box.x, box.y),
                (box.x + box.w - 1, box.y + box.h - 1),
                80,
                2,
            )

        page = template.copy()
        cv2.line(page, (76, 71), (82, 78), 30, 3)
        cv2.line(page, (82, 78), (91, 105), 30, 3)
        config = TemplatePreset(
            page_count=1,
            template_dilate_pct=0.0,
            fields=[
                Field(name="A", boxes=[marked_box], value_map=["marked"]),
                Field(name="B", boxes=[blank_box], value_map=["false-positive"]),
            ],
        )

        row, _, _, _, _, _ = process_survey_data(
            {
                "fname": "sample",
                "row_title": "sample_1p",
                "gray_pages": {0: page},
            },
            config,
            {0: template},
        )

        self.assertEqual(row["A"], "marked")
        self.assertEqual(row["B"], "")

    def test_single_choice_prefers_strong_halo_check_over_three_pixel_direct_noise(
        self,
    ):
        template = np.full((220, 320), 255, np.uint8)
        boxes = [
            Box(page_idx=0, x=70, y=90, w=24, h=24),
            Box(page_idx=0, x=160, y=90, w=24, h=24),
        ]
        for box in boxes:
            cv2.rectangle(
                template,
                (box.x, box.y),
                (box.x + box.w - 1, box.y + box.h - 1),
                80,
                2,
            )

        page = template.copy()
        page[102, 80:83] = 30
        cv2.line(page, (180, 111), (194, 97), 30, 3)
        config = TemplatePreset(
            page_count=1,
            reverse_numbering=False,
            template_dilate_pct=0.0,
            fields=[
                Field(
                    name="Q",
                    boxes=boxes,
                    value_map=["noise", "actual"],
                )
            ],
        )

        row, _, _, _, _, _ = process_survey_data(
            {
                "fname": "sample",
                "row_title": "sample_1p",
                "gray_pages": {0: page},
            },
            config,
            {0: template},
        )

        self.assertEqual(row["Q"], "actual")

    def test_open_line_intersection_is_not_a_reliable_checkbox(self):
        page = np.full((140, 180), 255, np.uint8)
        expected = Box(page_idx=0, x=70, y=60, w=24, h=24)
        cv2.line(page, (76, 55), (76, 90), 20, 2)
        cv2.line(page, (60, 66), (100, 66), 20, 2)
        cv2.circle(page, (86, 76), 2, 20, -1)
        config = TemplatePreset(
            page_count=1,
            template_dilate_pct=0.0,
            fields=[
                Field(
                    name="Q",
                    boxes=[expected],
                    value_map=["false-positive"],
                )
            ],
        )

        row, _, _, _, _, _ = process_survey_data(
            {
                "fname": "sample",
                "row_title": "sample_1p",
                "gray_pages": {0: page},
            },
            config,
            {0: page.copy()},
        )

        self.assertEqual(row["Q"], "")

    def test_repeated_external_check_is_recovered_from_original_page(self):
        page = np.full((220, 320), 255, np.uint8)
        first_box = Box(page_idx=0, x=70, y=90, w=24, h=24)
        second_box = Box(page_idx=0, x=160, y=90, w=24, h=24)
        for box in (first_box, second_box):
            cv2.rectangle(
                page,
                (box.x, box.y),
                (box.x + box.w - 1, box.y + box.h - 1),
                80,
                2,
            )
        cv2.line(page, (90, 111), (104, 97), 30, 3)

        contaminated_template = page.copy()
        pure_ink = extract_pure_ink_mask(
            page,
            contaminated_template,
            template_dilate_pct=0.0,
        )
        config = TemplatePreset(
            page_count=1,
            reverse_numbering=False,
            template_dilate_pct=0.0,
            fields=[
                Field(
                    name="Q",
                    boxes=[first_box, second_box],
                    value_map=["edge-check", "empty"],
                )
            ],
        )

        row, _, _, _, _, _ = process_survey_data(
            {
                "fname": "sample",
                "row_title": "sample_1p",
                "gray_pages": {0: page},
            },
            config,
            {0: contaminated_template},
        )

        self.assertEqual(cv2.countNonZero(pure_ink), 0)
        self.assertEqual(row["Q"], "edge-check")

    def test_multi_response_direct_scoring_rejects_three_pixel_noise(self):
        self.assertEqual(
            evaluate_checkbox_marks(
                [3, 30],
                [14 * 14, 14 * 14],
                strict=True,
            ),
            [False, True],
        )

    def test_multi_response_strength_does_not_depend_on_strongest_mark(self):
        self.assertEqual(
            evaluate_checkbox_marks(
                [8, 40],
                [14 * 14, 14 * 14],
                strict=True,
                strengths=[0.04, 0.20],
            ),
            [True, True],
        )

    def test_multi_response_strength_rejects_four_pixel_residual(self):
        self.assertEqual(
            evaluate_checkbox_marks(
                [4, 64],
                [12 * 12, 12 * 12],
                strict=True,
                strengths=[0.02972, 0.44444],
            ),
            [False, True],
        )

    def test_trusted_file_layout_recovers_check_from_faded_page(self):
        page = np.full((220, 320), 255, np.uint8)
        first_box = Box(page_idx=0, x=70, y=90, w=24, h=24)
        second_box = Box(page_idx=0, x=160, y=90, w=24, h=24)
        for box in (first_box, second_box):
            cv2.rectangle(
                page,
                (box.x, box.y),
                (box.x + box.w - 1, box.y + box.h - 1),
                220,
                1,
            )
        cv2.line(page, (76, 102), (81, 107), 30, 2)
        cv2.line(page, (81, 107), (88, 97), 30, 2)

        info = extract_checkbox_ink_info(page, first_box)
        config = TemplatePreset(
            page_count=1,
            reverse_numbering=False,
            template_dilate_pct=0.0,
            fields=[
                Field(
                    name="Q",
                    boxes=[first_box, second_box],
                    value_map=["faded-check", "empty"],
                )
            ],
        )

        row, _, _, _, _, _ = process_survey_data(
            {
                "fname": "sample",
                "row_title": "sample_1p",
                "gray_pages": {0: page},
            },
            config,
            {0: page.copy()},
            trust_checkbox_layout=True,
        )

        self.assertLess(info.border_confidence, 0.35)
        self.assertEqual(row["Q"], "faded-check")


if __name__ == "__main__":
    unittest.main()
