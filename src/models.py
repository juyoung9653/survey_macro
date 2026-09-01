from collections.abc import Iterable
from dataclasses import asdict, dataclass, field


RESERVED_FIELD_NAMES = frozenset({"파일명", "페이지"})


def validate_field_names(names: Iterable[str]) -> list[str]:
    """Return trimmed field names or raise for names that would corrupt output."""
    cleaned = ["" if name is None else str(name).strip() for name in names]
    if any(not name for name in cleaned):
        raise ValueError("모든 문항에 이름을 입력해주세요.")

    reserved_keys = {name.casefold() for name in RESERVED_FIELD_NAMES}
    reserved = [name for name in cleaned if name.casefold() in reserved_keys]
    hidden = [name for name in cleaned if name.startswith("__")]
    if reserved or hidden:
        invalid = list(dict.fromkeys([*reserved, *hidden]))
        raise ValueError(
            "결과 파일에서 사용하는 이름은 문항 이름으로 쓸 수 없습니다: "
            + ", ".join(invalid)
        )

    first_by_key: dict[str, str] = {}
    duplicates: list[str] = []
    for name in cleaned:
        key = name.casefold()
        if key in first_by_key:
            original = first_by_key[key]
            if original not in duplicates:
                duplicates.append(original)
        else:
            first_by_key[key] = name
    if duplicates:
        raise ValueError(
            "문항 이름은 서로 달라야 합니다. 중복된 이름: "
            + ", ".join(duplicates)
        )
    return cleaned


@dataclass
class Box:
    page_idx: int  # 템플릿 내 몇 번째 페이지인지
    x: int
    y: int
    w: int
    h: int
    is_checked: bool = False

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            page_idx=int(data.get("page_idx", 0)),
            x=int(data.get("x", 0)),
            y=int(data.get("y", 0)),
            w=int(data.get("w", 0)),
            h=int(data.get("h", 0)),
        )

    def to_dict(self):
        return asdict(self)


@dataclass
class Field:
    name: str
    boxes: list[Box] = field(default_factory=list)
    value_map: list[str] = field(default_factory=list)
    is_comment: bool = False
    allow_duplicates: bool = False
    show_average: bool = False

    @classmethod
    def from_dict(cls, data: dict):
        boxes = [Box.from_dict(b) for b in data.get("boxes", [])]
        raw_map = data.get("value_map", [])
        value_map = [str(v) for v in raw_map] if isinstance(raw_map, list) else []
        is_comment = bool(data.get("is_comment", False))
        allow_duplicates = bool(data.get("allow_duplicates", False))
        show_average = bool(data.get("show_average", False))
        return cls(
            name=str(data.get("name", "")),
            boxes=boxes,
            value_map=value_map,
            is_comment=is_comment,
            allow_duplicates=allow_duplicates,
            show_average=show_average,
        )

    def to_dict(self):
        return {
            "name": self.name,
            "boxes": [b.to_dict() for b in self.boxes],
            "value_map": self.value_map,
            "is_comment": self.is_comment,
            "allow_duplicates": self.allow_duplicates,
            "show_average": self.show_average,
        }


@dataclass
class TemplatePreset:
    page_count: int = 1
    fine_angle: float = 0.0
    rot_code: int = -1
    reverse_numbering: bool = True
    template_dilate_pct: float = 0.5
    fields: list[Field] = field(default_factory=list)
    page_fine_angles: list[float] = field(default_factory=list)

    def fine_angle_for_page(self, page_idx: int) -> float:
        page_angle = 0.0
        if 0 <= page_idx < len(self.page_fine_angles):
            page_angle = float(self.page_fine_angles[page_idx])
        return float(self.fine_angle) + page_angle
