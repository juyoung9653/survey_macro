from dataclasses import asdict, dataclass, field


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
    reverse_numbering: bool = False
    fields: list[Field] = field(default_factory=list)
