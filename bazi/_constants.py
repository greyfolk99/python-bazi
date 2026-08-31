"""사주팔자 도메인 상수."""
from __future__ import annotations

STEMS = ("甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸")
BRANCHES = ("子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥")
ELEMENTS = ("wood", "fire", "earth", "metal", "water")

STEM_INDEX: dict[str, int] = {v: i for i, v in enumerate(STEMS)}
BRANCH_INDEX: dict[str, int] = {v: i for i, v in enumerate(BRANCHES)}
ELEMENT_INDEX: dict[str, int] = {v: i for i, v in enumerate(ELEMENTS)}

STEM_ELEMENTS: dict[str, str] = {
    "甲": "wood", "乙": "wood",
    "丙": "fire", "丁": "fire",
    "戊": "earth", "己": "earth",
    "庚": "metal", "辛": "metal",
    "壬": "water", "癸": "water",
}

BRANCH_ELEMENTS: dict[str, str] = {
    "子": "water", "丑": "earth", "寅": "wood", "卯": "wood",
    "辰": "earth", "巳": "fire", "午": "fire", "未": "earth",
    "申": "metal", "酉": "metal", "戌": "earth", "亥": "water",
}

HIDDEN_STEMS: dict[str, tuple[str, ...]] = {
    "子": ("癸",),
    "丑": ("己", "癸", "辛"),
    "寅": ("甲", "丙", "戊"),
    "卯": ("乙",),
    "辰": ("戊", "乙", "癸"),
    "巳": ("丙", "戊", "庚"),
    "午": ("丁", "己"),
    "未": ("己", "丁", "乙"),
    "申": ("庚", "壬", "戊"),
    "酉": ("辛",),
    "戌": ("戊", "辛", "丁"),
    "亥": ("壬", "甲"),
}

STEM_COMBINATIONS: frozenset[frozenset[str]] = frozenset({
    frozenset(("甲", "己")), frozenset(("乙", "庚")),
    frozenset(("丙", "辛")), frozenset(("丁", "壬")),
    frozenset(("戊", "癸")),
})

BRANCH_LIUHE: frozenset[frozenset[str]] = frozenset({
    frozenset(("子", "丑")), frozenset(("寅", "亥")),
    frozenset(("卯", "戌")), frozenset(("辰", "酉")),
    frozenset(("巳", "申")), frozenset(("午", "未")),
})

BRANCH_CLASH: frozenset[frozenset[str]] = frozenset({
    frozenset(("子", "午")), frozenset(("丑", "未")),
    frozenset(("寅", "申")), frozenset(("卯", "酉")),
    frozenset(("辰", "戌")), frozenset(("巳", "亥")),
})

BRANCH_HARM: frozenset[frozenset[str]] = frozenset({
    frozenset(("子", "未")), frozenset(("丑", "午")),
    frozenset(("寅", "巳")), frozenset(("卯", "辰")),
    frozenset(("申", "亥")), frozenset(("酉", "戌")),
})

GENERATES: dict[str, str] = {
    "wood": "fire", "fire": "earth", "earth": "metal",
    "metal": "water", "water": "wood",
}

CONTROLS: dict[str, str] = {
    "wood": "earth", "earth": "water", "water": "fire",
    "fire": "metal", "metal": "wood",
}

CONTROLLED_BY: dict[str, str] = {controlled: controller for controller, controlled in CONTROLS.items()}
