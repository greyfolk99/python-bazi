from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from ._config import config
from ._engine import BaziChart, STEMS, BRANCHES
from ._tables import (
    STEM_ELEMENT, STEM_YIN_YANG,
    BRANCH_ELEMENT, BRANCH_YIN_YANG,
    BRANCH_HIDDEN_STEMS,
    SHI_SHEN_NAMES as _SHI_SHEN_NAMES_BY_LANG,
)

_GENERATES = {0: 1, 1: 2, 2: 3, 3: 4, 4: 0}  # 木→火→土→金→水→木
_CONTROLS  = {0: 2, 1: 3, 2: 4, 3: 0, 4: 1}  # 木克土 火克金 土克水 金克木 水克火

_DAY_LABEL = {"zh": "日元", "ko": "일원", "en": "Self"}


def _ss_name(idx: int, lang: str) -> str:
    return _SHI_SHEN_NAMES_BY_LANG[lang][idx]


def _shi_shen_idx(day_stem_idx: int, other_stem_idx: int) -> int:
    """일간 인덱스 × 타간 인덱스 → 십성 인덱스 (0~9)."""
    d_elem = day_stem_idx // 2
    o_elem = other_stem_idx // 2
    same_yin_yang = (day_stem_idx % 2) == (other_stem_idx % 2)

    if d_elem == o_elem:
        return 0 if same_yin_yang else 1
    elif _GENERATES[d_elem] == o_elem:
        return 2 if same_yin_yang else 3
    elif _CONTROLS[d_elem] == o_elem:
        return 4 if same_yin_yang else 5
    elif _CONTROLS[o_elem] == d_elem:
        return 6 if same_yin_yang else 7
    else:
        return 8 if same_yin_yang else 9


@dataclass(frozen=True)
class ElementProfile:
    counts: dict[str, int]
    ratios: dict[str, float]

    @property
    def dominant(self) -> str:
        return max(self.counts, key=self.counts.get)  # type: ignore[arg-type]


@dataclass(frozen=True)
class PillarDetail:
    pillar_name: str
    stem: str
    branch: str
    stem_element: str
    branch_element: str
    stem_yin_yang: str
    branch_yin_yang: str
    stem_shi_shen: str
    branch_shi_shen: str
    hidden_stems: list[str]


@dataclass(frozen=True)
class DaYun:
    start_age: int
    stem: str
    branch: str
    stem_element: str
    branch_element: str
    stem_shi_shen: str
    branch_shi_shen: str


@dataclass(frozen=True)
class ChartAnalysis:
    chart: BaziChart
    elements: ElementProfile
    pillars: dict[str, PillarDetail]
    dayun: list[DaYun]


def analyze(
    chart: BaziChart,
    *,
    lang: str | None = None,
    sex: Literal["male", "female"] | None = None,
    birth: date | None = None,
    dayun_count: int = 8,
) -> ChartAnalysis:
    """팔자 분석.

    Args:
        chart: BaziChart (bazi.chart()로 생성)
        lang: 출력 언어 ('zh'|'ko'|'en'). None이면 bazi.config.lang 사용.
        sex: 성별 (대운 계산에 사용)
        birth: 생년월일 (대운 계산 시 필요)
        dayun_count: 뽑을 대운 개수 (기본 8개)
    """
    effective_lang = lang or config.lang

    pillars_raw: dict[str, "Pillar"] = {
        "year": chart.year,
        "month": chart.month,
        "day": chart.day,
    }
    if chart.hour is not None:
        pillars_raw["hour"] = chart.hour
    day_stem_idx = STEMS.index(chart.day.stem)

    element_counts: dict[str, int] = {"木": 0, "火": 0, "土": 0, "金": 0, "水": 0}
    for p in pillars_raw.values():
        element_counts[STEM_ELEMENT[STEMS.index(p.stem)]] += 1
        element_counts[BRANCH_ELEMENT[BRANCHES.index(p.branch)]] += 1
    total = sum(element_counts.values())
    ratios = {k: v / total for k, v in element_counts.items()}

    pillar_details: dict[str, PillarDetail] = {}
    for key, p in pillars_raw.items():
        s_idx = STEMS.index(p.stem)
        b_idx = BRANCHES.index(p.branch)
        hidden = [STEMS[i] for i in BRANCH_HIDDEN_STEMS[b_idx]]
        main_hidden_idx = STEMS.index(hidden[-1])

        if key == "day":
            stem_ss = _DAY_LABEL[effective_lang]
            branch_ss = _ss_name(_shi_shen_idx(day_stem_idx, main_hidden_idx), effective_lang)
        else:
            stem_ss = _ss_name(_shi_shen_idx(day_stem_idx, s_idx), effective_lang)
            branch_ss = _ss_name(_shi_shen_idx(day_stem_idx, main_hidden_idx), effective_lang)

        pillar_details[key] = PillarDetail(
            pillar_name=key,
            stem=p.stem,
            branch=p.branch,
            stem_element=STEM_ELEMENT[s_idx],
            branch_element=BRANCH_ELEMENT[b_idx],
            stem_yin_yang=STEM_YIN_YANG[s_idx],
            branch_yin_yang=BRANCH_YIN_YANG[b_idx],
            stem_shi_shen=stem_ss,
            branch_shi_shen=branch_ss,
            hidden_stems=hidden,
        )

    dayun_list: list[DaYun] = []
    if birth is not None:
        dayun_list = _calc_dayun(chart, birth, sex, day_stem_idx, dayun_count, effective_lang)

    return ChartAnalysis(
        chart=chart,
        elements=ElementProfile(counts=element_counts, ratios=ratios),
        pillars=pillar_details,
        dayun=dayun_list,
    )


def _calc_dayun(
    chart: BaziChart,
    birth: date,
    sex: str,
    day_stem_idx: int,
    count: int,
    lang: str,
) -> list[DaYun]:
    from ._engine import _engine, _EPOCH_ORD
    import numpy as np

    e = _engine()
    year_stem_idx = STEMS.index(chart.year.stem)
    year_yang = (year_stem_idx % 2 == 0)

    forward = (year_yang and sex == "male") or (not year_yang and sex == "female")

    birth_ord = birth.toordinal()
    birth_minute = (birth_ord - _EPOCH_ORD) * 1440

    pos = np.searchsorted(e._jie_min, np.array([birth_minute], dtype=np.int64), side="left") - 1
    pos = int(np.clip(pos, 0, len(e._jie_min) - 2)[0])

    if forward:
        jie_minute = int(e._jie_min[pos + 1])
    else:
        jie_minute = int(e._jie_min[pos])

    days_to_jie = abs(jie_minute - birth_minute) // 1440
    start_age = max(1, round(days_to_jie / 3))

    month_stem_idx = STEMS.index(chart.month.stem)
    month_branch_idx = BRANCHES.index(chart.month.branch)

    dayuns = []
    for i in range(1, count + 1):
        offset = i if forward else -i
        s_idx = (month_stem_idx + offset) % 10
        b_idx = (month_branch_idx + offset) % 12
        main_hidden_idx = BRANCH_HIDDEN_STEMS[b_idx][-1]

        dayuns.append(DaYun(
            start_age=start_age + (i - 1) * 10,
            stem=STEMS[s_idx],
            branch=BRANCHES[b_idx],
            stem_element=STEM_ELEMENT[s_idx],
            branch_element=BRANCH_ELEMENT[b_idx],
            stem_shi_shen=_ss_name(_shi_shen_idx(day_stem_idx, s_idx), lang),
            branch_shi_shen=_ss_name(_shi_shen_idx(day_stem_idx, main_hidden_idx), lang),
        ))
    return dayuns
