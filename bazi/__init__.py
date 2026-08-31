from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from ._config import config
from ._constants import (
    STEMS, BRANCHES, ELEMENTS,
    STEM_INDEX, BRANCH_INDEX, ELEMENT_INDEX,
    STEM_ELEMENTS, BRANCH_ELEMENTS,
    HIDDEN_STEMS,
    STEM_COMBINATIONS, BRANCH_LIUHE, BRANCH_CLASH, BRANCH_HARM,
    GENERATES, CONTROLS, CONTROLLED_BY,
)
from ._engine import BaziChart, Pillar, bazi_vectorized as _bazi_vectorized
from ._lunar import lunar_to_solar, solar_to_lunar

__all__ = [
    "chart", "vectorized", "catalog", "analyze",
    "config",
    "lunar_to_solar", "solar_to_lunar",
    "BaziChart", "Pillar",
    "STEMS", "BRANCHES", "ELEMENTS",
    "STEM_INDEX", "BRANCH_INDEX", "ELEMENT_INDEX",
    "STEM_ELEMENTS", "BRANCH_ELEMENTS",
    "HIDDEN_STEMS",
    "STEM_COMBINATIONS", "BRANCH_LIUHE", "BRANCH_CLASH", "BRANCH_HARM",
    "GENERATES", "CONTROLS", "CONTROLLED_BY",
]
__version__ = "0.2.0"


def chart(
    dt: datetime | date,
    time_basis: str | None = None,
    longitude: float | None = None,
    timezone: str | None = None,
) -> BaziChart:
    """생년월일시 → 사주팔자.

    Args:
        dt: 생년월일시. datetime이면 시주 포함, date이면 시주 None(삼주).
        time_basis: None (표준시) | 'solar' (진태양시) | 'lunar' (음력 입력)
        longitude: 출생지 경도 (time_basis='solar' 시 필수)
        timezone: IANA 타임존 (time_basis='solar' 시 필수)
    """
    from ._engine import _engine

    return _engine().chart(dt, time_basis=time_basis, longitude=longitude, timezone=timezone)


def vectorized(dates_ordinal, hours):
    """날짜(ordinal) 배열 + 시간 배열 → 팔자 인덱스 8-tuple.

    Returns:
        (year_gan, year_zhi, month_gan, month_zhi, day_gan, day_zhi, hour_gan, hour_zhi)
        각 원소는 numpy int 배열. STEMS/BRANCHES로 문자 변환.
    """
    import numpy as np

    from ._engine import _engine

    e = _engine()
    return e.bazi_vectorized(
        np.asarray(dates_ordinal, dtype=np.int64),
        np.asarray(hours, dtype=np.int64),
    )


def catalog(
    year_start: int,
    year_end: int,
    hours: tuple = (23, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21),
) -> dict:
    """연도 범위 + 시간 목록 → 사주 행렬 딕셔너리.

    ground_truth 데이터에서 해당 연도 범위를 쿼리한다.

    Args:
        year_start: 시작 연도 (포함)
        year_end:   끝 연도 (포함)
        hours:      포함할 시간 목록 (기본: 12시주 대표시간)

    Returns:
        dict with numpy arrays:
            years:      [N] int16
            months:     [N] int8
            days:       [N] int8
            hours:      [N] int8
            slot_index: [N] int8  (hours 인자 내 인덱스)
            stems:      [N, 4] int8  (년간·월간·일간·시간)
            branches:   [N, 4] int8  (년지·월지·일지·시지)
    """
    import numpy as np
    from datetime import date
    from pathlib import Path

    data = np.load(Path(__file__).parent / "_data" / "ground_truth_1800_2200.npz", allow_pickle=False)
    ordinals: np.ndarray = data["ordinals"]  # (N_days,) int32
    bazi_raw: np.ndarray = data["bazi"]      # (N_days, 24, 8) int8

    start_ord = date(year_start, 1, 1).toordinal()
    end_ord = date(year_end + 1, 1, 1).toordinal()
    mask = (ordinals >= start_ord) & (ordinals < end_ord)

    sel_ord = ordinals[mask]
    sel_bazi = bazi_raw[mask][:, list(hours), :]  # (M, S, 8)

    M, S = len(sel_ord), len(hours)

    epoch_ord = date(1970, 1, 1).toordinal()
    dt64 = (sel_ord.astype(np.int64) - epoch_ord).astype("datetime64[D]")
    year_arr  = (dt64.astype("datetime64[Y]").astype(np.int32) + 1970).astype(np.int16)
    month_arr = ((dt64 - dt64.astype("datetime64[Y]")).astype("datetime64[M]").astype(np.int32) + 1).astype(np.int8)
    day_arr   = ((dt64 - dt64.astype("datetime64[M]")).astype("datetime64[D]").astype(np.int32) + 1).astype(np.int8)

    bazi_flat = sel_bazi.reshape(-1, 8)
    return {
        "years":      np.repeat(year_arr,  S),
        "months":     np.repeat(month_arr, S),
        "days":       np.repeat(day_arr,   S),
        "hours":      np.tile(np.array(hours, dtype=np.int8), M),
        "slot_index": np.tile(np.arange(S, dtype=np.int8), M),
        "stems":      bazi_flat[:, [0, 2, 4, 6]].astype(np.int8),
        "branches":   bazi_flat[:, [1, 3, 5, 7]].astype(np.int8),
    }


def analyze(chart: BaziChart, **kwargs):
    """팔자 분석 → ChartAnalysis.

    출력 언어: bazi.config.lang 또는 BAZI_LANG 환경변수 (기본 'zh')

    Args:
        chart: BaziChart
        lang: 'zh'|'ko'|'en'. None이면 bazi.config.lang 사용.
        sex: 'male' | 'female' (대운 계산용)
        birth: datetime.date (대운 계산 시 필요)
        dayun_count: 대운 개수 (기본 8)
    """
    from .analysis import analyze as _analyze

    return _analyze(chart, **kwargs)
