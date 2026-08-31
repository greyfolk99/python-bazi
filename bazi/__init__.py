from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from ._config import config
from ._engine import BaziChart, Pillar, bazi_vectorized as _bazi_vectorized
from ._lunar import lunar_to_solar, solar_to_lunar

__all__ = [
    "chart", "vectorized", "analyze",
    "config",
    "lunar_to_solar", "solar_to_lunar",
    "BaziChart", "Pillar",
]
__version__ = "0.1.0"


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
