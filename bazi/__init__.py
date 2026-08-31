from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from ._engine import BaziChart, Pillar, bazi_vectorized as _bazi_vectorized

__all__ = ["chart", "vectorized", "BaziChart", "Pillar"]
__version__ = "0.1.0"


def chart(
    dt: datetime,
    longitude: float = 126.978,
    timezone: str = "Asia/Seoul",
    time_basis: str = "civil",
) -> BaziChart:
    """생년월일시 → 사주팔자.

    Args:
        dt: 생년월일시 (naive datetime, 현지 시각 기준)
        longitude: 경도 (태양시 보정용, time_basis='solar'일 때 사용)
        timezone: IANA 타임존 (time_basis='solar'일 때 사용)
        time_basis: 'civil' (표준시) | 'solar' (진태양시 보정)
    """
    from ._engine import _engine

    return _engine().chart(dt, longitude=longitude, timezone=timezone, time_basis=time_basis)


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
