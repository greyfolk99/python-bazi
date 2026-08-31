from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import cos, pi, sin
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

STEMS = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
BRANCHES = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]

_DATA_DIR = Path(__file__).parent / "_data"
_EPOCH_ORD = date(1970, 1, 1).toordinal()
_BASE_ORD = date(1900, 1, 31).toordinal()  # 甲辰日
_BASE_GAN = 0
_BASE_ZHI = 4

_YIN_MONTH_GAN = np.array([2, 4, 6, 8, 0, 2, 4, 6, 8, 0], dtype=np.int8)
# mi=0=大雪(子月)·1=小寒(丑月)·2=立春(寅月)·...·11=立冬(亥月)
_JIE_TO_MONTH_ZHI    = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int8)
# 寅月(mi=2) 기준 오프셋: 子=-2≡8, 丑=-1≡9, 寅=0, 卯=1, ...
_JIE_TO_MONTH_OFFSET = np.array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int8)
_DAY_GAN_TO_HOUR_BASE = np.array([0, 2, 4, 6, 8, 0, 2, 4, 6, 8], dtype=np.int8)


@dataclass(frozen=True)
class Pillar:
    stem: str
    branch: str

    @property
    def text(self) -> str:
        return f"{self.stem}{self.branch}"

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class BaziChart:
    year: Pillar
    month: Pillar
    day: Pillar
    hour: Pillar | None = None

    @property
    def pillars(self) -> tuple[Pillar, ...]:
        if self.hour is None:
            return (self.year, self.month, self.day)
        return (self.year, self.month, self.day, self.hour)

    def __str__(self) -> str:
        if self.hour is None:
            return f"{self.year} {self.month} {self.day}"
        return f"{self.year} {self.month} {self.day} {self.hour}"


class _BaziEngine:
    def __init__(self) -> None:
        data = np.load(_DATA_DIR / "jieqi_1800_2200_min.npz")
        self._jie_min: np.ndarray = data["min_arr"]
        self._jie_month: np.ndarray = data["month_idx"]
        self._jie_year: np.ndarray = data["year_arr"]

    def bazi_vectorized(
        self, dates_ord: np.ndarray, hours: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        diff = dates_ord - _BASE_ORD
        day_gan = (_BASE_GAN + diff) % 10
        day_zhi = (_BASE_ZHI + diff) % 12

        next_day = hours >= 23
        hour_zhi = np.where(next_day, 0, ((hours + 1) // 2) % 12)
        day_gan_for_hour = np.where(next_day, (day_gan + 1) % 10, day_gan)
        hour_gan = (_DAY_GAN_TO_HOUR_BASE[day_gan_for_hour].astype(np.int64) + hour_zhi) % 10

        dt_min = (dates_ord - _EPOCH_ORD) * 1440 + hours * 60
        pos = np.searchsorted(self._jie_min, dt_min, side="left") - 1
        pos = np.clip(pos, 0, len(self._jie_min) - 1)

        month_seq = self._jie_month[pos].astype(np.int64)
        jy = self._jie_year[pos].astype(np.int64)
        yg_base = (jy - 4) % 10
        # 年干支: 입춘(mi=2) 기준 — 大雪(mi=0)·小寒(mi=1)은 전년도 연주
        year_yg = np.where(month_seq < 2, (yg_base - 1) % 10, yg_base)
        # 月干: 절기 연도(jy) 그대로 (소한 이후부터 새해 오호둔 적용)
        month_gan = (_YIN_MONTH_GAN[yg_base].astype(np.int64) + _JIE_TO_MONTH_OFFSET[month_seq]) % 10
        month_zhi = _JIE_TO_MONTH_ZHI[month_seq].astype(np.int64)

        year_zhi_base = (jy - 4) % 12
        year_zhi = np.where(month_seq < 2, (year_zhi_base - 1) % 12, year_zhi_base)

        return year_yg, year_zhi, month_gan, month_zhi, day_gan, day_zhi, hour_gan, hour_zhi

    def chart(
        self,
        dt: datetime | date,
        time_basis: str | None = None,
        longitude: float | None = None,
        timezone: str | None = None,
    ) -> BaziChart:
        date_only = not isinstance(dt, datetime)

        if time_basis == "solar":
            if date_only:
                raise ValueError("time_basis='solar' requires datetime, not date")
            if longitude is None:
                raise ValueError("longitude required for time_basis='solar'")
            if timezone is None:
                raise ValueError("timezone required for time_basis='solar'")
            chart_dt = _true_solar_time(dt, longitude, timezone)
        elif time_basis == "lunar":
            from ._lunar import lunar_to_solar
            solar_date = lunar_to_solar(dt.year, dt.month, dt.day)
            if date_only:
                chart_date = solar_date
                chart_dt = None
            else:
                chart_dt = dt.replace(year=solar_date.year, month=solar_date.month, day=solar_date.day)
                chart_date = None
        elif time_basis is not None:
            raise ValueError(f"unknown time_basis: {time_basis!r}. Use None, 'solar', or 'lunar'.")
        else:
            chart_dt = None if date_only else dt
            chart_date = dt if date_only else None

        if chart_dt is not None:
            actual_date = date(chart_dt.year, chart_dt.month, chart_dt.day)
        else:
            actual_date = chart_date  # type: ignore[assignment]

        d_ord = np.array([actual_date.toordinal()], dtype=np.int64)

        if date_only or chart_dt is None:
            yg, yz, mg, mz, dg, dz, _, _ = self.bazi_vectorized(d_ord, np.zeros(1, dtype=np.int64))
            return BaziChart(
                year=Pillar(STEMS[yg[0]], BRANCHES[yz[0]]),
                month=Pillar(STEMS[mg[0]], BRANCHES[mz[0]]),
                day=Pillar(STEMS[dg[0]], BRANCHES[dz[0]]),
                hour=None,
            )

        h_arr = np.array([chart_dt.hour], dtype=np.int64)
        yg, yz, mg, mz, dg, dz, hg, hz = self.bazi_vectorized(d_ord, h_arr)
        return BaziChart(
            year=Pillar(STEMS[yg[0]], BRANCHES[yz[0]]),
            month=Pillar(STEMS[mg[0]], BRANCHES[mz[0]]),
            day=Pillar(STEMS[dg[0]], BRANCHES[dz[0]]),
            hour=Pillar(STEMS[hg[0]], BRANCHES[hz[0]]),
        )


def bazi_vectorized(dates_ordinal: np.ndarray, hours: np.ndarray) -> tuple[np.ndarray, ...]:
    return _engine().bazi_vectorized(dates_ordinal, hours)



def _true_solar_time(dt: datetime, longitude: float, timezone: str) -> datetime:
    tz = ZoneInfo(timezone)
    aware = dt.replace(tzinfo=tz)
    offset_hours = aware.utcoffset().total_seconds() / 3600.0
    standard_meridian = 15.0 * offset_hours
    n = dt.timetuple().tm_yday
    b = 2.0 * pi * (n - 81) / 364.0
    eot = 9.87 * sin(2.0 * b) - 7.53 * cos(b) - 1.5 * sin(b)
    total_minutes = 4.0 * (longitude - standard_meridian) + eot
    return dt + timedelta(minutes=total_minutes)


_cached_engine: _BaziEngine | None = None


def _engine() -> _BaziEngine:
    global _cached_engine
    if _cached_engine is None:
        _cached_engine = _BaziEngine()
    return _cached_engine
