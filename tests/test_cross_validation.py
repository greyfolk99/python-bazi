"""lunar_python 원본과 결과 비교 — cross-validation."""
from datetime import date, datetime

import pytest

lp = pytest.importorskip("lunar_python", reason="lunar_python not installed")
from lunar_python import Lunar, Solar  # noqa: E402

import bazi  # noqa: E402


# ── 음력 → 양력 ────────────────────────────────────────────

LUNAR_TO_SOLAR_CASES = [
    # (lunar_year, lunar_month, lunar_day, is_leap)
    (1900, 1, 1, False),
    (1920, 5, 15, False),
    (1950, 12, 1, False),
    (1970, 1, 1, False),
    (1992, 7, 6, False),
    (2000, 1, 1, False),
    (2004, 2, 29, False),
    (2020, 3, 1, False),
    (2020, 4, 1, True),   # 윤4월
    (2020, 4, 15, True),  # 윤4월 중순
    (2023, 2, 1, True),   # 윤2월
    (2050, 6, 1, False),
    (2099, 6, 15, False),
]


@pytest.mark.parametrize("ly,lm,ld,leap", LUNAR_TO_SOLAR_CASES)
def test_lunar_to_solar_vs_lunar_python(ly, lm, ld, leap):
    lp_month = -lm if leap else lm
    lp_lunar = Lunar.fromYmd(ly, lp_month, ld)
    lp_solar = lp_lunar.getSolar()
    expected = date(lp_solar.getYear(), lp_solar.getMonth(), lp_solar.getDay())

    result = bazi.lunar_to_solar(ly, lm, ld, is_leap=leap)
    assert result == expected, f"lunar {ly}-{'윤' if leap else ''}{lm}-{ld}: {result} != {expected}"


# ── 양력 → 음력 ────────────────────────────────────────────

SOLAR_TO_LUNAR_CASES = [
    date(1900, 2, 1),
    date(1920, 6, 21),
    date(1950, 1, 17),
    date(1970, 1, 27),
    date(1992, 8, 4),
    date(2000, 2, 5),
    date(2020, 4, 23),   # 음력 4월
    date(2020, 5, 23),   # 윤4월
    date(2023, 3, 22),   # 윤2월
    date(2050, 7, 4),
    date(2099, 8, 1),
]


@pytest.mark.parametrize("solar", SOLAR_TO_LUNAR_CASES)
def test_solar_to_lunar_vs_lunar_python(solar):
    lp_solar = Solar.fromYmd(solar.year, solar.month, solar.day)
    lp_lunar = lp_solar.getLunar()
    lp_m = lp_lunar.getMonth()
    expected = (lp_lunar.getYear(), abs(lp_m), lp_lunar.getDay(), lp_m < 0)

    result = bazi.solar_to_lunar(solar)
    assert result == expected, f"solar {solar}: {result} != {expected}"


# ── 사주 계산 ──────────────────────────────────────────────

BAZI_CASES = [
    # (datetime, expected_str)
    datetime(1900, 1, 31, 0, 0),
    datetime(1992, 8, 4, 3, 30),
    datetime(2000, 1, 1, 0, 0),
    datetime(2024, 2, 10, 12, 0),
    datetime(1970, 6, 15, 18, 0),
    datetime(1850, 3, 20, 6, 0),
    datetime(2100, 11, 5, 22, 0),
]


@pytest.mark.parametrize("dt", BAZI_CASES)
def test_bazi_chart_vs_lunar_python(dt):
    # lunar_python: Solar → Lunar → EightChar
    # lunar_python은 음력 날짜 기반 Lunar.fromYmdHms로 사주를 구하므로
    # Solar.fromYmdHms 로 직접 접근
    lp_solar = Solar.fromYmdHms(dt.year, dt.month, dt.day, dt.hour, dt.minute, 0)
    lp_lunar = lp_solar.getLunar()
    ec = lp_lunar.getEightChar()
    expected = f"{ec.getYear()} {ec.getMonth()} {ec.getDay()} {ec.getTime()}"

    result = str(bazi.chart(dt))
    assert result == expected, f"{dt}: {result} != {expected}"
