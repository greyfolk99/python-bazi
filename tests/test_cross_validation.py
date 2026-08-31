"""lunar_python 원본과 결과 비교 — cross-validation."""
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

lp = pytest.importorskip("lunar_python", reason="lunar_python not installed")
from lunar_python import Lunar, LunarYear, Solar  # noqa: E402

import bazi  # noqa: E402

_START = date(1800, 1, 1)
_END   = date(2200, 1, 1)  # exclusive
_GT_PATH = Path(__file__).parent.parent / "bazi/_data/ground_truth_1800_2200.npz"


# ── 음력 → 양력 (spot check) ───────────────────────────────

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


# ── 양력 → 음력 (spot check) ───────────────────────────────

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


# ── 사주 계산 (spot check) ─────────────────────────────────

BAZI_CASES = [
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
    lp_solar = Solar.fromYmdHms(dt.year, dt.month, dt.day, dt.hour, dt.minute, 0)
    lp_lunar = lp_solar.getLunar()
    ec = lp_lunar.getEightChar()
    expected = f"{ec.getYear()} {ec.getMonth()} {ec.getDay()} {ec.getTime()}"

    result = str(bazi.chart(dt))
    assert result == expected, f"{dt}: {result} != {expected}"


# ── 전체 범위 cross-validation (ground_truth 기반, fast) ──

@pytest.fixture(scope="module")
def gt():
    if not _GT_PATH.exists():
        pytest.skip("ground_truth_1800_2200.npz 없음. scripts/build_ground_truth.py 실행 필요")
    return np.load(_GT_PATH)


def test_solar_to_lunar_full_range(gt):
    """1800-2200년 전체 날짜 양력→음력 변환 전수 비교 (ground_truth)."""
    ordinals = gt["ordinals"]       # (n_days,) int32
    gt_lunar = gt["lunar"]          # (n_days, 3) int16: [year, month(음수=윤달), day]

    failures = []
    for i, ord_val in enumerate(ordinals):
        d = date.fromordinal(int(ord_val))
        expected = (int(gt_lunar[i, 0]), abs(int(gt_lunar[i, 1])), int(gt_lunar[i, 2]), bool(gt_lunar[i, 1] < 0))
        result = bazi.solar_to_lunar(d)
        if result != expected:
            failures.append(f"{d}: {result} != {expected}")
            if len(failures) >= 20:
                break

    assert not failures, f"{len(failures)} failures:\n" + "\n".join(failures)


def test_lunar_to_solar_full_range(gt):
    """1800-2200년 전체 날짜의 음력 역방향 변환 전수 비교 (ground_truth)."""
    ordinals = gt["ordinals"]
    gt_lunar = gt["lunar"]          # (n_days, 3) int16: [year, month, day]

    failures = []
    for i, ord_val in enumerate(ordinals):
        expected_solar = date.fromordinal(int(ord_val))
        ly   = int(gt_lunar[i, 0])
        lm   = int(gt_lunar[i, 1])
        ld   = int(gt_lunar[i, 2])
        leap = lm < 0
        try:
            result = bazi.lunar_to_solar(ly, abs(lm), ld, is_leap=leap)
        except Exception as e:
            failures.append(f"lunar {ly}-{'윤' if leap else ''}{abs(lm)}-{ld}: exception {e}")
            continue
        if result != expected_solar:
            failures.append(f"lunar {ly}-{'윤' if leap else ''}{abs(lm)}-{ld}: {result} != {expected_solar}")
            if len(failures) >= 20:
                break

    assert not failures, f"{len(failures)} failures:\n" + "\n".join(failures)


def test_bazi_full_range(gt):
    """1800-2200년 전체 날짜 × 6 시간대 사주 전수 비교 (ground_truth, vectorized)."""
    ordinals  = gt["ordinals"].astype(np.int64)   # (n_days,)
    hours     = gt["hours"].astype(np.int64)      # (n_hours,)
    gt_bazi   = gt["bazi"]                        # (n_days, n_hours, 8) int8
    n_days, n_hours = len(ordinals), len(hours)

    ord_rep = np.repeat(ordinals, n_hours)
    hr_tile = np.tile(hours, n_days)
    computed = np.stack(bazi.vectorized(ord_rep, hr_tile), axis=-1)   # (N, 8)
    computed = computed.reshape(n_days, n_hours, 8).astype(np.int8)

    diff = computed != gt_bazi
    if not diff.any():
        return

    col_names = ["year_gan", "year_zhi", "month_gan", "month_zhi",
                 "day_gan",  "day_zhi",  "hour_gan",  "hour_zhi"]
    failures = []
    for di, hi, ci in np.argwhere(diff)[:20]:
        d = date.fromordinal(int(ordinals[di]))
        h = int(hours[hi])
        failures.append(
            f"{col_names[ci]}: {d} {h:02d}h → got {computed[di,hi,ci]} expected {gt_bazi[di,hi,ci]}"
        )
    total = int(diff.any(axis=2).sum())
    assert False, f"{total} mismatch days×hours (first 20 cells):\n" + "\n".join(failures)
