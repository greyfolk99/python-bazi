"""python-bazi 기본 동작 테스트."""
from datetime import date, datetime

import pytest

import bazi


# ── 사주 계산 ──────────────────────────────────────────────

def test_chart_standard():
    c = bazi.chart(datetime(1992, 8, 4, 3, 30))
    assert str(c) == "壬申 丁未 壬子 壬寅"


def test_chart_lunar_equals_solar():
    c_solar = bazi.chart(datetime(1992, 8, 4, 3, 30))
    c_lunar = bazi.chart(datetime(1992, 7, 6, 3, 30), time_basis="lunar")
    assert c_solar == c_lunar


def test_chart_solar_requires_longitude():
    with pytest.raises(ValueError):
        bazi.chart(datetime(1992, 8, 4, 3, 30), time_basis="solar")


def test_chart_unknown_time_basis():
    with pytest.raises(ValueError):
        bazi.chart(datetime(1992, 8, 4, 3, 30), time_basis="unknown")


# ── 음력 변환 ──────────────────────────────────────────────

def test_lunar_to_solar():
    assert bazi.lunar_to_solar(1992, 7, 6) == date(1992, 8, 4)


def test_solar_to_lunar():
    assert bazi.solar_to_lunar(date(1992, 8, 4)) == (1992, 7, 6, False)


def test_lunar_leap_month():
    # 2020 윤4월 1일 → 양력 2020-05-23
    assert bazi.lunar_to_solar(2020, 4, 1, is_leap=True) == date(2020, 5, 23)


def test_solar_to_lunar_leap():
    year, month, day, is_leap = bazi.solar_to_lunar(date(2020, 5, 23))
    assert (year, month, is_leap) == (2020, 4, True)


def test_lunar_roundtrip(tmp_path):
    samples = [
        (1900, 1, 1, False),
        (2000, 6, 15, False),
        (2020, 4, 1, True),
        (2099, 6, 15, False),
    ]
    for ly, lm, ld, leap in samples:
        solar = bazi.lunar_to_solar(ly, lm, ld, is_leap=leap)
        back = bazi.solar_to_lunar(solar)
        assert back == (ly, lm, ld, leap), f"roundtrip failed for {ly}-{lm}-{ld} leap={leap}"


# ── 분석 ──────────────────────────────────────────────────

@pytest.fixture
def hjseo_chart():
    return bazi.chart(datetime(1992, 8, 4, 3, 30))


def test_analyze_elements(hjseo_chart):
    r = bazi.analyze(hjseo_chart)
    assert r.elements.counts["水"] == 4
    assert r.elements.dominant == "水"


def test_analyze_day_pillar(hjseo_chart):
    bazi.config.lang = "zh"
    r = bazi.analyze(hjseo_chart)
    assert r.pillars["day"].stem == "壬"
    assert r.pillars["day"].stem_shi_shen == "日元"


def test_analyze_lang_global(hjseo_chart):
    bazi.config.lang = "ko"
    r = bazi.analyze(hjseo_chart)
    assert r.pillars["year"].stem_shi_shen == "비견"
    bazi.config.lang = "zh"  # restore


def test_analyze_lang_override(hjseo_chart):
    bazi.config.lang = "ko"
    r = bazi.analyze(hjseo_chart, lang="zh")
    assert r.pillars["year"].stem_shi_shen == "比肩"
    bazi.config.lang = "zh"  # restore


def test_analyze_dayun(hjseo_chart):
    r = bazi.analyze(hjseo_chart, sex="male", birth=date(1992, 8, 4))
    assert len(r.dayun) == 8
    assert r.dayun[0].start_age == 1
    assert r.dayun[0].stem == "戊"
    assert r.dayun[0].branch == "申"


# ── config 검증 ────────────────────────────────────────────

def test_config_invalid_lang():
    with pytest.raises(ValueError):
        bazi.config.lang = "jp"


def test_config_env(monkeypatch):
    monkeypatch.setenv("BAZI_LANG", "en")
    from importlib import reload
    import bazi._config as cfg
    reload(cfg)
    assert cfg.config.lang == "en"
    cfg.config.lang = "zh"  # restore


# ── 벡터 연산 ─────────────────────────────────────────────

def test_vectorized():
    import numpy as np

    ordinals = np.array([date(1992, 8, 4).toordinal()])
    hours = np.array([3])
    yg, yz, mg, mz, dg, dz, hg, hz = bazi.vectorized(ordinals, hours)
    assert bazi.BaziChart(
        year=bazi.Pillar(bazi._engine.STEMS[yg[0]], bazi._engine.BRANCHES[yz[0]]),
        month=bazi.Pillar(bazi._engine.STEMS[mg[0]], bazi._engine.BRANCHES[mz[0]]),
        day=bazi.Pillar(bazi._engine.STEMS[dg[0]], bazi._engine.BRANCHES[dz[0]]),
        hour=bazi.Pillar(bazi._engine.STEMS[hg[0]], bazi._engine.BRANCHES[hz[0]]),
    ) == bazi.chart(datetime(1992, 8, 4, 3, 30))
