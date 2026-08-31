"""절기(節) 테이블 생성 → bazi/_data/jieqi_1800_2200_sec.npz

12절(節): 小寒·立春·驚蟄·清明·立夏·芒種·小暑·立秋·白露·寒露·立冬·大雪
사주 월주 계산 기준. 절기 시각(초 단위, Unix epoch 기준) + 월 인덱스 + 태양력 연도 저장.

실행:
    python3.12 scripts/build_jieqi.py

의존: lunar_python (빌드 타임만, pip install python-bazi[build])
"""
from datetime import date
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent.parent / "bazi/_data/jieqi_1800_2200_sec.npz"
_JD_UNIX = 2440587.5   # JD at 1970-01-01 00:00 UTC
_EPOCH_ORD = date(1970, 1, 1).toordinal()

START_YEAR = 1800
END_YEAR   = 2201

# lunar_python getJieQiJulianDays(): 연도별 24절기 JD 배열
# 짝수 인덱스 = 절(節), 홀수 인덱스 = 기(氣/중기)
# 절 12개 순서 (짝수 인덱스 0,2,4,...,22):
#   小寒(0)·立春(2)·驚蟄(4)·清明(6)·立夏(8)·芒種(10)·
#   小暑(12)·立秋(14)·白露(16)·寒露(18)·立冬(20)·大雪(22)
# month_idx 0~11 에 순서대로 매핑


def build():
    from lunar_python import LunarYear

    seen_secs: set[int] = set()
    entries: list[tuple[int, int, int]] = []  # (seconds, month_idx, year)

    for year in range(START_YEAR - 1, END_YEAR):
        ly = LunarYear.fromYear(year)
        jd_list = ly.getJieQiJulianDays()

        for raw_i, jd in enumerate(jd_list):
            if raw_i % 2 != 0:
                continue  # 홀수 = 기(氣/중기), 건너뜀. 짝수 = 절(節) 선택
            month_idx = raw_i // 2  # 0~11: 大雪(0)·小寒(1)·立春(2)·...·立冬(11)
            if month_idx > 11:
                continue

            seconds = round((jd - _JD_UNIX) * 24 * 3600)
            if seconds in seen_secs:
                continue

            d_ord = seconds // 86400 + _EPOCH_ORD
            d = date.fromordinal(d_ord)
            if d.year < START_YEAR - 1 or d.year >= END_YEAR:
                continue

            seen_secs.add(seconds)
            entries.append((seconds, month_idx, year))

    entries.sort(key=lambda x: x[0])

    sec_arr   = np.array([e[0] for e in entries], dtype=np.int64)
    month_arr = np.array([e[1] for e in entries], dtype=np.int8)
    year_arr  = np.array([e[2] for e in entries], dtype=np.int16)

    np.savez_compressed(OUT, sec_arr=sec_arr, month_idx=month_arr, year_arr=year_arr)
    print(f"저장 완료: {OUT}")
    print(f"  절기 개수: {len(sec_arr)}")
    diffs = np.diff(sec_arr)
    print(f"  절기 간격: {diffs.min()/86400:.1f}~{diffs.max()/86400:.1f}일 (평균 {diffs.mean()/86400:.2f}일)")
    print(f"  파일 크기: {OUT.stat().st_size/1024:.1f}KB")


if __name__ == "__main__":
    build()
