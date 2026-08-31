"""음력 달력 테이블 생성 → bazi/_data/lunar_table_1800_2200.npz

각 음력 달의 양력 시작 날짜(ordinal)·음력 연도·월·윤달 여부를 저장.
lunar_python의 getFirstJulianDay() 기반 — 내부 계산과 완벽히 일치.

실행:
    python3.12 scripts/build_lunar_table.py

의존: lunar_python (빌드 타임만)
"""
from datetime import date
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent.parent / "bazi/_data/lunar_table_1800_2200.npz"
_EPOCH_ORD = date(1970, 1, 1).toordinal()
_JD_NOON_1970 = 2440588  # 정수 JD: 1970-01-01 정오 (getFirstJulianDay() 기준)

START_YEAR = 1800
END_YEAR   = 2201


def jd_to_ordinal(jd: int) -> int:
    """정수 JD(정오 기준) → 양력 날짜 ordinal.

    getFirstJulianDay()는 정수 JD를 반환하며, 정수 JD n = n일의 정오.
    _JD_NOON_1970(2440588) = 1970-01-01 정오이므로 정수 연산으로 정확히 변환.
    """
    return jd - _JD_NOON_1970 + _EPOCH_ORD


def build():
    from lunar_python import LunarYear

    ordinals: list[int] = []
    years: list[int] = []
    months: list[int] = []   # 음수 = 윤달 (예: -4 = 윤4월)

    seen_jd: set[int] = set()

    for year in range(START_YEAR, END_YEAR):
        ly = LunarYear.fromYear(year)
        for m in ly.getMonths():
            jd = int(m.getFirstJulianDay())
            if jd in seen_jd:
                continue
            seen_jd.add(jd)

            y = m.getYear()
            mo = m.getMonth()  # 윤달이면 음수 (예: 윤4월 = -4)

            # 연도 범위 필터
            if y < START_YEAR - 1 or y >= END_YEAR:
                continue

            ordinals.append(jd_to_ordinal(jd))
            years.append(y)
            months.append(mo)

    # ordinal 기준 정렬
    idx = sorted(range(len(ordinals)), key=lambda i: ordinals[i])
    ord_arr = np.array([ordinals[i] for i in idx], dtype=np.int32)
    yr_arr  = np.array([years[i] for i in idx], dtype=np.int16)
    mo_arr  = np.array([months[i] for i in idx], dtype=np.int8)

    np.savez_compressed(OUT, ordinals=ord_arr, years=yr_arr, months=mo_arr)
    print(f"저장 완료: {OUT}")
    print(f"  달 수: {len(ord_arr)} ({len(ord_arr)//12:.0f}년치)")
    print(f"  파일 크기: {OUT.stat().st_size/1024:.1f}KB")

    # 검증: 2020 윤4월
    mask = (yr_arr == 2020) & (mo_arr == -4)
    if mask.any():
        idx2020 = np.where(mask)[0][0]
        d = date.fromordinal(int(ord_arr[idx2020]))
        print(f"  2020 윤4월 시작: 양력 {d}")
    else:
        print("  ! 2020 윤4월 없음")


if __name__ == "__main__":
    build()
