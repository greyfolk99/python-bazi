"""음력↔양력 변환 (lunar_table npz 기반, 외부 라이브러리 불필요).

lunar_python으로 미리 빌드된 달력 테이블(lunar_table_1800_2200.npz)을
이진 탐색으로 조회. UTC/CST 불일치 문제 없음.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).parent / "_data"

_ord_arr: np.ndarray | None = None
_yr_arr: np.ndarray | None = None
_mo_arr: np.ndarray | None = None


def _load() -> None:
    global _ord_arr, _yr_arr, _mo_arr
    if _ord_arr is None:
        data = np.load(_DATA_DIR / "lunar_table_1800_2200.npz")
        _ord_arr = data["ordinals"]  # 각 달 시작 양력 ordinal (int32)
        _yr_arr  = data["years"]     # 음력 연도 (int16)
        _mo_arr  = data["months"]    # 음력 월, 윤달이면 음수 (int8)


def lunar_to_solar(
    lunar_year: int,
    lunar_month: int,
    lunar_day: int,
    is_leap: bool = False,
) -> date:
    """음력 날짜 → 양력 날짜."""
    _load()
    mo_target = -lunar_month if is_leap else lunar_month
    yr_s = _yr_arr.astype(np.int32)
    mo_s = _mo_arr.astype(np.int32)
    idx = np.where((yr_s == lunar_year) & (mo_s == mo_target))[0]
    if len(idx) == 0:
        raise ValueError(f"음력 {lunar_year}년 {'윤' if is_leap else ''}{lunar_month}월 없음")
    return date.fromordinal(int(_ord_arr[idx[0]]) + lunar_day - 1)


def solar_to_lunar(d: date) -> tuple[int, int, int, bool]:
    """양력 날짜 → (음력 연도, 음력 월, 음력 일, 윤달 여부)."""
    _load()
    d_ord = d.toordinal()
    pos = int(np.searchsorted(_ord_arr, d_ord, side="right")) - 1
    if pos < 0 or pos >= len(_ord_arr):
        raise ValueError(f"양력 {d} 는 지원 범위(1800~2200) 밖")
    mo = int(_mo_arr[pos])
    return (
        int(_yr_arr[pos]),
        abs(mo),
        d_ord - int(_ord_arr[pos]) + 1,
        mo < 0,
    )
