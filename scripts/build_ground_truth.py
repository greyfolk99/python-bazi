"""ground_truth 생성 → bazi/_data/ground_truth_1800_2200.npz

1800-2200년 전체 날짜 × 6 시간대 사주 + 전체 날짜 음력 변환 결과를
lunar_python 원본으로 계산해서 저장.

저장 구조:
  ordinals : (n_days,)         int32  — 날짜 ordinal
  hours    : (n_hours,)        int8   — [0..23]
  bazi     : (n_days, n_hours, 8) int8  — [yg, yz, mg, mz, dg, dz, hg, hz]
  lunar    : (n_days, 3)       int16  — [year, month(음수=윤달), day]

실행:
    python3.12 scripts/build_ground_truth.py
의존: lunar_python (빌드 타임만)
"""
from datetime import date, timedelta
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

OUT   = Path(__file__).parent.parent / "bazi/_data/ground_truth_1800_2200.npz"
HOURS = list(range(24))
START = date(1800, 1, 1)
END   = date(2200, 1, 1)  # exclusive

_STEMS    = "甲乙丙丁戊己庚辛壬癸"
_BRANCHES = "子丑寅卯辰巳午未申酉戌亥"


def _process_chunk(args: tuple[list[date], list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """프로세스별 날짜 청크 처리. (bazi_chunk, lunar_chunk) 반환."""
    from lunar_python import Solar

    stem_idx   = {s: i for i, s in enumerate(_STEMS)}
    branch_idx = {b: i for i, b in enumerate(_BRANCHES)}

    chunk_days, hours = args
    n = len(chunk_days)
    n_h = len(hours)

    bazi_chunk  = np.zeros((n, n_h, 8), dtype=np.int8)
    lunar_chunk = np.zeros((n, 3), dtype=np.int16)

    for i, d in enumerate(chunk_days):
        lp_l = Solar.fromYmd(d.year, d.month, d.day).getLunar()
        lunar_chunk[i, 0] = lp_l.getYear()
        lunar_chunk[i, 1] = lp_l.getMonth()
        lunar_chunk[i, 2] = lp_l.getDay()

        for j, h in enumerate(hours):
            ec = Solar.fromYmdHms(d.year, d.month, d.day, h, 0, 0).getLunar().getEightChar()
            yr = ec.getYear(); mo = ec.getMonth()
            dy = ec.getDay();  ti = ec.getTime()
            bazi_chunk[i, j, 0] = stem_idx[yr[0]];   bazi_chunk[i, j, 1] = branch_idx[yr[1]]
            bazi_chunk[i, j, 2] = stem_idx[mo[0]];   bazi_chunk[i, j, 3] = branch_idx[mo[1]]
            bazi_chunk[i, j, 4] = stem_idx[dy[0]];   bazi_chunk[i, j, 5] = branch_idx[dy[1]]
            bazi_chunk[i, j, 6] = stem_idx[ti[0]];   bazi_chunk[i, j, 7] = branch_idx[ti[1]]

    return bazi_chunk, lunar_chunk


def build():
    days = []
    cur = START
    while cur < END:
        days.append(cur)
        cur += timedelta(days=1)

    n_days  = len(days)
    n_hours = len(HOURS)
    n_cores = cpu_count()
    print(f"날짜 수: {n_days:,}, 시간 수: {n_hours}, 코어 수: {n_cores}")
    print(f"총 사주 케이스: {n_days * n_hours:,}")

    # 날짜를 코어 수만큼 청크로 분할
    chunks = [days[i::n_cores] for i in range(n_cores)]
    args   = [(chunk, HOURS) for chunk in chunks]

    bazi_full  = np.zeros((n_days, n_hours, 8), dtype=np.int8)
    lunar_full = np.zeros((n_days, 3), dtype=np.int16)

    with Pool(n_cores) as pool:
        results = pool.map(_process_chunk, args)

    # 청크를 원래 순서로 재조합 (i::n_cores 분배의 역순)
    for core_i, (bazi_chunk, lunar_chunk) in enumerate(results):
        indices = list(range(core_i, n_days, n_cores))
        bazi_full[indices]  = bazi_chunk
        lunar_full[indices] = lunar_chunk

    ordinals = np.array([d.toordinal() for d in days], dtype=np.int32)

    np.savez_compressed(
        OUT,
        ordinals=ordinals,
        hours=np.array(HOURS, dtype=np.int8),
        bazi=bazi_full,
        lunar=lunar_full,
    )
    print(f"\n저장 완료: {OUT}")
    print(f"  파일 크기: {OUT.stat().st_size / 1024 / 1024:.1f}MB")


if __name__ == "__main__":
    build()
