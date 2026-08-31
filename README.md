# python-bazi

사주팔자(四柱八字) 엔진 — 순수 numpy, 외부 의존성 없음.

- 날짜: 1800 ~ 2200년
- 언어: 한자(zh) · 한국어(ko) · 영어(en)
- 음력 변환: 윤달 포함

## 왜 만들었나

기존 파이썬 사주 라이브러리들은 대부분 두 가지 문제가 있었다:

1. **음력을 사주력으로 착각** — 사주는 음력이 아니라 절기력(節氣曆) 기반이다. 달의 위상이 아니라 태양의 황경으로 월주를 정한다.
2. **외부 의존성 과다** — 런타임에 천문 계산 라이브러리를 요구하는 경우가 많다.

python-bazi는 미리 계산해둔 절기 데이터(npz)를 numpy 이진 탐색으로 조회해서 빠르고 정확하게 팔자를 계산한다. 런타임 의존성은 `numpy` 하나뿐이다.

## 설치

```bash
pip install python-bazi
```

데이터 파일을 직접 재생성하려면 빌드 의존성 포함:

```bash
pip install python-bazi[build]
python scripts/build_jieqi.py
python scripts/build_lunar_table.py
```

## 기본 사용법

```python
import bazi
from datetime import datetime, date

# 표준시 기준 사주
chart = bazi.chart(datetime(1992, 8, 4, 3, 30))
print(chart)  # 壬申 丁未 壬子 壬寅

# 음력 입력
chart = bazi.chart(datetime(1992, 7, 6, 3, 30), time_basis='lunar')

# 진태양시 보정 (출생지 경도 필요)
chart = bazi.chart(
    datetime(1992, 8, 4, 3, 30),
    time_basis='solar',
    longitude=127.0,
    timezone='Asia/Seoul',
)
```

## 분석

```python
result = bazi.analyze(chart, sex='male', birth=date(1992, 8, 4))

# 오행 분포
print(result.elements.counts)   # {'木': 1, '火': 1, '土': 1, '金': 1, '水': 4}

# 각 기둥 상세
for key, p in result.pillars.items():
    print(f"{key}: {p.stem}{p.branch} ({p.stem_shi_shen}/{p.branch_shi_shen})")

# 대운
for dy in result.dayun:
    print(f"{dy.start_age}세: {dy.stem}{dy.branch} ({dy.stem_shi_shen})")
```

## 언어 설정

`bazi.config.lang`으로 전역 설정한다. 기본값은 한자(`zh`).

```python
bazi.config.lang = 'ko'   # 한국어: 비견, 겁재, 식신 ...
bazi.config.lang = 'zh'   # 한자:   比肩, 劫財, 食神 ...
bazi.config.lang = 'en'   # 영어:   Friend, Rob Wealth, Food God ...
```

환경변수로도 설정 가능:

```bash
BAZI_LANG=ko python my_script.py
```

개별 호출에서 override:

```python
result = bazi.analyze(chart, lang='en')
```

## 음력 변환

```python
import bazi
from datetime import date

# 음력 → 양력
solar = bazi.lunar_to_solar(1992, 7, 6)           # date(1992, 8, 4)
solar = bazi.lunar_to_solar(2020, 4, 1, is_leap=True)  # 윤4월

# 양력 → 음력
year, month, day, is_leap = bazi.solar_to_lunar(date(1992, 8, 4))
```

## 벡터 연산

날짜 배열을 한번에 처리한다:

```python
import numpy as np

ordinals = np.array([date(1990, 1, 1).toordinal(), date(2000, 6, 15).toordinal()])
hours    = np.array([6, 14])

yg, yz, mg, mz, dg, dz, hg, hz = bazi.vectorized(ordinals, hours)
```

