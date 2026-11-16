
"""[문서 처리] 메타데이터 불러오기

- CSV 파일을 읽어 파일명 → 메타데이터 dict 형태로 변환한다.
- 공고 번호는 문자열로 유지한다.
- 공고 차수와 사업 금액은 숫자형으로 변환한다.
- 날짜는 YYYY-MM-DD 형식으로 정규화한다.
- 빈 값과 원본 참고 텍스트는 메타데이터에서 제외한다.
"""

from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd


METADATA_COLUMNS = [
    "공고 번호",
    "공고 차수",
    "사업명",
    "사업 금액",
    "발주 기관",
    "공개 일자",
    "입찰 참여 시작일",
    "입찰 참여 마감일",
    "사업 요약",
    "파일형식",
]

REQUIRED_COLUMNS = [
    *METADATA_COLUMNS,
    "파일명",
    "텍스트",
]


def is_missing(value) -> bool:
    """None, NaN 등 결측값 여부를 확인한다."""
    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalize_number(value, integer: bool = False):
    """숫자 문자열을 숫자형으로 변환한다."""
    if is_missing(value) or str(value).strip() == "":
        return None

    try:
        number = Decimal(str(value).replace(",", "").strip())

        if integer and number == number.to_integral_value():
            return int(number)

        return float(number)

    except (InvalidOperation, ValueError):
        return None


def normalize_datetime(value):
    """날짜를 YYYY-MM-DD 또는 YYYY-MM-DD HH:MM:SS로 변환한다."""
    if is_missing(value) or str(value).strip() == "":
        return None

    raw = str(value).strip()
    parsed = pd.to_datetime(raw, errors="coerce")

    if pd.isna(parsed):
        return raw

    if parsed.hour == 0 and parsed.minute == 0 and parsed.second == 0:
        if len(raw) <= 10:
            return parsed.strftime("%Y-%m-%d")

    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def build_metadata(row) -> dict:
    """CSV 한 행에서 메타데이터를 생성한다."""
    metadata = {}

    for key in METADATA_COLUMNS:
        value = row.get(key)

        if is_missing(value) or str(value).strip() == "":
            continue

        if key == "공고 번호":
            value = str(value).strip()

        elif key in ("공고 차수", "사업 금액"):
            value = normalize_number(value, integer=True)

        elif key in (
            "공개 일자",
            "입찰 참여 시작일",
            "입찰 참여 마감일",
        ):
            value = normalize_datetime(value)

        elif key == "파일형식":
            value = str(value).strip().lower().replace(".", "")

        else:
            value = str(value).strip()

        if value is not None:
            metadata[key] = value

    return metadata


def load_metadata(path: str | Path) -> dict[str, dict]:
    """CSV를 읽어 파일명별 메타데이터 딕셔너리를 반환한다."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"메타데이터 파일을 찾을 수 없습니다: {path}")

    df = pd.read_csv(
        path,
        dtype={"공고 번호": "string"},
        encoding="utf-8-sig",
    )

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"CSV 필수 컬럼이 없습니다: {missing_columns}"
        )

    metadata_map = {}

    for _, row in df.iterrows():
        filename = row["파일명"]

        if is_missing(filename) or str(filename).strip() == "":
            continue

        filename = str(filename).strip()

        if filename in metadata_map:
            raise ValueError(
                f"중복된 파일명이 CSV에 존재합니다: {filename}"
            )

        metadata_map[filename] = build_metadata(row)

    return metadata_map
