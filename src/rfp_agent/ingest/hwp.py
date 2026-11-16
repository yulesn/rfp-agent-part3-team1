
"""[문서 처리] HWP/HWPX 파서

- syhwp를 사용해 HWP/HWPX 문서를 파싱한다.
- 문단, 표, 수식, 이미지 정보를 읽기 순서대로 추출한다.
- 표는 Markdown 형식으로 변환해 본문에 포함한다.
"""

from pathlib import Path

import syhwp


def safe_get(obj, attr, default=None):
    """객체의 속성을 안전하게 가져온다."""
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def parse_hwp(path: str | Path) -> str:
    """HWP/HWPX 파일을 읽어 본문 텍스트를 반환한다."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    doc = syhwp.open(str(path))
    text_parts = []

    for block in getattr(doc, "blocks", []) or []:
        class_name = block.__class__.__name__.lower()
        content = ""

        # 표
        if "table" in class_name:
            try:
                content = block.to_markdown() or ""
            except Exception:
                cells = safe_get(block, "cells", []) or []
                content = "\n".join(
                    str(safe_get(cell, "text", "") or "")
                    for cell in cells
                    if safe_get(cell, "text", "")
                )

        # 수식
        elif "equation" in class_name:
            content = (
                safe_get(block, "latex")
                or safe_get(block, "text")
                or ""
            )

        # 이미지
        elif "image" in class_name or "picture" in class_name:
            content = (
                safe_get(block, "alt_text")
                or safe_get(block, "description")
                or ""
            )
            if content:
                content = f"[그림] {content}"

        # 일반 문단 및 기타 블록
        else:
            content = safe_get(block, "text", "") or ""

        content = str(content).strip()

        if content:
            text_parts.append(content)

    # 블록에서 텍스트를 추출하지 못한 경우 전체 Markdown 사용
    if not text_parts:
        fallback = safe_get(doc, "export_to_markdown", None)

        if callable(fallback):
            fallback = fallback()

        if fallback:
            text_parts.append(str(fallback).strip())

    return "\n\n".join(text_parts).strip()
