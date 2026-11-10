"""[공통] 설정 불러오기

- configs/*.yaml 과 .env 를 읽어서 다른 모듈이 같은 설정을 쓰게 해 주는 파일입니다.
- 다른 파일에서는 `from rfp_agent.config import load_config` 로 불러서 씁니다.
- 설정 항목을 새로 추가하면 configs/default.yaml 에도 기본값을 꼭 넣어 주세요.
"""

from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    """YAML 설정 파일을 dict로 읽고, .env 의 환경 변수(API 키)도 함께 불러옵니다."""
    load_dotenv(PROJECT_ROOT / ".env")
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
