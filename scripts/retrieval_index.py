"""청크 JSONL 검증 → Chroma 생성 → 저장 상태 확인. 팀 파서와 독립 실행할 수 있습니다."""

import argparse
import importlib
import json
import os
import platform
from pathlib import Path

from rfp_agent.config import PROJECT_ROOT, load_config
from rfp_agent.retrieval.index_io import describe_chunks, read_chunks


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def inspect_index(config: dict) -> dict:
    """API를 호출하거나 없는 DB/컬렉션을 만들지 않고 저장 상태만 조회합니다."""
    path = project_path(config["paths"]["vectorstore_dir"])
    if not (path / "chroma.sqlite3").is_file():
        return {"state": "missing", "chunks": 0, "api_called": False}
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(str(path.resolve()), Settings(anonymized_telemetry=False))
    collection = client.get_collection(
        config["vectorstore"]["collection_name"], embedding_function=None
    )
    metadata = collection.metadata or {}
    return {
        "state": metadata.get("rfp_index_state", "unverified"),
        "chunks": collection.count(),
        "embedding_provider": metadata.get("rfp_embedding_provider"),
        "embedding_model": metadata.get("rfp_embedding_model"),
        "api_called": False,
    }


def doctor() -> dict:
    """키 값은 출력하지 않고 로컬 의존성과 설정 여부만 확인합니다."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    result = {
        "python": platform.python_version(),
        "api_key_configured": bool(key and key != "sk-..."),
    }
    imports = {}
    for name in ("chromadb", "langchain_chroma", "langchain_openai"):
        try:
            importlib.import_module(name)
            imports[name] = "ok"
        except (ImportError, OSError) as exc:
            imports[name] = f"{type(exc).__name__}: {exc}"
    result["imports"] = imports
    result["api_called"] = False
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/retrieval_sample.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "build"):
        sub = commands.add_parser(command)
        sub.add_argument("--chunks", required=True, help="전달받은 page_content/metadata JSONL")
        sub.add_argument("--report", help="키를 제외한 검사 결과 JSON 저장 위치")
    commands.add_parser("inspect")
    commands.add_parser("doctor")
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        if args.command == "doctor":
            result = doctor()
        elif args.command == "inspect":
            result = inspect_index(config)
        else:
            path = project_path(args.chunks)
            rows = read_chunks(path)
            result = describe_chunks(path, rows)
            if args.command == "build":
                # 키 부족은 모델/DB 초기화 전에 판정합니다.
                if config["embedding"]["provider"] == "openai":
                    key = os.environ.get("OPENAI_API_KEY", "").strip()
                    if not key or key == "sk-...":
                        raise ValueError("OPENAI_API_KEY를 .env에 입력한 뒤 다시 실행하세요.")
                from langchain_core.documents import Document

                from rfp_agent.retrieval.vectorstore import build_vectorstore

                build_vectorstore([Document(**row) for row in rows], config)
                # 모델명과 입력 해시만 기록하며 config 전체나 .env를 복사하지 않습니다.
                result.update(
                    state="ready",
                    api_called=True,
                    embedding={
                        "provider": config["embedding"]["provider"],
                        "model": config["embedding"]["model"],
                    },
                )
            if args.report:
                report = project_path(args.report)
                report.parent.mkdir(parents=True, exist_ok=True)
                report.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"실행 실패: {exc}\n")
    except Exception as exc:  # noqa: BLE001 -- CLI 경계에서 외부 오류 본문을 숨깁니다.
        # 외부 API 오류 응답에 키 일부가 포함될 수 있어 본문을 출력하지 않습니다.
        parser.exit(1, f"실행 실패: {type(exc).__name__}, HTTP {getattr(exc, 'status_code', None)}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
