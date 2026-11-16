"""로컬 데이터 입출력과 재현 가능한 실행 버전 기록."""
import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + '\n')
    temporary.replace(path)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def configuration_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def load_documents(root, slim=True):
    """Phase2에서 이미 추출한 본문을 읽습니다. 바이너리 재파싱이나 원격 호출은 하지 않습니다.

    Phase3는 표 본문을 primary blocks에서만 청킹하므로 중복된 구조화 cells/rows는
    메모리에서 제외합니다. 표 좌표 메타데이터와 모든 실제 원문 블록은 보존합니다.
    """
    documents = []
    with (Path(root) / 'data/source_documents.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            document = json.loads(line)
            if slim:
                document['tables'] = [{k: v for k, v in table.items() if k not in ['cells', 'rows']}
                                      for table in document.get('tables', [])]
            documents.append(document)
    return documents


def load_chunks(root):
    return read_jsonl(Path(root) / 'data/chunks.jsonl')


def verify_saved_run(root):
    root = Path(root).resolve()
    path = root / 'reports/run_manifest.json'
    if not path.is_file():
        raise FileNotFoundError('검색 실행 기록이 없습니다. 먼저 로컬 인덱스를 구축하세요.')
    saved = json.loads(path.read_text(encoding='utf-8'))
    for group in ['files_sha256', 'code_sha256']:
        for relative, expected in saved.get(group, {}).items():
            target = (root / relative).resolve()
            try:
                target.relative_to(root)
            except ValueError as error:
                raise ValueError('실행 기록의 경로가 프로젝트 폴더를 벗어납니다.') from error
            if not target.is_file() or sha256_file(target) != expected:
                raise ValueError(f'저장 결과와 현재 파일의 버전이 다릅니다: {relative}. 인덱스를 다시 구축하세요.')
    return saved
