"""Read-only Phase 3 bootstrap with explicit notebook environment diagnostics."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys


class EnvironmentMismatch(RuntimeError):
    pass


def environment_report(root):
    """No scientific-library import, mutation, install, or network request."""
    root = Path(root).resolve()
    meta_path = root / 'index/index_meta.json'
    if not meta_path.is_file():
        raise FileNotFoundError('동봉 인덱스가 없습니다. 전체 Phase 4 ZIP을 먼저 압축 해제하세요.')
    stored = json.loads(meta_path.read_text(encoding='utf-8')).get('dependencies', {})
    current = {}
    for name in ['numpy', 'scipy', 'scikit-learn']:
        try:
            current[name] = version(name)
        except PackageNotFoundError:
            current[name] = None
    mismatches = {name: {'expected': expected, 'current': current.get(name)}
                  for name, expected in stored.items() if current.get(name) != expected}
    return {'python_executable': sys.executable, 'python_version': sys.version.split()[0],
            'project_root': str(root), 'expected_dependencies': stored,
            'current_dependencies': current, 'mismatches': mismatches,
            'compatible': not mismatches,
            'kernel_hint': 'VS Code 노트북 오른쪽 위 → Select Another Kernel → Python Environments → 이 프로젝트의 .venv/bin/python',
            'install_hint': 'team_handoff 폴더에서 uv sync 후 .venv 커널을 선택하세요.',
            'automatic_install_or_rebuild': False}


def bootstrap(root):
    """Open the preserved index and source snapshot, without changing baselines."""
    root = Path(root).resolve()
    environment = environment_report(root)
    if not environment['compatible']:
        detail = '; '.join(f'{name}: 저장={item["expected"]}, 현재={item["current"]}'
                           for name, item in environment['mismatches'].items())
        raise EnvironmentMismatch(
            f'노트북 Python: {sys.executable}\n{detail}\n'
            + environment['kernel_hint'] + '\n'
            + environment['install_hint'] + '\n커널을 재시작하고 처음부터 실행하세요. 설정의 OpenAI 활성화 값과는 무관합니다.')
    from rfp_rag.retrieval.io import load_chunks, load_documents, verify_saved_run
    from rfp_rag.retrieval.retrieval import LocalSearchIndex
    from .evidence import EvidenceRegistry
    source_manifest = verify_saved_run(root)
    documents = load_documents(root)
    chunks = load_chunks(root)
    index = LocalSearchIndex(root / 'index')
    metadata = index.get_metadata()
    if metadata['chunk_count'] != len(chunks):
        index.close()
        raise ValueError('Phase 3 청크와 인덱스 개수가 다릅니다.')
    registry = EvidenceRegistry(documents, chunks)
    config = json.loads((root / 'configs/phase4.json').read_text(encoding='utf-8'))
    return {'root': root, 'index': index, 'documents': documents, 'chunks': chunks,
            'registry': registry, 'config': config, 'environment': environment,
            'source_manifest': source_manifest}


def source_allowlist(index, doc_ids):
    """Caller-selected public documents only; unknown IDs fail in the index."""
    if not doc_ids:
        raise ValueError('API로 전송할 공개 문서 ID를 먼저 선택하세요.')
    canonical = index.resolve_doc_ids(doc_ids)
    catalog = {row['doc_id']: row for row in index.get_catalog()}
    return {doc_id: catalog[doc_id]['sha256'] for doc_id in canonical}
