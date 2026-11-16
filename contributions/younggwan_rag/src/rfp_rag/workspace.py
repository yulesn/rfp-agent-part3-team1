"""Build and open a portable local corpus; never call an embedding or chat API."""
from __future__ import annotations

from contextlib import AbstractContextManager
from importlib import resources
import json
from pathlib import Path
import shutil
import tempfile

from .generation.evidence import EvidenceRegistry
from .generation.runtime import environment_report
from .retrieval.chunking import build_chunks
from .retrieval.io import load_chunks, load_documents, sha256_file, verify_saved_run, write_json, write_jsonl
from .retrieval.quality import validate_chunks
from .retrieval.retrieval import LocalSearchIndex


def package_hashes():
    package = Path(__file__).resolve().parent
    return {p.relative_to(package).as_posix(): sha256_file(p)
            for p in sorted(package.rglob('*')) if p.is_file() and p.suffix in {'.py', '.json', '.html'}}


def settings_from_file(path=None):
    """Merge user YAML over the existing baseline, without choosing a new model."""
    defaults = json.loads(resources.files('rfp_rag').joinpath('defaults.json').read_text(encoding='utf-8'))
    settings = {
        'chunking': defaults['default']['chunking'],
        'index': defaults['default']['retrieval'],
        'retrieval': defaults['phase6']['retrieval'],
        'generation': {**defaults['phase5']['generation'], **defaults['phase8']['generation']},
        'conversation': defaults['phase6']['conversation'],
        'summary': defaults['phase5']['summary'],
    }
    if path is not None:
        import yaml
        supplied = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        if not isinstance(supplied, dict) or set(supplied) - set(settings):
            raise ValueError('설정은 chunking/index/retrieval/generation/conversation/summary로 구성하세요.')
        for key, values in supplied.items():
            if not isinstance(values, dict) or set(values) - set(settings[key]):
                raise ValueError(f'지원하지 않는 설정: {key}')
            settings[key].update(values)
    if settings['generation'].get('enabled') is not False:
        raise ValueError('설정의 enabled는 false로 유지하고 실행 시 --live를 지정하세요.')
    return settings


def _config_files(settings):
    defaults = json.loads(resources.files('rfp_rag').joinpath('defaults.json').read_text(encoding='utf-8'))
    defaults['default']['chunking'] = settings['chunking']
    defaults['default']['retrieval'] = settings['index']
    defaults['phase5'].update(generation=settings['generation'], retrieval=settings['retrieval'],
                              summary=settings['summary'], pricing=None)
    defaults['phase6'].update(conversation=settings['conversation'], retrieval=settings['retrieval'])
    defaults['phase8']['generation'] = {key: settings['generation'][key]
                                      for key in ('max_calls', 'max_total_input_bytes')}
    return defaults


def build_workspace(source, destination, *, config=None):
    """Create a new source-bound index atomically. Existing workspaces are verified."""
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    settings = settings_from_file(config)
    if destination.exists():
        with open_workspace(destination) as env:
            saved = env['source_manifest']
            if saved['input_sha256'] != sha256_file(source) or saved['settings'] != settings:
                raise ValueError('원문 또는 설정이 바뀌었습니다. 새 workspace 경로를 지정하세요.')
            return {'workspace': str(destination), 'documents': len(env['documents']),
                    'chunks': len(env['chunks']), 'reused': True, 'api_calls': 0}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + '.building-', dir=destination.parent))
    try:
        (temporary / 'data').mkdir()
        shutil.copyfile(source, temporary / 'data/source_documents.jsonl')
        documents = load_documents(temporary)
        if not documents:
            raise ValueError('색인할 원문 문서가 없습니다.')
        chunks, audit = build_chunks(documents, settings['chunking'])
        quality = validate_chunks(documents, chunks, settings['chunking']['chunk_size_bytes'])
        if not quality['valid'] or not chunks:
            raise ValueError('청크 원문 보존 검사에 실패했습니다: ' + str(quality.get('errors', []))[:1000])
        EvidenceRegistry(documents, chunks)
        write_jsonl(temporary / 'data/chunks.jsonl', chunks)
        for name, value in _config_files(settings).items():
            write_json(temporary / f'configs/{name}.json', value)
        with LocalSearchIndex(temporary / 'index') as index:
            index.build(chunks, settings['index'])
        write_json(temporary / 'reports/chunk_audit.json', audit)
        write_json(temporary / 'reports/chunk_integrity.json', quality)
        files = [p for folder in ('data', 'configs', 'index', 'reports')
                 for p in (temporary / folder).rglob('*') if p.is_file()]
        manifest = {'version': 'rfp-team-workspace-v1', 'input_sha256': sha256_file(source),
                    'settings': settings, 'package_code_sha256': package_hashes(),
                    'files_sha256': {p.relative_to(temporary).as_posix(): sha256_file(p) for p in sorted(files)},
                    'code_sha256': {}, 'api_calls': 0, 'semantic_accuracy': None}
        write_json(temporary / 'reports/run_manifest.json', manifest)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {'workspace': str(destination), 'documents': len(documents),
            'chunks': len(chunks), 'reused': False, 'api_calls': 0}


class Workspace(dict, AbstractContextManager):
    def __exit__(self, *exc):
        self['index'].close()


def open_workspace(root):
    """Verify exact data, index and package bindings before exposing any evidence."""
    root = Path(root).expanduser().resolve()
    manifest = verify_saved_run(root)
    if manifest.get('version') != 'rfp-team-workspace-v1':
        raise ValueError('이 패키지로 생성한 workspace가 아닙니다. 원문 JSONL로 새로 구축하세요.')
    if manifest.get('package_code_sha256') != package_hashes():
        raise ValueError('구현 코드가 바뀌었습니다. 새 workspace를 구축하세요.')
    environment = environment_report(root)
    if not environment['compatible']:
        raise ValueError('색인 구축 환경과 현재 의존성이 다릅니다. 현재 환경에서 새 workspace를 구축하세요.')
    documents, chunks = load_documents(root), load_chunks(root)
    registry = EvidenceRegistry(documents, chunks)
    index = LocalSearchIndex(root / 'index')
    try:
        if index.get_metadata()['chunk_count'] != len(chunks):
            raise ValueError('저장 청크와 인덱스 개수가 다릅니다.')
        return Workspace(root=root, documents=documents, chunks=chunks, registry=registry,
                         index=index, environment=environment, source_manifest=manifest,
                         config=json.loads((root / 'configs/phase4.json').read_text(encoding='utf-8')),
                         phase5_config=json.loads((root / 'configs/phase5.json').read_text(encoding='utf-8')))
    except BaseException:
        index.close()
        raise


def build_demo_workspace(destination, *, config=None):
    """The bundled documents are invented fixtures, not a quality benchmark."""
    source = resources.files('rfp_rag').joinpath('demo_documents.jsonl')
    with resources.as_file(source) as path:
        return build_workspace(path, destination, config=config)
