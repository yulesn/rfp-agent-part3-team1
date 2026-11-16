"""Build B/C locally and optionally reuse verified C/D document embeddings.

This script never creates an OpenAI client. The D-profile implementation is a
prerequisite; all runtime modules, A/D profiles and existing logs stay unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import sys
import tempfile

from rfp_rag.retrieval.io import sha256_file, write_json
from rfp_rag.retrieval.semantic import SemanticSearchIndex, _lock
from rfp_rag.workspace import build_workspace, open_workspace, settings_from_file


PACKAGE = Path(__file__).resolve().parents[1]


def _preflight(source, destinations, configs):
    """Check both destinations before either build, including protected scopes."""
    if not source.is_file():
        raise ValueError('원문 JSONL 경로를 확인하세요: ' + str(source))
    for left in destinations:
        for right in destinations:
            if left != right and (left.is_relative_to(right) or right.is_relative_to(left)):
                raise ValueError('B와 C workspace는 서로 포함되지 않는 별도 경로여야 합니다.')
    if len(set(destinations)) != 2:
        raise ValueError('B와 C workspace는 별도 경로여야 합니다.')
    source_hash = sha256_file(source)
    for destination, config in zip(destinations, configs):
        if not destination.exists():
            continue
        with open_workspace(destination, lexical_only=True) as env:
            saved = env['source_manifest']
            if saved['input_sha256'] != source_hash or saved['settings'] != settings_from_file(config):
                raise ValueError('기존 workspace의 원문 또는 설정이 다릅니다. 새 B/C 경로를 지정하세요.')


def _matching_semantic(source, target):
    source_settings = source['source_manifest']['settings']['semantic']
    target_settings = target['source_manifest']['settings']['semantic']
    if not source_settings.get('enabled') or source_settings != target_settings:
        raise ValueError('재사용할 의미 검색의 모델·차원·설정이 C와 다릅니다.')
    origin = source['index'].get_metadata()
    destination = target['index'].get_metadata()
    if origin.get('lexical_index_fingerprint') != destination['index_fingerprint']:
        raise ValueError('원문 청크·순서·검색 인덱스가 달라 임베딩을 재사용할 수 없습니다.')


def reuse_semantic(source_root, target_root):
    """Copy only source-bound vector artifacts; never copy budgets or chat logs.

    Source and staged copy are validated by the same runtime used by search.
    Publication uses the embedding builder's lock and an atomic directory move.
    Existing target embeddings are verified and retained, never overwritten.
    """
    source_root = Path(source_root).expanduser().resolve()
    target_root = Path(target_root).expanduser().resolve()
    if source_root == target_root or source_root.is_relative_to(target_root) or target_root.is_relative_to(source_root):
        raise ValueError('재사용 원본과 C workspace는 서로 포함되지 않는 별도 경로여야 합니다.')
    with ExitStack() as stack:
        source = stack.enter_context(open_workspace(source_root))
        target = stack.enter_context(open_workspace(target_root, lexical_only=True))
        _matching_semantic(source, target)
        config = target['source_manifest']['settings']['semantic']
        with _lock(target_root / 'semantic_build/.lock'):
            if (target_root / 'semantic').exists():
                index = SemanticSearchIndex(target['index'], target_root, target['chunks'], config)
                if sha256_file(target_root / 'semantic/vectors.npy') != sha256_file(source_root / 'semantic/vectors.npy'):
                    raise ValueError('기존 C 벡터가 재사용 원본과 다릅니다. 덮어쓰지 않으므로 새 C 경로를 지정하세요.')
                return {'status': 'existing_verified', 'api_calls_this_run': 0,
                        'semantic_index_fingerprint': index.get_metadata()['semantic_index_fingerprint']}
            # A partial C build has its own spending history. Keep it visible
            # instead of concealing a failed/uncertain batch by installing D.
            if any((target_root / 'semantic_build').glob('*.json')):
                raise ValueError('C에 임베딩 시도 기록이 있습니다. 기록을 보존하고 새 C 경로를 지정하세요.')
            stage = Path(tempfile.mkdtemp(prefix='.semantic-reuse-', dir=target_root))
            try:
                (stage / 'semantic').mkdir()
                for name in ('metadata.json', 'vectors.npy'):
                    shutil.copyfile(source_root / 'semantic' / name, stage / 'semantic' / name)
                index = SemanticSearchIndex(target['index'], stage, target['chunks'], config)
                fingerprint = index.get_metadata()['semantic_index_fingerprint']
                (stage / 'semantic').rename(target_root / 'semantic')
                receipt = {'status': 'copied_verified', 'source_workspace': str(source_root),
                           'semantic_index_fingerprint': fingerprint, 'api_calls_this_run': 0,
                           'source_generation_and_query_logs_copied': False}
                write_json(target_root / 'reports/semantic_reuse.json', receipt)
            finally:
                shutil.rmtree(stage, ignore_errors=True)
        return receipt


def prepare_bc(source, *, workspace_b='workspaces/team-b-v1', workspace_c='workspaces/team-c-v1',
               reuse_semantic_from=None):
    """Prepare the two requested profiles with zero generation/embedding calls."""
    source = Path(source).expanduser().resolve()
    destinations = [Path(p).expanduser().resolve() for p in (workspace_b, workspace_c)]
    configs = [PACKAGE / 'configs/rag_b.yaml', PACKAGE / 'configs/rag_c.yaml']
    _preflight(source, destinations, configs)
    if reuse_semantic_from is not None:
        donor = Path(reuse_semantic_from).expanduser().resolve()
        if any(donor == dest or donor.is_relative_to(dest) or dest.is_relative_to(donor) for dest in destinations):
            raise ValueError('재사용 원본 workspace를 B/C 경로와 분리하세요.')
        # Validate the donor before building or touching either destination.
        with open_workspace(donor) as env:
            if not env['source_manifest']['settings']['semantic'].get('enabled'):
                raise ValueError('재사용할 workspace에 완료된 의미 검색 인덱스가 필요합니다.')
            if env['source_manifest']['input_sha256'] != sha256_file(source):
                raise ValueError('재사용 원본과 B/C에 지정한 원문 JSONL이 다릅니다.')
    built = [build_workspace(source, dest, config=config)
             for dest, config in zip(destinations, configs)]
    semantic = {'status': 'awaiting_embeddings', 'api_calls_this_run': 0}
    if reuse_semantic_from is not None:
        semantic = reuse_semantic(reuse_semantic_from, destinations[1])
    elif (destinations[1] / 'semantic').exists():
        with open_workspace(destinations[1]) as env:
            semantic = {'status': 'existing_verified', 'api_calls_this_run': 0,
                        'semantic_index_fingerprint': env['index'].get_metadata()['semantic_index_fingerprint']}
    return {'B': built[0], 'C': {**built[1], 'semantic': semantic},
            'api_calls_this_run': 0, 'semantic_accuracy': None,
            'independent_quality_verified': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description='B/C 로컬 구축 및 검증된 문서 임베딩 재사용 (API 호출 없음)')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--workspace-b', type=Path, default=Path('workspaces/team-b-v1'))
    parser.add_argument('--workspace-c', type=Path, default=Path('workspaces/team-c-v1'))
    parser.add_argument('--reuse-semantic-from', type=Path)
    args = parser.parse_args(argv)
    try:
        result = prepare_bc(args.source, workspace_b=args.workspace_b, workspace_c=args.workspace_c,
                            reuse_semantic_from=args.reuse_semantic_from)
    except (ValueError, OSError, RuntimeError) as error:
        print('B/C 준비 중단: ' + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
