"""Command line entry points; importing this module has no execution side effects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .workspace import build_demo_workspace, build_workspace, embed_workspace, open_workspace


def parser():
    p = argparse.ArgumentParser(prog='rfp-rag', description='원문과 인용을 보존하는 RFP RAG')
    sub = p.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build', help='추출 원문 JSONL로 로컬 색인 구축')
    source = build.add_mutually_exclusive_group(required=True)
    source.add_argument('--source', type=Path)
    source.add_argument('--demo', action='store_true')
    build.add_argument('--workspace', type=Path, required=True)
    build.add_argument('--config', type=Path)
    embed = sub.add_parser('embed', help='선택한 workspace 원문을 의미 검색용으로 색인 (실제 API는 --live)')
    embed.add_argument('--workspace', type=Path, required=True)
    embed.add_argument('--live', action='store_true')
    embed.add_argument('--env-file', type=Path)
    for name in ('catalog', 'search', 'ask', 'summarize', 'validate', 'serve'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--workspace', type=Path, required=True)
        if name in ('search', 'ask', 'summarize', 'validate'):
            cmd.add_argument('--doc-id', action='append', required=True)
        if name in ('search', 'ask'):
            cmd.add_argument('--question', action='append', required=True)
        if name in ('search', 'ask', 'summarize', 'validate'):
            cmd.add_argument('--live', action='store_true')
            cmd.add_argument('--env-file', type=Path)
        if name == 'summarize':
            cmd.add_argument('--run-id', default='summary-01')
        if name == 'serve':
            cmd.add_argument('--port', type=int, default=8765)
            cmd.add_argument('--live-search', action='store_true',
                             help='D의 질문 임베딩 API 허용; 웹 답변 생성 설정과 별개')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'build':
        result = (build_demo_workspace(args.workspace, config=args.config) if args.demo else
                  build_workspace(args.source, args.workspace, config=args.config))
    elif args.command == 'embed':
        def progress(value):
            print(f"문서 임베딩 {value['completed_batches']}/{value['total_batches']} 배치 완료",
                  file=sys.stderr, flush=True)
        result = embed_workspace(args.workspace, live=args.live, env_file=args.env_file,
                                 progress=progress)
    elif args.command == 'serve':
        from .pilot.web import create_server
        from .pilot.pilot import PilotSession
        # One single-threaded server owns its SQLite handle and cumulative budget.
        with open_workspace(args.workspace, allow_query_api=args.live_search) as env:
            if env['index'].get_metadata().get('semantic_embedding_model') and not args.live_search:
                parser().error('D 웹 검색은 --live-search가 필요합니다. 질문 임베딩 API를 허용한 뒤 실행하세요.')
            server = create_server(PilotSession(env), port=args.port)
            print(f'http://127.0.0.1:{server.server_address[1]}', flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
        return 0
    else:
        from .pilot.pilot import PilotSession
        from .workflows import search, submit_questions, summarize, validate
        if getattr(args, 'live', False) and getattr(args, 'env_file', None) is not None:
            from dotenv import load_dotenv
            load_dotenv(args.env_file, override=False)
        with open_workspace(args.workspace, allow_query_api=getattr(args, 'live', False),
                            lexical_only=args.command == 'catalog') as env:
            if args.command == 'catalog':
                result = env['index'].get_catalog()
            elif args.command == 'search':
                result = [search(env, q, doc_ids=args.doc_id) for q in args.question]
            elif args.command == 'ask':
                pilot = PilotSession(env)
                records = submit_questions(pilot, args.question, doc_ids=args.doc_id,
                                           live=args.live, env_file=args.env_file)
                result = {'session': str(pilot.folder), 'records': records}
            elif args.command == 'summarize':
                runner, results = summarize(env, args.doc_id, live=args.live,
                                            run_id=args.run_id, env_file=args.env_file)
                result = {'summary_run': str(runner.folder), 'results': results}
            else:
                folder, report = validate(env, args.doc_id, live=args.live, env_file=args.env_file)
                result = {'validation_run': str(folder), 'report': report}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == 'ask':
        return int(any(r['result'].get('execution_status') in
                       {'error', 'validation_failed', 'incomplete', 'not_approved'}
                       for r in result['records']))
    if args.command == 'validate':
        return int(not result['report']['mechanical_checks_passed'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
