"""A dependency-free, single-user localhost pilot. One server owns one budget."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from rfp_rag.recovery.runner import load_api_key
from .pilot import embedding_usage, readiness, review_template, save_review, session_report
from .summaries import SummaryInputError, WebSummaries


def create_server(pilot, *, port=8765, summary_client=None):
    """Use HTTPServer on the index's owning thread, never ThreadingHTTPServer."""
    page = (Path(__file__).parent / 'pilot.html').read_bytes()
    local_readiness = readiness(pilot.env)
    summaries = WebSummaries(pilot.root, client=summary_client)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # User questions, document IDs and provider bodies are not access logs.

        def send_value(self, status, value, *, html=False):
            raw = value if html else json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'text/html; charset=utf-8' if html else 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(raw)

        def allowed_request(self):
            expected = f'127.0.0.1:{self.server.server_port}'
            return (self.headers.get('Host') == expected
                and self.headers.get('Origin', f'http://{expected}') == f'http://{expected}')

        def find_record(self, turn_id):
            return next(r for r in pilot.records if r['turn_id'] == turn_id)

        def do_GET(self):
            if not self.allowed_request():
                self.send_value(403, {'error': '표시된 127.0.0.1 주소에서 여세요.'})
                return
            parsed = urlsplit(self.path)
            try:
                if parsed.path == '/':
                    self.send_value(200, page, html=True)
                    return
                if parsed.path == '/api/state':
                    value = {'readiness': local_readiness, 'session_id': pilot.session_id,
                        'documents': pilot.chat.list_documents(), 'active_doc_ids': pilot.chat.active_doc_ids,
                        'filters': pilot.chat.filters, 'approved_doc_ids': pilot.approved_doc_ids,
                        'live_enabled': pilot.chat.llm.config.enabled, 'halted': pilot.halted,
                        'usage': pilot.chat.llm.usage_report(), 'model': pilot.chat.llm.config.model,
                        'semantic_model': pilot.env['index'].get_metadata().get('semantic_embedding_model'),
                        'embedding_usage': embedding_usage(pilot.env),
                        'summary_job': summaries.current(),
                        'turns': [{'turn_id': r['turn_id'], 'question': r['question'],
                                   'status': r['result']['status']} for r in pilot.records]}
                elif parsed.path == '/api/record':
                    value = self.find_record(parse_qs(parsed.query).get('turn_id', [''])[0])
                elif parsed.path == '/api/summary':
                    did = parse_qs(parsed.query).get('doc_id', [''])[0]
                    value = summaries.saved(pilot.env, did)
                    if value['status'] == 'unavailable':
                        value = pilot.saved_summary(did)
                elif parsed.path == '/api/summary/status':
                    value = summaries.status(parse_qs(parsed.query).get('job_id', [''])[0])
                elif parsed.path == '/api/report':
                    value = session_report(pilot.folder)
                    value['web_summary'] = summaries.report()
                else:
                    self.send_value(404, {'error': '없는 경로입니다.'})
                    return
                self.send_value(200, value)
            except (ValueError, TypeError, KeyError, StopIteration):
                self.send_value(400, {'error': '문서·실행 ID 또는 저장 결과를 확인하세요.'})
            except Exception:
                self.send_value(500, {'error': '로컬 조회를 완료하지 못했습니다.'})

        def do_POST(self):
            if not self.allowed_request():
                self.send_value(403, {'error': '현재 로컬 화면에서 제출하세요.'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if self.headers.get('Content-Type') != 'application/json' or not 0 < size <= 32768:
                    raise ValueError('JSON 요청 크기 또는 형식을 확인하세요.')
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError('요청은 JSON 객체여야 합니다.')
                path = urlsplit(self.path).path
                if summaries.running() and path in {
                        '/api/select', '/api/generation', '/api/ask', '/api/reset',
                        '/api/resume', '/api/summary/plan'}:
                    raise SummaryInputError('요약 생성이 진행 중입니다. 완료 후 다른 요청을 실행하세요.')
                if path == '/api/select':
                    value = {'selected': pilot.select(body.get('doc_ids', []), filters=body.get('filters', {}))}
                elif path == '/api/generation':
                    enabled = body.get('enabled')
                    approved = body.get('approved_doc_ids', [])
                    # Validate before reading .env; showing the page never loads a key.
                    if type(enabled) is not bool or enabled and not approved:
                        raise ValueError('API 실행 여부와 전송할 문서를 명시하세요.')
                    if enabled:
                        pilot.env['registry'].resolve_doc_ids(approved)
                        load_api_key(pilot.root)
                    pilot.configure_generation(enabled=enabled, approved_doc_ids=approved)
                    value = {'live_enabled': enabled, 'approved_doc_ids': pilot.approved_doc_ids}
                elif path == '/api/ask':
                    value = pilot.submit(body.get('question'), request_id=body.get('request_id'))
                elif path in {'/api/summary/plan', '/api/summary/generate'}:
                    did = pilot.env['registry'].resolve_doc_ids([body.get('doc_id')])[0]
                    if pilot.chat.active_doc_ids != [did]:
                        raise SummaryInputError('요약할 문서 하나만 선택하고 선택 적용을 누르세요.')
                    if path == '/api/summary/plan':
                        value = summaries.prepare(pilot.env, did)
                    else:
                        if (not pilot.chat.llm.config.enabled
                                or did not in pilot.approved_doc_ids or pilot.halted):
                            raise SummaryInputError('선택 문서의 실제 API 설정을 적용한 뒤 요약을 생성하세요.')
                        value = summaries.start(pilot.env, did, plan_id=body.get('plan_id'),
                                                allow_api=body.get('allow_api', False))
                elif path == '/api/reset':
                    pilot.reset_conversation()
                    value = {'reset': True, 'usage': pilot.chat.llm.usage_report()}
                elif path == '/api/resume':
                    pilot.resume_after_error()
                    value = {'halted': False, 'usage': pilot.chat.llm.usage_report()}
                elif path == '/api/review':
                    record = self.find_record(body.get('turn_id'))
                    review = review_template(record)
                    review.update({k: body.get(k, review[k]) for k in ('reviewer', 'judgements', 'notes')})
                    save_review(pilot.folder / 'turns' / record['turn_id'], review)
                    value = {'saved': True, 'report': session_report(pilot.folder)}
                else:
                    self.send_value(404, {'error': '없는 경로입니다.'})
                    return
                self.send_value(200, value)
            except SummaryInputError as error:
                self.send_value(400, {'error': str(error)})
            except (ValueError, TypeError, KeyError, StopIteration):
                self.send_value(400, {'error': '입력·문서 범위·실행 상태를 확인하세요. 실패 후에는 오류 확인 후 계속을 누르세요.'})
            except Exception:
                self.send_value(500, {'error': '처리를 완료하지 못했습니다. 로컬 실행 기록을 확인하세요.'})

    return HTTPServer(('127.0.0.1', port), Handler)
