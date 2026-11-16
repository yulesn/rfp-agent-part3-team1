"""A/D integration tests use invented sources and injected clients; no paid API."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rfp_rag.cli import main, parser
from rfp_rag.generation.llm import LLMConfig, OpenAIGroundedLLM
from rfp_rag.workspace import build_demo_workspace, open_workspace, settings_from_file


PACKAGE = Path(__file__).resolve().parents[1]
CONFIGS = PACKAGE / 'configs'
ORIGINAL_A_YAML = '''# 기존 Phase 8 기준선. Task 2의 모델/검색 성능 개선은 적용하지 않았습니다.
generation:
  enabled: false
  model: gpt-5.4-mini
  max_calls: 8
  max_total_input_bytes: 600000
retrieval:
  method: rrf
  top_k: 8
'''


class FakeEmbeddings:
    def __init__(self):
        self.embeddings = self
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        vectors = []
        for number, text in enumerate(kwargs['input']):
            vector = [0.0] * kwargs['dimensions']
            vector[0] = 1.0
            if '예산' in text or '사업비' in text:
                vector[1] = 1.0
            vectors.append(SimpleNamespace(index=number, embedding=vector))
        return SimpleNamespace(data=vectors, usage=SimpleNamespace(prompt_tokens=7,
                               total_tokens=7), model=kwargs['model'])


class FakeResponses:
    def __init__(self):
        self.responses = self
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = {'status': 'answered', 'claims': [{'doc_id': 'A', 'category': 'budget',
                   'text': '사업비는 1억 원이다.',
                   'citations': [{'chunk_id': 'c1', 'quote': '사업비 1억 원'}]}],
                   'missing_information': []}
        return SimpleNamespace(status='completed', output_text=json.dumps(payload),
                               output=[], usage=SimpleNamespace(input_tokens=101,
                               output_tokens=31, total_tokens=132))


class DProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        network = patch('socket.socket.connect', side_effect=AssertionError('network forbidden'))
        network.start()
        self.addCleanup(network.stop)

    def build_d(self):
        root = self.root / 'D'
        result = build_demo_workspace(root, config=CONFIGS / 'rag_d.yaml')
        self.assertEqual(result['api_calls'], 0)
        return root

    def test_a_yaml_is_exact_original_and_default_python_path_stays_a(self):
        self.assertEqual((CONFIGS / 'rag_a.yaml').read_text(encoding='utf-8'), ORIGINAL_A_YAML)
        baseline = settings_from_file(CONFIGS / 'rag_a.yaml')
        self.assertEqual(settings_from_file(), baseline)
        self.assertEqual(baseline['generation']['model'], 'gpt-5.4-mini')
        self.assertEqual(baseline['generation']['reasoning_effort'], 'none')
        self.assertFalse(baseline['semantic']['enabled'])

    def test_d_is_active_yaml_and_explicit_profile(self):
        active = settings_from_file(CONFIGS / 'rag.yaml')
        self.assertEqual(active, settings_from_file(CONFIGS / 'rag_d.yaml'))
        self.assertEqual(active['generation']['model'], 'gpt-6.1-sol')
        self.assertEqual(active['generation']['reasoning_effort'], 'low')
        self.assertFalse(active['generation']['enabled'])
        self.assertTrue(active['semantic']['enabled'])
        self.assertEqual(active['semantic']['model'], 'text-embedding-3-large')
        self.assertEqual(active['semantic']['dimensions'], 3072)

    def test_a_retains_lexical_results_and_never_uses_embeddings(self):
        baseline = self.root / 'A'
        old_default = self.root / 'old_default'
        build_demo_workspace(baseline, config=CONFIGS / 'rag_a.yaml')
        build_demo_workspace(old_default)
        fake = FakeEmbeddings()
        with open_workspace(baseline, embedding_client=fake) as env_a:
            with open_workspace(old_default) as env_old:
                query = '사업 예산과 기간은?'
                actual = env_a['index'].search(query, method='rrf', allowed_doc_ids=['DEMO_A'])
                expected = env_old['index'].search(query, method='rrf', allowed_doc_ids=['DEMO_A'])
        self.assertEqual(actual, expected)
        self.assertTrue(actual)
        self.assertEqual(fake.calls, [])

    def test_d_lexical_build_has_no_embedding_side_effect_and_is_repeatable(self):
        root = self.build_d()
        result = build_demo_workspace(root, config=CONFIGS / 'rag_d.yaml')
        self.assertTrue(result['reused'])
        self.assertEqual(result['api_calls'], 0)
        with open_workspace(root, lexical_only=True) as env:
            self.assertEqual(env['phase5_config']['generation']['model'], 'gpt-6.1-sol')
        with self.assertRaises((ValueError, RuntimeError, FileNotFoundError)):
            open_workspace(root)

    def test_embedding_build_requires_explicit_live_and_reuses_completed_vectors(self):
        from rfp_rag.workspace import embed_workspace
        root = self.build_d()
        fake = FakeEmbeddings()
        with self.assertRaises((ValueError, RuntimeError)):
            embed_workspace(root, client=fake)
        self.assertEqual(fake.calls, [])
        manifest = (root / 'reports/run_manifest.json').read_bytes()
        embed_workspace(root, live=True, client=fake)
        self.assertGreater(len(fake.calls), 0)
        count = len(fake.calls)
        embed_workspace(root, live=True, client=fake)
        self.assertEqual(len(fake.calls), count)
        self.assertEqual((root / 'reports/run_manifest.json').read_bytes(), manifest)
        self.assertTrue(all(c['model'] == 'text-embedding-3-large' for c in fake.calls))
        self.assertTrue(all(c['dimensions'] == 3072 for c in fake.calls))

    def test_d_scope_is_preserved_and_query_api_is_explicit(self):
        from rfp_rag.workspace import embed_workspace
        root = self.build_d()
        fake = FakeEmbeddings()
        embed_workspace(root, live=True, client=fake)
        count = len(fake.calls)
        with open_workspace(root, embedding_client=fake) as env:
            self.assertEqual(len(env['index'].get_catalog()), 2)
            self.assertEqual(len(fake.calls), count)
            with self.assertRaises((ValueError, RuntimeError)):
                env['index'].search('사업 예산은?', allowed_doc_ids=['DEMO_A'])
        self.assertEqual(len(fake.calls), count)
        with open_workspace(root, allow_query_api=True, embedding_client=fake) as env:
            hits = env['index'].search('사업 예산은?', allowed_doc_ids=['DEMO_A'])
            self.assertTrue(hits)
            self.assertEqual({hit['doc_id'] for hit in hits}, {'DEMO_A'})
            env['registry'].validate_chunks(hits, allowed_doc_ids=['DEMO_A'])
            self.assertGreater(len(fake.calls), count)

    def test_d_cannot_replace_existing_a_workspace(self):
        root = self.root / 'A'
        build_demo_workspace(root, config=CONFIGS / 'rag_a.yaml')
        original = (root / 'reports/run_manifest.json').read_bytes()
        with self.assertRaisesRegex(ValueError, '새 workspace'):
            build_demo_workspace(root, config=CONFIGS / 'rag_d.yaml')
        self.assertEqual((root / 'reports/run_manifest.json').read_bytes(), original)

    def test_cli_exposes_separate_build_and_paid_embedding_and_query_flags(self):
        args = parser().parse_args(['embed', '--workspace', str(self.root), '--live',
                                   '--env-file', '.env'])
        self.assertTrue(args.live)
        self.assertEqual(args.env_file, Path('.env'))
        args = parser().parse_args(['serve', '--workspace', str(self.root), '--live-search'])
        self.assertTrue(args.live_search)
        args = parser().parse_args(['search', '--workspace', str(self.root), '--live',
                                   '--doc-id', 'DEMO_A', '--question', '사업 내용은?'])
        self.assertTrue(args.live)
        args = parser().parse_args(['build', '--demo', '--workspace', str(self.root)])
        self.assertFalse(getattr(args, 'live', False))

    def test_d_cli_catalog_remains_offline_before_embedding_build(self):
        root = self.build_d()
        with redirect_stdout(io.StringIO()) as output:
            result = main(['catalog', '--workspace', str(root)])
        self.assertEqual(result, 0)
        self.assertEqual(len(json.loads(output.getvalue())), 2)

    def test_sol_none_is_rejected_before_request(self):
        with self.assertRaises(ValueError):
            LLMConfig(model='gpt-6.1-sol', reasoning_effort='none')

    def test_d_pilot_keeps_embedding_usage_separate_from_generation(self):
        from regression.conversation.test_focus import GroundedClient
        from rfp_rag.pilot.pilot import PilotSession, readiness, session_report
        from rfp_rag.workspace import embed_workspace
        root = self.build_d()
        embeddings = FakeEmbeddings()
        embed_workspace(root, live=True, client=embeddings)
        build_calls = len(embeddings.calls)
        generator = GroundedClient()
        with open_workspace(root, allow_query_api=True, embedding_client=embeddings) as env:
            pilot = PilotSession(env, client=generator)
            pilot.select(['DEMO_A'])
            self.assertEqual(readiness(env)['api_calls'], 0)
            self.assertEqual(len(embeddings.calls), build_calls)
            self.assertEqual(generator.bodies, [])
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-dummy-not-a-real-key'}):
                pilot.configure_generation(enabled=True, approved_doc_ids=['DEMO_A'])
                record = pilot.submit('사업 예산은?', request_id='D-integration')
                replay = pilot.submit('사업 예산은?', request_id='D-integration')
            self.assertEqual(record, replay)
            self.assertEqual(record['model'], 'gpt-6.1-sol')
            self.assertEqual(len(generator.bodies), 1)
            self.assertEqual(generator.bodies[0]['reasoning'], {'effort': 'low'})
            self.assertEqual(record['usage_delta']['api_calls'], 1)
            self.assertEqual(record['usage_delta']['mock_calls'], 1)
            self.assertEqual(record['usage_delta']['live_api_calls'], 0)
            self.assertGreater(record['embedding_usage_delta']['api_calls'], 0)
            self.assertTrue(record['result']['source_chunks'])
            self.assertEqual({c['doc_id'] for c in record['result']['source_chunks']}, {'DEMO_A'})
            report = session_report(pilot.folder)
            self.assertEqual(report['usage']['api_calls'], 1)
            self.assertEqual(report['embedding_usage']['api_calls'],
                             record['embedding_usage_delta']['api_calls'])
            self.assertIsNone(report['accuracy_on_reviewed_live_turns'])
            self.assertFalse(report['independent_quality_verified'])

    def test_sol_low_request_keeps_structured_output_and_store_false(self):
        fake = FakeResponses()
        sha = 'a' * 64
        schema = {'type': 'object', 'additionalProperties': False,
                  'properties': {'status': {'type': 'string'}, 'claims': {'type': 'array'},
                                 'missing_information': {'type': 'array'}},
                  'required': ['status', 'claims', 'missing_information']}
        evidence = [{'chunk_id': 'c1', 'doc_id': 'A', 'canonical_doc_id': 'A',
                     'sha256': sha, 'file_name': '가상.hwp', 'locator': 'Section0 paragraph1',
                     'text': '사업비 1억 원, 부가세 포함', 'kind': 'prose',
                     'source_method': 'raw_parser'}]
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-dummy-not-a-real-key'}):
            config = LLMConfig(enabled=True, model='gpt-6.1-sol', reasoning_effort='low')
            llm = OpenAIGroundedLLM(self.root / 'cache', config=config,
                                   allowed_sources={'A': sha}, client=fake)
            result = llm.generate(task='qa', question='예산은?', evidence=evidence, schema=schema)
        self.assertEqual(len(fake.calls), 1)
        request = fake.calls[0]
        self.assertEqual(request['model'], 'gpt-6.1-sol')
        self.assertEqual(request['reasoning'], {'effort': 'low'})
        self.assertIs(request['store'], False)
        self.assertEqual(request['text']['format']['type'], 'json_schema')
        self.assertTrue(request['text']['format']['strict'])
        self.assertEqual(result['execution_mode'], 'injected_test_client')
        self.assertFalse(result['live_api'])


if __name__ == '__main__':
    unittest.main()
