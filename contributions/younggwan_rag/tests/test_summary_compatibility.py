"""Keep paid ABCD vectors and exact source integrity during the UI upgrade."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rfp_rag.pilot.summaries import WebSummaries
from rfp_rag.recovery.client import RecoveryLLM
from rfp_rag.workspace import build_demo_workspace, embed_workspace, open_workspace, package_hashes
from rfp_rag.workspace_compat import LEGACY_ABCD, compatible_package
from test_d_profile import FakeEmbeddings
from test_web_summaries import SummaryClient, wait_for_summary


PACKAGE = Path(__file__).resolve().parents[1]


class SummaryCompatibilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.network = patch('socket.socket.connect', side_effect=AssertionError('network forbidden'))
        self.network.start()
        self.addCleanup(self.network.stop)
        key = patch.dict('os.environ', {'OPENAI_API_KEY': 'synthetic-test-key'})
        key.start()
        self.addCleanup(key.stop)

    def old_workspace(self, profile):
        root = self.base / profile
        with patch('rfp_rag.workspace.package_hashes', return_value=LEGACY_ABCD):
            build_demo_workspace(root, config=PACKAGE / f'configs/rag_{profile.lower()}.yaml')
        return root

    def test_only_exact_known_legacy_or_current_runtime_is_accepted(self):
        current = package_hashes()
        self.assertTrue(compatible_package(current, current))
        self.assertTrue(compatible_package(LEGACY_ABCD, current))
        for path in ('retrieval/semantic.py', 'retrieval/chunking.py', 'generation/evidence.py',
                     'generation/llm.py', 'summary_full/engine.py', 'pilot/web.py', 'pilot/pilot.html'):
            changed = dict(current)
            changed[path] = '0' * 64
            self.assertFalse(compatible_package(LEGACY_ABCD, changed), path)
        unknown = dict(current, unexpected_code='0' * 64)
        self.assertFalse(compatible_package(LEGACY_ABCD, unknown))
        old = dict(LEGACY_ABCD)
        old['retrieval/chunking.py'] = '1' * 64
        self.assertFalse(compatible_package(old, current))
        self.assertFalse(compatible_package({}, current))

    def test_abcd_workspaces_open_and_summary_uses_each_saved_model(self):
        embeddings = FakeEmbeddings()
        for profile in 'ABCD':
            with self.subTest(profile=profile):
                root = self.old_workspace(profile)
                if profile in 'CD':
                    embed_workspace(root, live=True, client=embeddings)
                before = len(embeddings.calls)
                with open_workspace(root) as env:
                    client = SummaryClient()
                    service = WebSummaries(root, client=client)
                    plan = service.prepare(env, 'DEMO_A')
                    job = service.start(env, 'DEMO_A', plan_id=plan['plan_id'], allow_api=True)
                    final = wait_for_summary(service, job['job_id'])
                    self.assertEqual(final['status'], 'completed', final)
                    expected = 'gpt-6.1-sol' if profile in 'BD' else 'gpt-5.4-mini'
                    self.assertEqual(plan['model'], expected)
                    self.assertEqual(client.bodies[0]['model'], expected)
                    self.assertEqual(client.bodies[0]['reasoning']['effort'], 'low' if profile in 'BD' else 'none')
                    self.assertEqual(len(embeddings.calls), before)
                    if profile in 'CD':
                        self.assertEqual(env['index'].usage_report()['api_calls'], 0)

    def test_existing_d_manifest_indices_vectors_and_embedding_ledger_stay_identical(self):
        root = self.old_workspace('D')
        embeddings = FakeEmbeddings()
        embed_workspace(root, live=True, client=embeddings)
        protected = {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in root.rglob('*') if p.is_file()}
        calls = len(embeddings.calls)
        with open_workspace(root) as env:
            service = WebSummaries(root, client=SummaryClient())
            plan = service.prepare(env, 'DEMO_A')
            job = service.start(env, 'DEMO_A', plan_id=plan['plan_id'], allow_api=True)
            self.assertEqual(wait_for_summary(service, job['job_id'])['status'], 'completed')
        for relative, sha in protected.items():
            self.assertEqual(hashlib.sha256((root / relative).read_bytes()).hexdigest(), sha, relative)
        self.assertEqual(len(embeddings.calls), calls)
        result = embed_workspace(root, live=True, client=embeddings)
        self.assertTrue(result['reused'])
        self.assertEqual(result['api_calls_this_run'], 0)
        self.assertEqual(len(embeddings.calls), calls)

    def test_legacy_compatibility_never_bypasses_source_or_vector_hash_checks(self):
        root = self.old_workspace('D')
        embed_workspace(root, live=True, client=FakeEmbeddings())
        vectors = root / 'semantic/vectors.npy'
        original = vectors.read_bytes()
        vectors.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with self.assertRaises(ValueError):
            open_workspace(root)
        vectors.write_bytes(original)
        with (root / 'data/source_documents.jsonl').open('a', encoding='utf-8') as handle:
            handle.write('\n')
        with self.assertRaises(ValueError):
            open_workspace(root)

    def test_recovery_parser_forwards_current_common_client_keyword_arguments(self):
        response = type('Response', (), {'output_text': json.dumps({
            'status': 'answered', 'claims': [{'doc_id': 'D001', 'category': 'budget',
                'text': '[budget_vat|fact] 합성 예산',
                'citations': [{'chunk_id': 'E001', 'quote': 'SOURCE'}]}],
            'missing_information': []}), 'output': [], 'status': 'completed'})()
        client = RecoveryLLM(self.base / 'cache')
        plain = client._response_payload(response)
        payload, adjustment = client._response_payload(response, normalize_incomplete=False, with_adjustment=True)
        self.assertEqual(payload, plain)
        self.assertIsNone(adjustment)


if __name__ == '__main__':
    unittest.main()
