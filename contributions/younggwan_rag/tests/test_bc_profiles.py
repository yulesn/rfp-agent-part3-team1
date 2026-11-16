"""B/C integration and verified C/D vector reuse; no real API calls."""
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rfp_rag.pilot.pilot import PilotSession
from rfp_rag.workspace import (build_demo_workspace, build_workspace, embed_workspace,
                               open_workspace, package_hashes, settings_from_file)
from test_d_profile import FakeEmbeddings


PACKAGE = Path(__file__).resolve().parents[1]
CONFIGS = PACKAGE / 'configs'
spec = importlib.util.spec_from_file_location('prepare_bc_script', PACKAGE / 'scripts/prepare_bc.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class BCProfileTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        blocked = patch('socket.socket.connect', side_effect=AssertionError('network forbidden'))
        blocked.start()
        self.addCleanup(blocked.stop)
        self.d = self.root / 'D'
        self.b = self.root / 'B'
        self.c = self.root / 'C'
        build_demo_workspace(self.d, config=CONFIGS / 'rag_d.yaml')
        self.source = self.d / 'data/source_documents.jsonl'
        self.embeddings = FakeEmbeddings()

    def prepare(self, *, donor=True):
        return prepare.prepare_bc(self.source, workspace_b=self.b, workspace_c=self.c,
                                  reuse_semantic_from=self.d if donor else None)

    def embed_d(self):
        embed_workspace(self.d, live=True, client=self.embeddings)

    def test_four_profiles_change_only_the_two_declared_factors(self):
        profiles = {p: settings_from_file(CONFIGS / f'rag_{p.lower()}.yaml') for p in 'ABCD'}
        for name, value in profiles.items():
            self.assertFalse(value['generation']['enabled'])
            self.assertEqual(value['generation']['model'], 'gpt-6.1-sol' if name in 'BD' else 'gpt-5.4-mini')
            self.assertEqual(value['generation']['reasoning_effort'], 'low' if name in 'BD' else 'none')
            self.assertEqual(value['semantic']['enabled'], name in 'CD')
        generator_swap = deepcopy(profiles['A'])
        generator_swap['generation'].update(model='gpt-6.1-sol', reasoning_effort='low')
        self.assertEqual(profiles['B'], generator_swap)
        retrieval_swap = deepcopy(profiles['A'])
        retrieval_swap['semantic'] = profiles['D']['semantic']
        self.assertEqual(profiles['C'], retrieval_swap)
        generator_swap['semantic'] = profiles['C']['semantic']
        self.assertEqual(profiles['D'], generator_swap)

    def test_prepare_without_d_embeddings_is_offline_and_c_remains_pending(self):
        before = package_hashes()
        with patch.dict(os.environ, {}, clear=True):
            result = self.prepare(donor=False)
        self.assertEqual(result['api_calls_this_run'], 0)
        self.assertEqual(result['C']['semantic']['status'], 'awaiting_embeddings')
        self.assertEqual(package_hashes(), before)
        self.assertFalse((self.c / 'semantic').exists())
        with open_workspace(self.b) as env:
            self.assertEqual(env['phase5_config']['generation']['model'], 'gpt-6.1-sol')
            self.assertTrue(env['index'].search('예산', allowed_doc_ids=['DEMO_A']))
        with self.assertRaises(ValueError):
            open_workspace(self.c)

    def test_reuse_keeps_d_and_c_source_bindings_without_any_extra_calls(self):
        self.embed_d()
        calls = len(self.embeddings.calls)
        donor_files = {p.relative_to(self.d): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in self.d.rglob('*') if p.is_file()}
        with patch.dict(os.environ, {}, clear=True):
            result = self.prepare()
        self.assertEqual(result['C']['semantic']['status'], 'copied_verified')
        self.assertEqual(result['api_calls_this_run'], 0)
        self.assertEqual(len(self.embeddings.calls), calls)
        for relative, expected in donor_files.items():
            self.assertEqual(hashlib.sha256((self.d / relative).read_bytes()).hexdigest(), expected)
        for name in ('metadata.json', 'vectors.npy'):
            self.assertEqual((self.c / 'semantic' / name).read_bytes(), (self.d / 'semantic' / name).read_bytes())
        self.assertEqual(list((self.c / 'semantic_build').glob('*.json')), [])
        self.assertFalse((self.c / 'semantic_outputs').exists())
        self.assertFalse((self.c / 'phase8_outputs').exists())
        with open_workspace(self.c) as env:
            self.assertEqual(env['phase5_config']['generation']['model'], 'gpt-5.4-mini')
            self.assertEqual(env['index'].get_metadata()['semantic_embedding_model'], 'text-embedding-3-large')

    def test_completed_c_reuse_is_idempotent_and_preserves_c_logs_and_budgets(self):
        self.embed_d()
        self.prepare()
        protected = self.c / 'semantic_outputs/do-not-reset.attempt.json'
        protected.parent.mkdir()
        protected.write_text('{"input_bytes":12}', encoding='utf-8')
        saved = (self.c / 'semantic/vectors.npy').read_bytes()
        result = self.prepare()
        self.assertTrue(result['B']['reused'])
        self.assertTrue(result['C']['reused'])
        self.assertEqual(result['C']['semantic']['status'], 'existing_verified')
        self.assertEqual((self.c / 'semantic/vectors.npy').read_bytes(), saved)
        self.assertEqual(protected.read_text(), '{"input_bytes":12}')

    def test_existing_a_cannot_be_overwritten_by_b_profile(self):
        build_demo_workspace(self.b, config=CONFIGS / 'rag_a.yaml')
        manifest = (self.b / 'reports/run_manifest.json').read_bytes()
        with self.assertRaisesRegex(ValueError, '기존 workspace'):
            self.prepare(donor=False)
        self.assertFalse(self.c.exists())
        self.assertEqual((self.b / 'reports/run_manifest.json').read_bytes(), manifest)

    def test_valid_but_different_c_vectors_are_not_silently_accepted_or_replaced(self):
        self.embed_d()
        self.prepare(donor=False)
        different = FakeEmbeddings()
        original = different.create
        def changed(**body):
            response = original(**body)
            for row in response.data:
                row.embedding[2] = 0.5
            return response
        different.create = changed
        embed_workspace(self.c, live=True, client=different)
        previous = (self.c / 'semantic/vectors.npy').read_bytes()
        with self.assertRaisesRegex(ValueError, '기존 C 벡터'):
            prepare.reuse_semantic(self.d, self.c)
        self.assertEqual((self.c / 'semantic/vectors.npy').read_bytes(), previous)

    def test_wrong_source_or_incomplete_d_fails_before_creating_b_c(self):
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.b.exists())
        self.embed_d()
        alternate = self.root / 'different.jsonl'
        alternate.write_bytes(self.source.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, '원문 JSONL'):
            prepare.prepare_bc(alternate, workspace_b=self.b, workspace_c=self.c,
                               reuse_semantic_from=self.d)
        self.assertFalse(self.b.exists())
        self.assertFalse(self.c.exists())

    def test_dimension_mismatch_is_rejected_before_copy(self):
        self.embed_d()
        alternate = self.root / 'dimensions.yaml'
        alternate.write_text((CONFIGS / 'rag_c.yaml').read_text().replace('dimensions: 3072', 'dimensions: 512'))
        build_workspace(self.source, self.c, config=alternate)
        with self.assertRaisesRegex(ValueError, '모델·차원·설정'):
            prepare.reuse_semantic(self.d, self.c)
        self.assertFalse((self.c / 'semantic').exists())

    def test_mismatched_chunk_source_is_rejected_before_copy(self):
        self.embed_d()
        alternate = self.root / 'different-text.jsonl'
        documents = [json.loads(line) for line in self.source.read_text().splitlines() if line.strip()]
        documents[0]['blocks'][0]['text'] += ' 가상 문서의 변경된 사업 범위.'
        alternate.write_text('\n'.join(json.dumps(d, ensure_ascii=False) for d in documents), encoding='utf-8')
        build_workspace(alternate, self.c, config=CONFIGS / 'rag_c.yaml')
        with self.assertRaisesRegex(ValueError, '원문 청크'):
            prepare.reuse_semantic(self.d, self.c)
        self.assertFalse((self.c / 'semantic').exists())

    def test_damaged_d_is_rejected_before_any_b_c_writes(self):
        self.embed_d()
        with (self.d / 'semantic/vectors.npy').open('ab') as stream:
            stream.write(b'damaged')
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.b.exists())
        self.assertFalse(self.c.exists())

    def test_damaged_staged_copy_is_not_published(self):
        self.embed_d()
        self.prepare(donor=False)
        original_copy = prepare.shutil.copyfile
        def damage(source, target):
            result = original_copy(source, target)
            if Path(target).name == 'vectors.npy':
                with Path(target).open('ab') as stream:
                    stream.write(b'damaged')
            return result
        with patch.object(prepare.shutil, 'copyfile', side_effect=damage):
            with self.assertRaises(ValueError):
                prepare.reuse_semantic(self.d, self.c)
        self.assertFalse((self.c / 'semantic').exists())
        self.assertEqual(list(self.c.glob('.semantic-reuse-*')), [])
        with open_workspace(self.d):
            pass

    def test_incomplete_c_spending_records_cannot_be_concealed_by_copy(self):
        self.embed_d()
        self.prepare(donor=False)
        record = self.c / 'semantic_build/000000.attempt.json'
        record.parent.mkdir()
        record.write_text('{"unfinished":true}')
        with self.assertRaisesRegex(ValueError, '임베딩 시도 기록'):
            prepare.reuse_semantic(self.d, self.c)
        self.assertFalse((self.c / 'semantic').exists())
        self.assertEqual(record.read_text(), '{"unfinished":true}')

    def test_equal_or_nested_workspace_paths_are_rejected(self):
        for b, c in ((self.b, self.b), (self.b, self.b / 'C')):
            with self.assertRaisesRegex(ValueError, '별도 경로'):
                prepare.prepare_bc(self.source, workspace_b=b, workspace_c=c)
        with self.assertRaises(ValueError):
            prepare.prepare_bc(self.source, workspace_b=self.d, workspace_c=self.c,
                               reuse_semantic_from=self.d)

    def test_four_way_requests_hold_evidence_and_prompt_fixed_within_retrieval_pair(self):
        from regression.conversation.test_focus import GroundedClient
        self.embed_d()
        self.prepare()
        a = self.root / 'A'
        build_workspace(self.source, a, config=CONFIGS / 'rag_a.yaml')
        bodies = {}
        for name, root in [('A', a), ('B', self.b), ('C', self.c), ('D', self.d)]:
            generator = GroundedClient()
            before = len(self.embeddings.calls)
            with open_workspace(root, allow_query_api=True, embedding_client=self.embeddings) as env:
                pilot = PilotSession(env, client=generator)
                pilot.select(['DEMO_A'])
                with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-dummy-not-a-real-key'}):
                    pilot.configure_generation(enabled=True, approved_doc_ids=['DEMO_A'])
                    record = pilot.submit('사업 예산은?', request_id=name)
                self.assertEqual(record['result']['execution_status'], 'completed', name)
                self.assertEqual({c['doc_id'] for c in record['result']['source_chunks']}, {'DEMO_A'})
                self.assertEqual(len(generator.bodies), 1)
                bodies[name] = deepcopy(generator.bodies[0])
            self.assertEqual(len(self.embeddings.calls) > before, name in 'CD')
        for name, body in bodies.items():
            self.assertEqual(body.pop('model'), 'gpt-6.1-sol' if name in 'BD' else 'gpt-5.4-mini')
            self.assertEqual(body.pop('reasoning'), {'effort': 'low' if name in 'BD' else 'none'})
        self.assertEqual(bodies['A'], bodies['B'])
        self.assertEqual(bodies['C'], bodies['D'])

    def test_cli_reports_zero_api_and_missing_source_is_nonzero(self):
        with redirect_stdout(io.StringIO()) as output:
            code = prepare.main(['--source', str(self.source), '--workspace-b', str(self.b),
                                 '--workspace-c', str(self.c)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())['api_calls_this_run'], 0)
        with redirect_stderr(io.StringIO()) as errors:
            code = prepare.main(['--source', str(self.root / 'missing.jsonl'),
                                 '--workspace-b', str(self.b), '--workspace-c', str(self.c)])
        self.assertEqual(code, 1)
        self.assertIn('원문 JSONL', errors.getvalue())


if __name__ == '__main__':
    unittest.main()
