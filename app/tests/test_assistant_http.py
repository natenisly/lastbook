#!/usr/bin/env python3
"""Final-round HTTP prefill for Inkling / vLLM: request body shape and HTTP 400 fallback.

  python3 -B -m unittest app.tests.test_assistant_http
"""
import io, json, os, sys, unittest, urllib.error
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'app' / 'tests')]
# Point DATA at the shared bridge fixture before engine.paths is imported (via assistant).
import fixture  # noqa: E402
os.environ['LASTBOOK_DATA'] = str(fixture.ensure())
os.environ.setdefault('SURVIVAL_MODEL_BASE', 'http://127.0.0.1:9')
os.environ.setdefault('SURVIVAL_DISABLE_INDEX_WATCH', '1')

import assistant  # noqa: E402


class _SSE:
    """Line-iterable stand-in for urllib's streaming HTTPResponse."""

    def __init__(self, text='Answer [S1]'):
        chunk = json.dumps({'choices': [{'delta': {'content': text}}]})
        payload = f'data: {chunk}\n\ndata: [DONE]\n'
        self._lines = [ln + b'\n' for ln in payload.encode().splitlines()]
        self.closed = False

    def __iter__(self):
        return iter(self._lines)

    def close(self):
        self.closed = True


def _sse(text='Answer [S1]'):
    return _SSE(text)


class FinalRoundPrefill(unittest.TestCase):
    def test_final_round_sends_empty_assistant_and_vllm_flags(self):
        sent, pieces = [], []

        def open_ok(req, timeout):
            sent.append(json.loads(req.data))
            return _sse()

        msgs = [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'How long without water?'}]
        with mock.patch.object(assistant, 'model_open', side_effect=open_ok):
            result = assistant._http_round(msgs, None, lambda: False, pieces.append, lambda s: None, final=True)
        self.assertEqual(result, ('Answer [S1]', {}))
        self.assertEqual(pieces, ['Answer [S1]'])
        self.assertEqual(len(sent), 1)
        body = sent[0]
        self.assertTrue(body['continue_final_message'])
        self.assertFalse(body['add_generation_prompt'])
        self.assertNotIn('tools', body)
        self.assertEqual(body['messages'][-1], {'role': 'assistant', 'content': ''})
        self.assertEqual(body['messages'][:-1], msgs)

    def test_non_final_round_does_not_prefill(self):
        sent = []

        def open_ok(req, timeout):
            sent.append(json.loads(req.data))
            return _sse()

        msgs = [{'role': 'user', 'content': 'hi'}]
        tools = [{'type': 'function', 'function': {'name': 'search_library'}}]
        with mock.patch.object(assistant, 'model_open', side_effect=open_ok):
            assistant._http_round(msgs, tools, lambda: False, lambda t: None, lambda s: None, final=False)
        body = sent[0]
        self.assertNotIn('continue_final_message', body)
        self.assertNotIn('add_generation_prompt', body)
        self.assertEqual(body['messages'], msgs)
        self.assertEqual(body['tools'], tools)

    def test_http_400_retries_without_prefill(self):
        sent, pieces = [], []

        def open_then_ok(req, timeout):
            body = json.loads(req.data)
            sent.append(body)
            if body.get('continue_final_message'):
                raise urllib.error.HTTPError(req.full_url, 400, 'Bad Request', hdrs={}, fp=io.BytesIO(b'unknown'))
            return _sse('Recovered [S1]')

        msgs = [{'role': 'user', 'content': 'water'}]
        with mock.patch.object(assistant, 'model_open', side_effect=open_then_ok):
            result = assistant._http_round(msgs, None, lambda: False, pieces.append, lambda s: None, final=True)
        self.assertEqual(result, ('Recovered [S1]', {}))
        self.assertEqual(pieces, ['Recovered [S1]'])
        self.assertEqual(len(sent), 2)
        self.assertTrue(sent[0]['continue_final_message'])
        self.assertEqual(sent[0]['messages'][-1], {'role': 'assistant', 'content': ''})
        self.assertNotIn('continue_final_message', sent[1])
        self.assertNotIn('add_generation_prompt', sent[1])
        self.assertEqual(sent[1]['messages'], msgs)

    def test_clean_final_text_strips_think_and_tool_markup(self):
        raw = '<think>planning a tool call</think>\nAbout three days.[S1]\n<tool_call>{"name":"search_library"}</tool_call>'
        self.assertEqual(assistant._clean_final_text(raw), 'About three days.[S1]')

    def test_answer_from_sources_builds_short_fallback(self):
        sess = assistant.Session()
        sess.add({'doc_id': 'd1', 'title': 'Water guide', 'excerpt': 'A person can survive about three days without water.', 'kind': 'pdf', 'page': 1})
        text = assistant._answer_from_sources(sess)
        self.assertIn('three days', text)
        self.assertIn('[S1]', text)
        self.assertIn('Water guide', text)


if __name__ == '__main__':
    unittest.main()
