"""実APIには接続せず、記録・履歴・追記・中断時の動作を確認する。"""

import json
from copy import deepcopy

import pytest
from google import genai
from google.genai import errors, types

from app import check_homework as check


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    answer = tmp_path / 'lectures/01/answer.md'
    answer.parent.mkdir(parents=True)
    answer.write_text('# 第1回 解答\n\n自分で書いた考察。\n', encoding='utf-8')
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'glossary.json').write_text(json.dumps({'terms': [
        {'id': 'homework-levels', 'definition': '基礎・実装・発展'}
    ]}), encoding='utf-8')
    monkeypatch.setattr(check, 'get_api_key', lambda: 'offline-secret')
    client = genai.Client(api_key='offline-secret', vertexai=False)
    monkeypatch.setattr(check.genai, 'Client', lambda **kw: client)
    calls = []

    def generate(**kw):
        calls.append(deepcopy(kw))
        return types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(role='model', parts=[
                types.Part(text='青。offline-secret\n' + check.END)
            ]), finish_reason='STOP')],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=10 * len(calls), candidates_token_count=5),
        )

    monkeypatch.setattr(client.models, 'generate_content', generate)
    yield tmp_path, answer, client, calls
    client.close()


def test_records_five_calls_and_real_sdk_history(prepared):
    root, answer, client, calls = prepared
    run = check.run_checks(root)
    assert len(calls) == 5
    assert [c.role for c in calls[1]['contents']] == ['user', 'model', 'user']
    assert len(calls[2]['contents']) == 1  # 新規セッションに以前の履歴はない
    assert isinstance(calls[3]['contents'], str)  # 共通演習は単発
    report = answer.read_text(encoding='utf-8')
    assert '自分で書いた考察。' in report
    assert '| 会話2：履歴を参照する | 応答取得 | 20 | 5 |' in report
    assert '基礎・実装・発展' in report
    assert 'offline-secret' not in report
    assert report.count(check.END) == 1
    log = json.loads(next((root / 'homework_runs').glob('*.json')).read_text())
    assert log['status'].startswith('完了')
    assert len(log['results']) == 5
    assert 'offline-secret' not in json.dumps(log)


def test_rerun_replaces_only_generated_section(prepared):
    root, answer, client, calls = prepared
    check.run_checks(root)
    answer.write_text(answer.read_text() + '\n後から書いた考察。\n')
    check.run_checks(root)
    report = answer.read_text()
    assert report.count(check.START) == report.count(check.END) == 1
    assert '自分で書いた考察。' in report and '後から書いた考察。' in report
    assert len(list((root / 'homework_runs').glob('*.json'))) == 2


@pytest.mark.parametrize('failure,expected', [
    (errors.ClientError(429, {'error': {'message': 'offline-secret'}}), 'APIエラー'),
    (KeyboardInterrupt(), '中断'),
])
def test_failure_saves_previous_results_and_unexecuted_cases(prepared, monkeypatch, failure, expected):
    root, answer, client, calls = prepared
    original = client.models.generate_content

    def fail_second(**kw):
        if calls:
            raise failure
        return original(**kw)

    monkeypatch.setattr(client.models, 'generate_content', fail_second)
    run = check.run_checks(root)
    assert not run['status'].startswith('完了')
    assert run['results'][0]['status'] == '応答取得'
    assert run['results'][1]['status'] == expected
    assert run['results'][2]['status'] == '未実施'
    assert 'offline-secret' not in answer.read_text()


def test_missing_key_preserves_answer(prepared, monkeypatch):
    root, answer, client, calls = prepared
    previous = answer.read_bytes()

    def no_key():
        raise SystemExit('キー未設定')

    monkeypatch.setattr(check, 'get_api_key', no_key)
    with pytest.raises(SystemExit):
        check.run_checks(root)
    assert answer.read_bytes() == previous
    assert not (root / 'homework_runs').exists()
    assert calls == []


def test_broken_marker_preserves_answer_and_does_not_call_api(prepared):
    root, answer, client, calls = prepared
    previous = '# 手書き\n' + check.START
    answer.write_text(previous)
    with pytest.raises(ValueError):
        check.run_checks(root)
    assert answer.read_text() == previous
    assert calls == []
