"""通信部分だけを置き換え、実際のSDKで会話履歴の引き継ぎを確認する。"""

from copy import deepcopy
from unittest.mock import Mock

import pytest
from google import genai
from google.genai import errors, types

from app import ask


def reply(text="LoRAは追加学習の手法です。", usage=True):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(
            role="model", parts=[types.Part(text=text)]
        ), finish_reason="STOP")],
        usage_metadata=(types.GenerateContentResponseUsageMetadata(
            prompt_token_count=12, candidates_token_count=8
        ) if usage else None),
    )


@pytest.fixture
def offline_client(monkeypatch):
    # ダミーキーはSDKの初期化用。実際のAPIキーやネットワークを使用しない。
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(ask, "get_api_key", lambda: "offline-test-key")
    client = genai.Client(api_key="offline-test-key", vertexai=False)
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    calls = []
    results = [reply(), reply("先ほどのLoRAについて補足します。")]

    def generate(**kwargs):
        calls.append(deepcopy(kwargs))
        result = results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(client.models, "generate_content", generate)
    yield calls, results
    client.close()


def enter(monkeypatch, values):
    answers = iter(values)
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))


def test_single_question_and_token_counts(offline_client, capsys):
    calls, _ = offline_client
    assert ask.ask("LoRAとは？") == "LoRAは追加学習の手法です。"
    assert calls[0]["contents"] == "LoRAとは？"
    assert "入力: 12 / 出力: 8" in capsys.readouterr().out


def test_followup_includes_user_and_model_history(offline_client, monkeypatch):
    calls, _ = offline_client
    enter(monkeypatch, ["LoRAとは？", "具体例を教えて", "/exit"])
    ask.chat()
    assert len(calls) == 2
    history = calls[1]["contents"]
    assert [m.role for m in history] == ["user", "model", "user"]
    assert [m.parts[0].text for m in history] == [
        "LoRAとは？", "LoRAは追加学習の手法です。", "具体例を教えて"
    ]


def test_blank_input_is_not_sent_and_initial_question_is_kept(offline_client, monkeypatch):
    calls, _ = offline_client
    enter(monkeypatch, ["  ", "具体例は？", "/exit"])
    ask.chat("LoRAとは？")
    assert len(calls) == 2
    assert calls[1]["contents"][0].parts[0].text == "LoRAとは？"


def test_api_error_does_not_break_session_or_add_failed_turn(offline_client, monkeypatch, capsys):
    calls, results = offline_client
    results[:] = [reply(), errors.ClientError(429, {"error": {"message": "rate limit"}}), reply()]
    enter(monkeypatch, ["LoRAとは？", "失敗する質問", "再質問", "/exit"])
    ask.chat()
    assert "レート制限" in capsys.readouterr().out
    history = calls[2]["contents"]
    assert [m.parts[0].text for m in history] == [
        "LoRAとは？", "LoRAは追加学習の手法です。", "再質問"
    ]


@pytest.mark.parametrize("stop", [EOFError, KeyboardInterrupt])
def test_terminal_exit_is_clean(offline_client, monkeypatch, capsys, stop):
    calls, _ = offline_client
    monkeypatch.setattr("builtins.input", Mock(side_effect=stop))
    ask.chat()
    assert calls == []
    assert "会話を終了" in capsys.readouterr().out


def test_missing_usage_is_unknown(capsys):
    assert ask.response_text(reply(usage=False))
    assert "入力: 不明 / 出力: 不明" in capsys.readouterr().out


def test_empty_response_is_explained(capsys):
    assert "テキストの回答がありません" in ask.response_text(types.GenerateContentResponse())


def test_zero_tokens_are_not_unknown(capsys):
    response = reply()
    response.usage_metadata.prompt_token_count = 0
    response.usage_metadata.candidates_token_count = 0
    ask.response_text(response)
    assert "入力: 0 / 出力: 0" in capsys.readouterr().out


def test_default_cli_still_uses_course_question(monkeypatch, capsys):
    stub = Mock(return_value="回答")
    monkeypatch.setattr(ask, "ask", stub)
    monkeypatch.setattr("sys.argv", ["app.ask"])
    ask.main()
    stub.assert_called_once_with(ask.DEFAULT_QUESTION)


def test_chat_cli_passes_initial_question(monkeypatch):
    stub = Mock()
    monkeypatch.setattr(ask, "chat", stub)
    monkeypatch.setattr("sys.argv", ["app.ask", "--chat", "LoRAとは？"])
    ask.main()
    stub.assert_called_once_with("LoRAとは？")


def test_missing_api_key_has_setup_instructions(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="GEMINI_API_KEY が見つかりません"):
        ask.ask("LoRAとは？")
