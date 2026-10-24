"""第1回: LLMへの単発質問と、履歴を保持する対話。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # /exit または Ctrl-D で終了
    uv run python -m app.ask --chat "LoRAとは何ですか？"

単発質問は会話履歴を持たない。--chat は同じセッション内の履歴を保持する。
ツール検索はどちらのモードでも行わない。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def response_text(response: types.GenerateContentResponse) -> str:
    """利用量が欠けた応答でも、0と誤認せず表示する。"""
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage else None
    output_tokens = usage.candidates_token_count if usage else None
    print(
        f"[tokens] 入力: {input_tokens if input_tokens is not None else '不明'} / "
        f"出力: {output_tokens if output_tokens is not None else '不明'}"
    )
    return response.text or "[回答] テキストの回答がありませんでした。"


def ask(question: str) -> str:
    with genai.Client(api_key=get_api_key()) as client:
        response = client.models.generate_content(model=MODEL, contents=question)
    return response_text(response)


def chat(initial_question: str | None = None) -> None:
    """SDKの同じチャットを再利用し、過去の発言と回答を次の質問へ渡す。"""
    print("会話モード：/exit または Ctrl-D で終了。履歴は終了すると消えます。")
    with genai.Client(api_key=get_api_key()) as client:
        session = client.chats.create(model=MODEL)
        question = initial_question
        while True:
            try:
                if question is None:
                    question = input("あなた> ")
                question = question.strip()
                if question == "/exit":
                    break
                if not question:
                    question = None
                    continue
                print(f"[質問] {question}")
                response = session.send_message(question)
                print(response_text(response))
            except (EOFError, KeyboardInterrupt):
                print("\n会話を終了しました。")
                break
            except errors.APIError as e:
                print(api_error_message(e.code, e.message))
            question = None


def main() -> None:
    parser = argparse.ArgumentParser(description="LLMに質問する（--chat で会話履歴を保持）")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--chat", action="store_true", help="会話履歴を保持して続けて質問する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    if args.chat:
        chat(args.question)
        return
    question = args.question if args.question is not None else DEFAULT_QUESTION
    print(f"[質問] {question}\n")
    try:
        print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
