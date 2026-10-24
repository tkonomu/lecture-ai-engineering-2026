"""第1回の実API確認を行い、JSONログとanswer.mdの専用欄に記録する。"""

import html
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from google import genai
from google.genai import errors

from app.ask import DEFAULT_QUESTION, request
from app.config import MODEL, get_api_key

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- homework-check:start -->"
END = "<!-- homework-check:end -->"
CASES = [
    ("会話1：情報を伝える", "私の好きな色は青です。覚えておいてください。", "shared"),
    ("会話2：履歴を参照する", "私の好きな色は何でしたか？", "shared"),
    ("会話3：新しいセッション", "私の好きな色は何でしたか？", "fresh"),
    ("共通演習：LoRA", "LoRAとは何ですか？", "single"),
    ("共通演習：宿題の3レベル", DEFAULT_QUESTION, "single"),
]


def quote(text):
    # 応答中のHTMLや自動更新マーカーが文書構造に干渉しないようにする。
    return "\n".join("> " + html.escape(line, quote=False) for line in text.splitlines())


def render(run):
    lines = [
        "## 実APIの動作確認（自動記録）", "",
        f"- 実行日時：{run['started_at']}",
        f"- 使用モデル：{html.escape(run['model'])}",
        f"- 実行ID：{run['id']}",
        f"- 実行状態：{run['status']}",
        "- 以下は実際に受け取った応答です。API応答の取得と、回答内容の正しさは別に確認します。",
        "- 会話3はプロセス再起動ではなく、新しいチャットセッションで履歴をリセットした確認です。",
        "", "| 確認項目 | 状態 | 入力トークン | 出力トークン | 応答時間（秒） |",
        "|---|---|---:|---:|---:|",
    ]
    for row in run['results']:
        inp = row.get('input_tokens')
        out = row.get('output_tokens')
        elapsed = row.get('seconds')
        lines.append(f"| {row['name']} | {row['status']} | "
                     f"{inp if inp is not None else '不明'} | "
                     f"{out if out is not None else '不明'} | "
                     f"{format(elapsed, '.3f') if elapsed is not None else '-'} |")
    for row in run['results']:
        lines.extend(["", f"### {row['name']}", "", "質問：", quote(row['question']), "",
                      "回答／実行結果：", quote(row.get('response') or row.get('error') or '未実施')])
    lines.extend([
        "", "### 用語集の正しい定義（比較用）", "", quote(run['reference']), "",
        "### 結果を読んで確認すること", "",
        "- 会話2の回答が『青』だったか。会話3では情報不足だと回答したか。",
        "- 宿題の3レベルへの回答と、上の用語集の定義で一致した点・異なる点。",
        "- 会話履歴が増えたときの入力トークン数の変化。",
        "", "内容の正しさ・気づき：未記入（実行結果を確認して、自動記録欄の外に追記する）。",
    ])
    return "\n".join(lines)


def merge_answer(original, report):
    block = f"{START}\n{report}\n{END}"
    if START not in original and END not in original:
        return original.rstrip() + "\n\n" + block + "\n"
    if original.count(START) != 1 or original.count(END) != 1:
        raise ValueError("answer.mdの自動記録マーカーが重複または欠落しています。")
    before, rest = original.split(START, 1)
    previous, after = rest.split(END, 1)
    if END in before:
        raise ValueError("answer.mdの自動記録マーカーの順序が不正です。")
    return before + block + after


def save(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_checks(root=ROOT):
    key = get_api_key()  # 未設定なら、既存の記録を変えずに終了する。
    answer = root / "lectures/01/answer.md"
    merge_answer(answer.read_text(encoding="utf-8"), "")  # 実API呼び出し前に形式を確認
    glossary = json.loads((root / "data/glossary.json").read_text(encoding="utf-8"))
    reference = next(t['definition'] for t in glossary['terms'] if t['id'] == 'homework-levels')
    run = dict(id=uuid4().hex, started_at=datetime.now().astimezone().isoformat(),
               model=MODEL, status="実行中", reference=reference,
               results=[dict(name=n, question=q, status="未実施") for n, q, _ in CASES])
    log = root / "homework_runs" / f"{run['id']}.json"

    def checkpoint():
        # キーはメタデータに含めない。モデルが偶然返した場合も既知のキーを除去する。
        safe = json.loads(json.dumps(run, ensure_ascii=False).replace(key, "[REDACTED]"))
        save(log, json.dumps(safe, ensure_ascii=False, indent=2) + "\n")
        save(answer, merge_answer(answer.read_text(encoding="utf-8"), render(safe)))

    checkpoint()
    try:
        with genai.Client(api_key=key) as client:
            shared = client.chats.create(model=MODEL)
            for row, (_, question, mode) in zip(run['results'], CASES):
                session = shared if mode == 'shared' else (
                    client.chats.create(model=MODEL) if mode == 'fresh' else None)
                print(f"[実行] {row['name']}", flush=True)
                started = perf_counter()
                try:
                    response = request(question, client=client, session=session)
                    usage = response.usage_metadata
                    row.update(status="応答取得" if response.text else "テキスト応答なし",
                               response=response.text or "テキストの回答がありませんでした。",
                               input_tokens=usage.prompt_token_count if usage else None,
                               output_tokens=usage.candidates_token_count if usage else None)
                except errors.APIError as exc:
                    # 詳細文字列にはURL等が含まれ得るため、ステータスコードのみ保存する。
                    row.update(status="APIエラー", error=f"Gemini APIエラー（HTTP {exc.code}）")
                except KeyboardInterrupt:
                    row.update(status="中断", error="リクエスト中に中断。応答は取得できていません。")
                    raise
                except Exception as exc:
                    row.update(status="実行エラー", error=f"応答取得に失敗（{type(exc).__name__}）")
                    raise
                finally:
                    row['seconds'] = perf_counter() - started
                    checkpoint()
                if row['status'] == 'APIエラー':
                    # 認証・利用上限エラー時は追加呼び出しを行わない。
                    break
        run['status'] = ("完了（回答内容は要確認）" if all(
            r['status'] == '応答取得' for r in run['results']) else "未完了（エラーまたは応答なし）")
    except KeyboardInterrupt:
        run['status'] = "中断（取得済みの結果を保存）"
    except Exception as exc:
        run['status'] = f"中断（{type(exc).__name__}）"
        raise
    finally:
        checkpoint()
        print(f"記録：{log}\n回答ファイル：{answer}", flush=True)
    return run


def main():
    print("Geminiに最大5回質問し、answer.mdの自動記録欄を更新します。", flush=True)
    run = run_checks()
    if not run['status'].startswith("完了"):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
