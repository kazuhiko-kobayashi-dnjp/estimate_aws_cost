#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Amazon Bedrock / Claude 用の無料トークンカウンター。
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError, ProfileNotFound
except ImportError:
    print("Error: boto3 がインストールされていません。\nインストール: pip install boto3", file=sys.stderr)
    sys.exit(1)

DEFAULT_RATES = {
    "anthropic.claude-sonnet-4-20250514-v1:0": {"input": 3.0, "output": 15.0},
    "anthropic.claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "anthropic.claude-opus-4-20250514-v1:0": {"input": 5.0, "output": 25.0},
    "anthropic.claude-opus-4-6-v1": {"input": 5.0, "output": 25.0},
    "anthropic.claude-haiku-4-5-20251001-v1:0": {"input": 1.0, "output": 5.0},
    "anthropic.claude-3-7-sonnet-20250219-v1:0": {"input": 3.0, "output": 15.0},
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {"input": 3.0, "output": 15.0},
    "anthropic.claude-3-5-haiku-20241022-v1:0": {"input": 0.8, "output": 4.0},
}

# 会話ログに記録される model 名（Anthropic API 形式）を、
# CountTokens で実際に使える Bedrock ID とレートへ対応づける。
# CountTokens 非対応の新モデルは、同系列の対応 ID でトークン数を近似計測する。
MODEL_ALIASES = {
    "claude-opus-4-8":   {"count_id": "anthropic.claude-opus-4-6-v1",            "input": 5.0,  "output": 25.0},
    "claude-opus-4-6":   {"count_id": "anthropic.claude-opus-4-6-v1",            "input": 5.0,  "output": 25.0},
    "claude-opus-4":     {"count_id": "anthropic.claude-opus-4-20250514-v1:0",   "input": 5.0,  "output": 25.0},
    "claude-sonnet-4-6": {"count_id": "anthropic.claude-sonnet-4-6",             "input": 3.0,  "output": 15.0},
    "claude-sonnet-4":   {"count_id": "anthropic.claude-sonnet-4-20250514-v1:0", "input": 3.0,  "output": 15.0},
    "claude-haiku-4-5":  {"count_id": "anthropic.claude-haiku-4-5-20251001-v1:0","input": 1.0,  "output": 5.0},
    "claude-fable-5":    {"count_id": "anthropic.claude-opus-4-6-v1",            "input": 10.0, "output": 50.0},
}

DEFAULT_PROFILE = "temp-mfa"


def resolve_model_for_log(log_model: str) -> Tuple[str, float | None, float | None]:
    """ログの model 名を (CountTokens 用 Bedrock ID, 入力レート, 出力レート) に解決する。

    完全一致 → 前方一致（バージョン差を吸収）の順で MODEL_ALIASES を引く。
    未知なら入力をそのまま count_id とし、レートは None（後段で DEFAULT_RATES を試す）。
    """
    if log_model in MODEL_ALIASES:
        a = MODEL_ALIASES[log_model]
        return a["count_id"], a["input"], a["output"]
    for prefix, a in MODEL_ALIASES.items():
        if log_model.startswith(prefix):
            return a["count_id"], a["input"], a["output"]
    # <synthetic> 等、実モデルでない名前は計測用IDに使えないので未解決(None)扱い
    if not log_model.startswith("claude") and not log_model.startswith("anthropic."):
        return None, None, None
    return log_model, None, None


def to_content_blocks(text: str) -> List[Dict[str, str]]:
    return [{"text": text}]


def build_converse_input_from_text(text: str, system: str | None = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {"messages": [{"role": "user", "content": to_content_blocks(text)}]}
    if system:
        body["system"] = [{"text": system}]
    return body

ALLOWED_TEXT_BLOCK_TYPES = {"text"}


def sanitize_content_blocks(content: Any) -> List[Dict[str, str]]:
    if isinstance(content, str):
        s = content.strip()
        return [{"text": s}] if s else []
    blocks: List[Dict[str, str]] = []
    if isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            s = content["text"].strip()
            if s:
                blocks.append({"text": s})
        return blocks
    if isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    blocks.append({"text": s})
                continue
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in ALLOWED_TEXT_BLOCK_TYPES and isinstance(item.get("text"), str):
                s = item["text"].strip()
                if s:
                    blocks.append({"text": s})
            elif item_type is None and isinstance(item.get("text"), str):
                s = item["text"].strip()
                if s:
                    blocks.append({"text": s})
        return blocks
    return blocks

VALID_ROLES = {"user", "assistant", "system"}


def build_converse_input_from_conversation_json(raw: Any, system: str | None = None) -> Dict[str, Any]:
    if isinstance(raw, dict) and "messages" in raw:
        messages_raw = raw.get("messages", [])
        system_raw = raw.get("system")
    elif isinstance(raw, list):
        messages_raw = raw
        system_raw = None
    else:
        raise ValueError("conversation JSON は list か {messages: [...]} 形式で指定してください。")
    messages: List[Dict[str, Any]] = []
    for m in messages_raw:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in VALID_ROLES or role == "system":
            continue
        blocks = sanitize_content_blocks(m.get("content"))
        if blocks:
            messages.append({"role": role, "content": blocks})
    body: Dict[str, Any] = {"messages": messages}
    resolved_system = system
    if resolved_system is None and system_raw is not None:
        if isinstance(system_raw, str):
            resolved_system = system_raw
        elif isinstance(system_raw, list):
            texts = [b.get("text", "") for b in sanitize_content_blocks(system_raw) if b.get("text")]
            if texts:
                resolved_system = "\n\n".join(texts)
    if resolved_system:
        body["system"] = [{"text": resolved_system}]
    return body


def flatten_message_content(content: Any) -> str:
    """会話 1 メッセージの content を、トークン計測用に 1 本のテキストへ簡約する。

    text / thinking はそのまま、tool_use は input JSON を、tool_result は
    内部の text ブロック（または文字列）を連結する。画像など非テキストは無視。
    """
    if isinstance(content, str):
        return content.strip()
    parts: List[str] = []
    if isinstance(content, list):
        for b in content:
            if isinstance(b, str):
                if b.strip():
                    parts.append(b)
                continue
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text" and isinstance(b.get("text"), str):
                parts.append(b["text"])
            elif t == "thinking" and isinstance(b.get("thinking"), str):
                parts.append(b["thinking"])
            elif t == "tool_use":
                parts.append(json.dumps(b.get("input", {}), ensure_ascii=False))
            elif t == "tool_result":
                c = b.get("content")
                if isinstance(c, str):
                    parts.append(c)
                elif isinstance(c, list):
                    for x in c:
                        if isinstance(x, dict) and isinstance(x.get("text"), str):
                            parts.append(x["text"])
                        elif isinstance(x, str):
                            parts.append(x)
    return "\n".join(p for p in parts if p and p.strip()).strip()


def slug_for_cwd(cwd: Path) -> str:
    """カレントディレクトリを Claude Code のプロジェクトログ用スラッグに変換する。"""
    return str(cwd.resolve()).replace("/", "-")


def list_transcripts() -> List[Path]:
    """カレントプロジェクトの会話ログ (.jsonl) を mtime 降順で返す（[0] が最新）。"""
    base = Path.home() / ".claude" / "projects" / slug_for_cwd(Path.cwd())
    if not base.is_dir():
        return []
    return sorted(base.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def find_latest_transcript() -> Path | None:
    """カレントプロジェクトの最新の会話ログ (.jsonl) を返す。見つからなければ None。"""
    logs = list_transcripts()
    return logs[0] if logs else None


def first_user_utterance(path: Path, limit: int = 44) -> str:
    """会話ログ先頭の「本物の」user 発言を要約用に short text で返す。

    スラッシュコマンド・IDE 通知・コマンド出力など、タグだけで中身が空の
    メッセージはスキップし、タグ除去後に文字が残る最初の発話を採用する。
    """
    import re
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(o, dict) or o.get("type") != "user":
                    continue
                msg = o.get("message")
                if not isinstance(msg, dict):
                    continue
                text = flatten_message_content(msg.get("content"))
                text = re.sub(r"<[^>]+>.*?</[^>]+>", " ", text, flags=re.DOTALL)  # 対タグ除去
                text = re.sub(r"<[^>]+>", " ", text)                              # 残った単独タグ
                text = re.sub(r"(?im)^Caveat:.*$", " ", text)                     # コマンド注意書き
                text = " ".join(text.split())
                if text:  # タグ除去後に中身が残る最初の発話＝本物の話題
                    return text[:limit] + ("…" if len(text) > limit else "")
    except OSError:
        pass
    return "(発話なし)"


def resolve_transcript_arg(arg: str) -> Path | None:
    """--transcript の引数を Path に解決する。

    "@latest" → 最新、数字 → 一覧の N 番目、それ以外 → パスとして扱う。
    """
    if arg == "@latest":
        return find_latest_transcript()
    if arg.isdigit():
        logs = list_transcripts()
        idx = int(arg)
        return logs[idx] if 0 <= idx < len(logs) else None
    p = Path(arg)
    return p if p.exists() else None


def print_transcript_list() -> int:
    """カレントプロジェクトの会話ログ一覧を表示する。"""
    import datetime
    logs = list_transcripts()
    if not logs:
        print("カレントプロジェクトの会話ログが見つかりません。", file=sys.stderr)
        return 2
    print(f"\n=== 会話ログ一覧（{len(logs)} 件 / [0] が最新） ===")
    print("-" * 78)
    for i, p in enumerate(logs):
        st = p.stat()
        dt = datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
        kb = st.st_size / 1024
        topic = first_user_utterance(p)
        tag = "  ← 最新" if i == 0 else ""
        print(f"[{i}] {dt}  {kb:7.0f}KB  {topic}{tag}")
    print("-" * 78)
    print("指定例: python3 estimate_cost.py --transcript 0   （番号 or パスで指定）")
    return 0


def build_converse_input_from_transcript(path: Path, system: str | None = None) -> Tuple[Dict[str, Any], int, bool]:
    """Claude Code の JSONL 会話ログを Converse 形式へ変換する。

    戻り値は (converse_input, メッセージ数, 末尾ダミー user を追加したか) のタプル。
    Converse の制約に合わせ、連続する同一 role はマージし、先頭の assistant は除去、
    末尾が assistant の場合はダミー user を 1 件追加する（count_tokens が user 終端を要求するため）。
    """
    raw_msgs: List[Dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(o, dict) or o.get("type") not in ("user", "assistant"):
                continue
            msg = o.get("message")
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            text = flatten_message_content(msg.get("content"))
            if text:
                raw_msgs.append({"role": role, "text": text})
    # 連続する同一 role をマージ
    merged: List[Dict[str, str]] = []
    for m in raw_msgs:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["text"] += "\n" + m["text"]
        else:
            merged.append(dict(m))
    # 先頭の assistant は除去（会話は user で始まる必要がある）
    while merged and merged[0]["role"] == "assistant":
        merged.pop(0)
    # 末尾が assistant なら、ダミー user を追加（count_tokens は user 終端を要求）
    appended_dummy = False
    if merged and merged[-1]["role"] == "assistant":
        merged.append({"role": "user", "text": "."})
        appended_dummy = True
    messages = [{"role": m["role"], "content": [{"text": m["text"]}]} for m in merged]
    body: Dict[str, Any] = {"messages": messages}
    if system:
        body["system"] = [{"text": system}]
    return body, len(messages), appended_dummy


def split_io_texts_from_transcript(path: Path) -> Tuple[str, str, int, int]:
    """会話ログを「入力側」「出力側」のテキストに分離する。

    入力側 = user ロール（あなたの発言・ツール結果）、
    出力側 = assistant ロール（AI が既に生成した返答・思考・ツール呼び出し）。
    戻り値は (入力テキスト, 出力テキスト, 入力メッセージ数, 出力メッセージ数)。
    """
    input_parts: List[str] = []
    output_parts: List[str] = []
    n_in = n_out = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(o, dict) or o.get("type") not in ("user", "assistant"):
                continue
            msg = o.get("message")
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            text = flatten_message_content(msg.get("content"))
            if not text:
                continue
            if role == "user":
                input_parts.append(text)
                n_in += 1
            elif role == "assistant":
                output_parts.append(text)
                n_out += 1
    return "\n".join(input_parts), "\n".join(output_parts), n_in, n_out


def detect_model_from_transcript(path: Path) -> str | None:
    """会話ログに記録された最後の assistant の model 名を返す。なければ None。

    セッション内でモデルは固定のはずだが、念のため最後に使われたものを採る。
    """
    found: str | None = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(o, dict):
                continue
            msg = o.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("model"), str) and msg["model"]:
                found = msg["model"]
    return found


class BedrockTokenCounter:
    def __init__(self, profile: str | None, region: str):
        try:
            session = boto3.Session(profile_name=profile, region_name=region) if profile else boto3.Session(region_name=region)
        except ProfileNotFound as e:
            raise SystemExit(f"AWS profile が見つかりません: {e}")
        self.client = session.client("bedrock-runtime", region_name=region)

    def count_converse(self, model_id: str, converse_input: Dict[str, Any]) -> int:
        if not hasattr(self.client, "count_tokens"):
            import botocore
            raise SystemExit(
                "Error: お使いの boto3/botocore は CountTokens API に未対応です。\n"
                f"  現在の botocore: {botocore.__version__}（CountTokens には 1.40 以降が必要）\n"
                "  対処: ライブラリを更新してください。\n"
                "    pip install -U 'boto3>=1.40'\n"
                "  ※ OS 同梱版(/usr/lib/python3/dist-packages)を使っている場合は、\n"
                "     仮想環境(venv)を作って pip で新しい boto3 を入れるのが安全です。"
            )
        try:
            resp = self.client.count_tokens(modelId=model_id, input={"converse": converse_input})
            return int(resp["inputTokens"])
        except ClientError as e:
            msg = str(e)
            if "doesn't support counting tokens" in msg or "model identifier is invalid" in msg:
                supported = ", ".join(sorted(DEFAULT_RATES))
                raise SystemExit(
                    f"Error: model-id '{model_id}' は Bedrock の CountTokens に対応していません。\n"
                    f"  Bedrock 形式のID（例: anthropic.claude-opus-4-6-v1）を --model-id に指定してください。\n"
                    f"  CountTokens 対応の内蔵ID: {supported}"
                ) from e
            if "AccessDenied" in e.response.get("Error", {}).get("Code", "") or "explicit deny" in msg or "MFA" in msg:
                raise SystemExit(
                    "Error: bedrock:CountTokens へのアクセスが拒否されました（AccessDenied）。\n"
                    "  MFA 必須ポリシーにより、MFA 未認証のクレデンシャルが明示的に拒否されている可能性があります。\n"
                    "  対処: MFA 済みプロファイルを --profile で指定するか、AWS_PROFILE に設定してください。\n"
                    "        例) python3 estimate_cost.py --transcript --profile temp-mfa --show-cost\n"
                    "  一時クレデンシャルが期限切れの場合は MFA セッションを取り直してください。"
                ) from e
            raise SystemExit(f"Error: CountTokens 呼び出しに失敗しました: {msg}") from e
        except BotoCoreError as e:
            raise SystemExit(f"Error: AWS 接続に失敗しました: {e}") from e


def read_text_file(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp932", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"{path} を読み込めませんでした")


def usd_for_tokens(tokens: int, usd_per_mtok: float) -> float:
    return (tokens / 1_000_000.0) * usd_per_mtok


def resolve_rates(model_id: str, input_rate: float | None, output_rate: float | None) -> Tuple[float | None, float | None]:
    if input_rate is not None or output_rate is not None:
        return input_rate, output_rate
    if model_id in DEFAULT_RATES:
        r = DEFAULT_RATES[model_id]
        return r["input"], r["output"]
    return None, None


def print_transcript_summary(path: Path, model_id: str, in_tokens: int, out_tokens: int,
                             n_in: int, n_out: int,
                             input_rate: float | None, output_rate: float | None) -> None:
    """会話ログの実コスト（既存の入力＋既存の出力）を表示する。"""
    JPY = 150.0
    total_tokens = in_tokens + out_tokens
    print("\n=== セッション実コスト（会話ログ集計） ===")
    print(f"model_id : {model_id}")
    print(f"log      : {path.name}")
    print("-" * 72)
    print(f"入力 (user/tool, {n_in:>3} msgs): {in_tokens:>9,} tokens")
    print(f"出力 (assistant, {n_out:>3} msgs): {out_tokens:>9,} tokens")
    print("-" * 72)
    print(f"合計トークン                : {total_tokens:>9,} tokens")
    if input_rate is None or output_rate is None:
        print("\nコスト: レート未設定。--input-rate / --output-rate を指定するか、内蔵モデルIDを使用してください。")
        return
    in_cost = usd_for_tokens(in_tokens, input_rate)
    out_cost = usd_for_tokens(out_tokens, output_rate)
    total_cost = in_cost + out_cost
    print(f"入力コスト                  : ${in_cost:>10.6f}  (rate=${input_rate}/MTok)")
    print(f"出力コスト                  : ${out_cost:>10.6f}  (rate=${output_rate}/MTok)")
    line = "=" * 72
    print(f"\n{line}")
    print(f"  ★ 総額: ${total_cost:.4f}  (約 {total_cost * JPY:,.1f} 円)")
    print(line)


def print_summary(rows: List[Dict[str, Any]], model_id: str, input_rate: float | None, output_rate: float | None) -> None:
    if not rows:
        print("結果がありません。")
        return
    total_tokens = sum(int(r["tokens"]) for r in rows)
    print("\n=== CountTokens 結果 ===")
    print(f"model_id : {model_id}")
    print("-" * 72)
    for r in rows:
        label = r["label"]
        chars = r.get("chars")
        desc = f"  文字数 {chars:,}" if chars is not None else ""
        print(f"{label}: {r['tokens']:,} tokens{desc}")
    print("-" * 72)
    print(f"合計     : {total_tokens:,} tokens")
    if input_rate is None:
        print("入力概算 : レート未設定。--input-rate / --output-rate を指定するか、内蔵モデルIDを使用してください。")
        return
    in_cost = usd_for_tokens(total_tokens, input_rate)
    print(f"入力概算 : ${in_cost:.6f}  (rate=${input_rate}/MTok)")
    if output_rate is not None:
        for out_tokens in (100, 500, 1000, 5000):
            total_cost = in_cost + usd_for_tokens(out_tokens, output_rate)
            print(f"  出力 {out_tokens:>5,} tokens 想定総額: ${total_cost:.6f}")
    # ── 一目でわかる総額（代表値: 出力1,000トークン想定 / 円換算 150円/USD） ──
    JPY = 150.0
    if output_rate is not None:
        headline = in_cost + usd_for_tokens(1000, output_rate)
        head_label = "総額（入力＋出力1,000tok想定）"
    else:
        headline = in_cost
        head_label = "総額（入力のみ）"
    line = "=" * 72
    print(f"\n{line}")
    print(f"  ★ {head_label}: ${headline:.4f}  (約 {headline * JPY:,.1f} 円)")
    print(line)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Amazon Bedrock CountTokens API で Claude の入力トークン数を無料カウントします。")
    p.add_argument("--file", "-f", action="append", help="カウント対象ファイル。複数指定可。")
    p.add_argument("--text", "-t", help="カウント対象テキスト。")
    p.add_argument("--stdin", action="store_true", help="標準入力から読む。")
    p.add_argument("--conversation-json", help="会話 JSON ファイルを読む。thinking/tool_use は text のみに簡約。")
    p.add_argument("--transcript", nargs="?", const="@latest", metavar="PATH|N",
                   help="Claude Code の会話ログ (.jsonl) を読んでトークン数を測る。"
                        "省略時は最新ログ。番号(--list の N)・パスでも指定可。")
    p.add_argument("--list", action="store_true", help="カレントプロジェクトの会話ログ一覧（番号・日時・話題）を表示する。")
    p.add_argument("--system", help="system prompt を追加。")
    p.add_argument("--model-id", "-m", default=None, help="Bedrock modelId。未指定かつ --transcript 時はログから自動検出。例: anthropic.claude-opus-4-6-v1")
    p.add_argument("--profile", default=os.environ.get("AWS_PROFILE", DEFAULT_PROFILE), help=f"AWS profile 名。未指定時は AWS_PROFILE か '{DEFAULT_PROFILE}'。")
    p.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"), help="AWS region。未指定時は AWS_REGION か us-east-1。")
    p.add_argument("--show-cost", action="store_true", help="概算コストを表示。")
    p.add_argument("--input-rate", type=float, help="入力レート (USD / 1M tokens)。")
    p.add_argument("--output-rate", type=float, help="出力レート (USD / 1M tokens)。")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        return print_transcript_list()
    # 入力が一つも指定されなければ、カレントプロジェクトの最新ログを集計する（既定動作）。
    if not args.file and not args.text and not args.stdin and not args.conversation_json and not args.transcript:
        args.transcript = "@latest"
    counter = BedrockTokenCounter(profile=args.profile, region=args.region)
    rows: List[Dict[str, Any]] = []
    # transcript 以外の入力では従来通り Sonnet を既定 model-id とする。
    fallback_model = args.model_id or "anthropic.claude-sonnet-4-20250514-v1:0"
    if args.text:
        body = build_converse_input_from_text(args.text, system=args.system)
        tokens = counter.count_converse(fallback_model, body)
        rows.append({"label": "text", "tokens": tokens, "chars": len(args.text)})
    if args.stdin:
        text = sys.stdin.read()
        body = build_converse_input_from_text(text, system=args.system)
        tokens = counter.count_converse(fallback_model, body)
        rows.append({"label": "stdin", "tokens": tokens, "chars": len(text)})
    if args.file:
        for fp in args.file:
            path = Path(fp)
            if not path.exists():
                print(f"警告: ファイルが見つかりません: {fp}", file=sys.stderr)
                continue
            text = read_text_file(path)
            body = build_converse_input_from_text(text, system=args.system)
            tokens = counter.count_converse(fallback_model, body)
            rows.append({"label": str(path), "tokens": tokens, "chars": len(text)})
    if args.conversation_json:
        path = Path(args.conversation_json)
        if not path.exists():
            print(f"Error: 会話 JSON ファイルが見つかりません: {path}", file=sys.stderr)
            return 2
        raw = json.loads(read_text_file(path))
        body = build_converse_input_from_conversation_json(raw, system=args.system)
        tokens = counter.count_converse(fallback_model, body)
        rows.append({"label": f"conversation:{path}", "tokens": tokens})
    if args.transcript:
        path = resolve_transcript_arg(args.transcript)
        if path is None:
            print(f"Error: 会話ログを解決できません: '{args.transcript}'。--list で一覧を確認してください。", file=sys.stderr)
            return 2
        in_text, out_text, n_in, n_out = split_io_texts_from_transcript(path)
        if n_in == 0 and n_out == 0:
            print(f"警告: 会話ログにテキストメッセージがありません: {path}", file=sys.stderr)
            return 0
        # モデルの決定: --model-id 明示 > ログから自動検出 > Sonnet 既定
        SONNET = "anthropic.claude-sonnet-4-20250514-v1:0"
        log_model = detect_model_from_transcript(path)
        if args.model_id:
            count_id, in_rate, out_rate = args.model_id, None, None
            display_model = args.model_id
        elif log_model:
            count_id, in_rate, out_rate = resolve_model_for_log(log_model)
            if count_id is None:
                # <synthetic> 等、計測に使えないモデル名 → Sonnet で代用計測
                count_id, display_model = SONNET, f"{log_model} (計測: {SONNET})"
            else:
                display_model = f"{log_model} (計測: {count_id})" if count_id != log_model else log_model
        else:
            count_id, in_rate, out_rate = SONNET, None, None
            display_model = count_id
        # --input-rate / --output-rate / DEFAULT_RATES の順でレートを上書き解決
        r_in, r_out = resolve_rates(count_id, args.input_rate, args.output_rate)
        in_rate = args.input_rate if args.input_rate is not None else (in_rate if in_rate is not None else r_in)
        out_rate = args.output_rate if args.output_rate is not None else (out_rate if out_rate is not None else r_out)
        in_tokens = counter.count_converse(count_id, build_converse_input_from_text(in_text)) if in_text else 0
        out_tokens = counter.count_converse(count_id, build_converse_input_from_text(out_text)) if out_text else 0
        print_transcript_summary(path, display_model, in_tokens, out_tokens, n_in, n_out, in_rate, out_rate)
        return 0
    input_rate, output_rate = (None, None)
    if args.show_cost:
        input_rate, output_rate = resolve_rates(fallback_model, args.input_rate, args.output_rate)
    print_summary(rows, fallback_model, input_rate, output_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
