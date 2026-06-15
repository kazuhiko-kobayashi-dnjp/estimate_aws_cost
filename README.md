# estimate_aws_cost

Amazon Bedrock / Claude のトークン数とコストを集計する CLI ツール。

**Claude Code の会話ログ (.jsonl) を読み込み、月次コストを Excel ファイルに出力する**のが主な用途。
会話ログに記録された `usage` フィールドを直読みするため、Bedrock API への接続不要で実行できる。

## スクリプト

| ファイル | 用途 | 認証 |
|---|---|---|
| `estimate_cost.py` | 月次 Excel 出力・セッション単位集計 | 通常は不要（usage 直読み）|
| `count_tokens.py`  | Anthropic API でトークン数を計測する簡易版 | `ANTHROPIC_API_KEY` |

## セットアップ

```bash
pip install -r requirements.txt
# Excel 出力に openpyxl が必要
pip install openpyxl
```

## 基本的な使い方: 月次 Excel 出力

```bash
# 当月の全セッションを集計して cost_YYYY-MM.xlsx を出力（カレントディレクトリ）
python3 estimate_cost.py --month

# 月を指定する場合
python3 estimate_cost.py --month 2026-05
```

出力ファイル `cost_YYYY-MM.xlsx` の構成:

| 行の種類 | 色 | 内容 |
|---|---|---|
| session | 白/薄灰 交互 | セッション 1 件。モデル名・課金レートも記録 |
| day_total | 薄黄・太線 | 日次合計 |
| week_total | 薄オレンジ・太線 | ISO 週次合計 |
| month_total | 赤・白文字・太線 | 月次合計 |

列: `type / label / date / iso_week / input_tokens / output_tokens / cache_write_tokens / cache_read_tokens / cost_usd / cost_jpy / unresolved_turns / project / model / input_rate / output_rate`

> **日付の判定について**: セッション開始日時は JSONL 内の最初のメッセージの `timestamp` を使用する。
> ファイルの更新日時 (mtime) は後から変わる場合があるため使用しない。

## その他のオプション

```bash
# 引数なし: カレントプロジェクトの最新の会話ログを集計して表示
python3 estimate_cost.py

# 会話ログ一覧（番号・日時・話題）を表示
python3 estimate_cost.py --list

# 番号 / パスでセッションを指定して集計
python3 estimate_cost.py --transcript 0          # 一覧の [0]（最新）
python3 estimate_cost.py --transcript 3          # 一覧の [3]
python3 estimate_cost.py --transcript /path/to/session.jsonl

# workspace 配下の全ログを一覧集計
python3 estimate_cost.py --all

# ファイル / テキストのトークン数とコスト（CountTokens API 使用・MFA 認証必要）
python3 estimate_cost.py --file foo.py --show-cost
python3 estimate_cost.py --text "こんにちは" --show-cost
```

### --transcript の出力例

```
=== セッション実コスト（会話ログ集計 / usage 直読み（実請求値）） ===
model_id : claude-sonnet-4-6
log      : xxxxxxxx-xxxx-xxxx.jsonl
------------------------------------------------------------------------
入力         ( 42 msgs):   123,456 tokens
出力         ( 63 msgs):    45,678 tokens
キャッシュ書込             :    12,345 tokens
キャッシュ読取             : 1,234,567 tokens
------------------------------------------------------------------------
合計トークン               : 1,416,046 tokens
入力コスト                 : $  0.370368  (rate=$3.0/MTok)
出力コスト                 : $  0.685170  (rate=$15.0/MTok)
キャッシュ書込コスト        : $  0.046294  (rate=$3.7500/MTok)
キャッシュ読取コスト        : $  0.037037  (rate=$0.3000/MTok)
========================================================================
  ★ 総額: $1.1388  (約 170.8 円)
========================================================================
```

## CountTokens API を使う場合（--file / --text）

MFA 必須の AWS 環境を想定し、既定プロファイルは `temp-mfa`。
`--profile` または環境変数 `AWS_PROFILE` で変更する。

> **boto3 のバージョンに注意**: `CountTokens` API には **boto3 1.40 以降**が必要。
> `'BedrockRuntime' object has no attribute 'count_tokens'` と出たら boto3 が古い。
> OS 同梱版は古いことが多いので仮想環境を推奨:
> ```bash
> python3 -m venv .venv && source .venv/bin/activate
> pip install -U 'boto3>=1.40' openpyxl
> ```

## 注意点（実請求との差）

- **usage 直読みモード**（`--transcript` / `--month`）: ログに記録された usage をそのまま集計するため実請求値と一致する
- **CountTokens モード**（`--file` / `--text`）: 入力トークン数の推定値。システムプロンプト・ツール定義はログに含まれないため実請求はやや高い場合がある
- レート（$/MTok）はスクリプト内の定数（`DEFAULT_RATES` / `MODEL_ALIASES`）。実際のレートと異なる場合は `--input-rate` / `--output-rate` で上書き可能

## count_tokens.py（簡易版）

```bash
export ANTHROPIC_API_KEY=your-api-key
python3 count_tokens.py --file ./README.md --model claude-sonnet-4-6
python3 count_tokens.py --text "こんにちは"
python3 count_tokens.py --conversation   # 対話モード
```
