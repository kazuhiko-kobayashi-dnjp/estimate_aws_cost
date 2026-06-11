# estimate_aws_cost

Amazon Bedrock / Claude のトークン数とコストを見積もる CLI ツール。

特に **Claude Code の会話ログ (.jsonl) を読み込み、セッションで実際に発生した
入力・出力トークン数と概算コストを集計**できる。Bedrock の `CountTokens` API は
無料なので、推論を実行せずにトークン数を計測できる。

## スクリプト

| ファイル | 経路 | 認証 |
|---|---|---|
| `estimate_cost.py` | AWS Bedrock (`CountTokens`) | AWS profile / credentials |
| `count_tokens.py`  | Anthropic API 直 | `ANTHROPIC_API_KEY` |

メインは `estimate_cost.py`。`count_tokens.py` は Anthropic API を直接使う簡易版。

## セットアップ

```bash
pip install -r requirements.txt
```

`estimate_cost.py` は MFA 必須の AWS 環境を想定し、既定プロファイルは `temp-mfa`。
環境に合わせて `--profile` か環境変数 `AWS_PROFILE` で変更する。

> **boto3 のバージョンに注意**: `CountTokens` API は比較的新しいため、**boto3 1.40 以降**が必要。
> `'BedrockRuntime' object has no attribute 'count_tokens'` と出たら boto3 が古い。
> 特に OS 同梱版（`/usr/lib/python3/dist-packages`）は古いことが多いので、
> 仮想環境を作って新しい boto3 を入れるとよい:
> ```bash
> python3 -m venv .venv && source .venv/bin/activate
> pip install -U 'boto3>=1.40'
> ```

## 使い方（estimate_cost.py）

```bash
# 引数なし: カレントプロジェクトの最新の会話ログを自動集計
#   - モデルはログから自動検出、プロファイルは AWS_PROFILE か temp-mfa
python3 estimate_cost.py

# 会話ログ一覧（番号・日時・話題）を表示
python3 estimate_cost.py --list

# 番号 / パスでセッションを指定して集計
python3 estimate_cost.py --transcript 0          # 一覧の [0]（最新）
python3 estimate_cost.py --transcript 3          # 一覧の [3]
python3 estimate_cost.py --transcript /path/to/session.jsonl

# ファイル / テキストのトークン数とコスト
python3 estimate_cost.py --file foo.py --show-cost
python3 estimate_cost.py --text "こんにちは" --show-cost
```

### 出力例

```
=== セッション実コスト（会話ログ集計） ===
model_id : claude-opus-4-8 (計測: anthropic.claude-opus-4-6-v1)
log      : 43213343-....jsonl
------------------------------------------------------------------------
入力 (user/tool,  69 msgs):    22,098 tokens
出力 (assistant, 105 msgs):    27,168 tokens
------------------------------------------------------------------------
合計トークン                :    49,266 tokens
入力コスト                  : $  0.110490  (rate=$5.0/MTok)
出力コスト                  : $  0.679200  (rate=$25.0/MTok)
========================================================================
  ★ 総額: $0.7897  (約 118.5 円)
========================================================================
```

## 注意点（実請求との差）

会話ログ本文だけを計測するため、実際の API 課金とは次の点でズレる目安値である。

- **システムプロンプト・ツール定義**はログに含まれないため未計上（実請求はやや高い）
- **プロンプトキャッシュ**の割引は未考慮（実請求はやや安い可能性）
- レート（$/MTok）はスクリプト内の手書き定数（`DEFAULT_RATES` / `MODEL_ALIASES`）
- ログのモデル名（例 `claude-opus-4-8`）が `CountTokens` 非対応の場合、同系列の
  対応 ID でトークン数を近似計測する（表示に「計測: ...」と明示）

## 使い方（count_tokens.py）

```bash
export ANTHROPIC_API_KEY=your-api-key
python3 count_tokens.py --file ./README.md --model claude-opus-4-8
python3 count_tokens.py --text "こんにちは"
python3 count_tokens.py --conversation   # 対話モード
```
