#!/usr/bin/env python3
"""
Claude APIトークンカウンター

使用例:
1. ファイルのトークン数をカウント:
   python count_tokens.py --file ./README.md

2. テキストのトークン数をカウント:
   python count_tokens.py --text "こんにちは、Claude!"

3. 複数ファイルのトークン数をカウント:
   python count_tokens.py --file ./file1.py --file ./file2.py

4. 会話形式でカウント:
   python count_tokens.py --conversation
"""

import os
import argparse
import sys
from typing import List, Dict, Any
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("Error: anthropic ライブラリがインストールされていません。")
    print("インストール: pip install anthropic")
    sys.exit(1)


class TokenCounter:
    def __init__(self):
        """APIキーを環境変数から取得してクライアントを初期化"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY 環境変数が設定されていません。")
            print("設定方法: export ANTHROPIC_API_KEY=your-api-key-here")
            sys.exit(1)

        self.client = anthropic.Anthropic(api_key=api_key)

        # モデルと料金設定
        self.models = {
            "claude-fable-5": {"input": 10.0, "output": 50.0},
            "claude-opus-4-8": {"input": 5.0, "output": 25.0},
            "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
            "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
        }

    def count_tokens(self, content: str, model: str = "claude-sonnet-4-6") -> Dict[str, Any]:
        """文字列のトークン数をカウント"""
        try:
            response = self.client.messages.count_tokens(
                model=model,
                messages=[{"role": "user", "content": content}]
            )
            return {
                "input_tokens": response.input_tokens,
                "model": model
            }
        except Exception as e:
            print(f"Error: トークンカウントに失敗しました: {e}")
            return None

    def count_file_tokens(self, file_path: str, model: str = "claude-sonnet-4-6") -> Dict[str, Any]:
        """ファイルのトークン数をカウント"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            result = self.count_tokens(content, model)
            if result:
                result["file_path"] = file_path
                result["file_size"] = len(content)
            return result
        except FileNotFoundError:
            print(f"Error: ファイル '{file_path}' が見つかりません。")
            return None
        except UnicodeDecodeError:
            print(f"Error: ファイル '{file_path}' をUTF-8で読み込めませんでした。")
            return None

    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """料金を計算（USD）"""
        if model not in self.models:
            print(f"Warning: モデル '{model}' の料金情報がありません。")
            return 0.0

        rates = self.models[model]
        input_cost = (input_tokens / 1_000_000) * rates["input"]
        output_cost = (output_tokens / 1_000_000) * rates["output"]
        return input_cost + output_cost

    def print_results(self, results: List[Dict[str, Any]], model: str):
        """結果を表示"""
        if not results or all(r is None for r in results):
            print("カウント可能な結果がありませんでした。")
            return

        total_tokens = 0
        total_size = 0

        print(f"\n=== トークンカウント結果 (モデル: {model}) ===")
        print("-" * 60)

        for result in results:
            if result is None:
                continue

            tokens = result["input_tokens"]
            total_tokens += tokens

            if "file_path" in result:
                size = result["file_size"]
                total_size += size
                print(f"ファイル: {result['file_path']}")
                print(f"  サイズ: {size:,} 文字")
                print(f"  トークン数: {tokens:,}")
            else:
                print(f"テキスト: {tokens:,} トークン")

        print("-" * 60)
        print(f"合計トークン数: {total_tokens:,}")
        if total_size > 0:
            print(f"合計文字数: {total_size:,}")

        # 料金計算
        input_cost = self.calculate_cost(total_tokens, 0, model)

        # 想定される出力トークン数での料金計算例
        estimated_outputs = [100, 500, 1000, 5000]
        print(f"\n=== 料金計算 ===")
        print(f"入力のみ: ${input_cost:.6f} (約{input_cost * 150:.2f}円)")

        print("\n想定出力トークン数での総料金:")
        for output_tokens in estimated_outputs:
            total_cost = self.calculate_cost(total_tokens, output_tokens, model)
            print(f"  出力{output_tokens:,}トークン: ${total_cost:.6f} (約{total_cost * 150:.2f}円)")


def interactive_mode():
    """対話モード"""
    counter = TokenCounter()

    print("=== Claude API トークンカウンター (対話モード) ===")
    print("終了するには 'quit' または 'exit' と入力してください。")
    print()

    while True:
        try:
            text = input("カウントしたいテキストを入力してください: ")
            if text.lower() in ['quit', 'exit', 'q']:
                break

            if not text.strip():
                continue

            result = counter.count_tokens(text)
            if result:
                tokens = result["input_tokens"]
                cost = counter.calculate_cost(tokens, 0, "claude-sonnet-4-6")
                print(f"トークン数: {tokens:,}")
                print(f"入力料金: ${cost:.6f} (約{cost * 150:.2f}円)")
                print()

        except KeyboardInterrupt:
            print("\n終了します。")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Claude APIを使用してトークン数をカウントします",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--file", "-f",
        action="append",
        help="カウントするファイルパス（複数指定可能）"
    )

    parser.add_argument(
        "--text", "-t",
        help="カウントするテキスト"
    )

    parser.add_argument(
        "--model", "-m",
        default="claude-sonnet-4-6",
        choices=["claude-fable-5", "claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"],
        help="使用するモデル（デフォルト: claude-sonnet-4-6）"
    )

    parser.add_argument(
        "--conversation", "-c",
        action="store_true",
        help="対話モードで実行"
    )

    args = parser.parse_args()

    # 対話モード
    if args.conversation:
        interactive_mode()
        return

    # 引数がない場合は使用方法を表示
    if not args.file and not args.text:
        parser.print_help()
        return

    counter = TokenCounter()
    results = []

    # ファイルのカウント
    if args.file:
        for file_path in args.file:
            result = counter.count_file_tokens(file_path, args.model)
            results.append(result)

    # テキストのカウント
    if args.text:
        result = counter.count_tokens(args.text, args.model)
        results.append(result)

    counter.print_results(results, args.model)


if __name__ == "__main__":
    main()