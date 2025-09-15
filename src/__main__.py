#!/usr/bin/env python3

from llm_sdk import Small_LLM_Model
from .inputs import FunctionDefinition, PromptItem, load_function_definitions, load_prompts
import json
from pathlib import Path


def main() -> None:
	# 入力データ
	base_path = Path(__file__).parent.parent
	functions_path = base_path / "input" / "functions_definition.json"
	prompts_path = base_path / "input" / "function_calling_tests.json"
	results_dir = base_path / "output"
	results_dir.mkdir(exist_ok=True)

	function_definitions = load_function_definitions(functions_path)
	prompts = load_prompts(prompts_path)

	# 関数定義を文字列に整形
	def format_fn(defn: FunctionDefinition) -> str:
		args_spec = ", ".join(f"{name}: {defn.args_types.get(name, 'any')}" for name in defn.args_names)
		return f"{defn.fn_name}({args_spec}) -> {defn.return_type}"

	functions_catalog = "\n".join(format_fn(d) for d in function_definitions)

	# モデルを初期化
	model = Small_LLM_Model(
		model_name="Qwen/Qwen3-0.6B",
		device="cpu",
		dtype=None,
	)

	# 各プロンプトを処理して関数呼び出し用のJSONを生成
	results: list[dict] = []
	max_new_tokens = 256

	for item in prompts:
		print(f"\n処理中のプロンプト: {item.prompt}")

		# プロンプトを構築（関数一覧 + 質問 + JSON出力指示）
		# ロールプレイプロンプティング
		user_prompt = (
			"role: user \n"
			"You are a function selector. Read the available functions and the question, then respond with exactly ONE JSON object.\n"
			"- The output MUST be valid JSON with exactly two top-level keys: fn_name (string) and args (object).\n"
			"- args must ALWAYS be a JSON object (even for a single argument). Use key-value pairs with the exact argument names.\n"
			"Available functions:\n" + functions_catalog + "\n\n" +
			"Question:\n" + item.prompt + "\n\n"
		)
		assistant_prompt = (
			"role: assistant \n" +
			"json content: "
		)
		prompt_text = user_prompt + assistant_prompt

		print("生成されたプロンプト:")
		print(prompt_text)

		# 内部APIでエンコード
		encoded = model._encode(prompt_text)
		input_ids = encoded.tolist()[0]
		start_len = len(input_ids)

		print("\nトークン生成中...")

		json_start_flag = False
		resource = 0
		next_start = start_len

		# 貪欲法による生成
		for i in range(max_new_tokens):
			logits = model.get_logits_from_input_ids(input_ids)
			# argmax を取得（numpyは未使用）
			next_id = max(range(len(logits)), key=lambda i: logits[i])
			input_ids.append(int(next_id))
			# 単純な終了条件: '}' が出現
			# テキストを生成して逐次検索。json_start_flag が真かつ resource({}) が 0 になったら停止
			# TODO: O(n) なので手法を改善する
			generated_text = model._decode(input_ids[next_start:])
			next_start += 1
			print("generated_text", generated_text)
			for char in generated_text:
				if char == "{":
					json_start_flag = True
					resource += 1
				elif char == "}":
					resource -= 1
			if json_start_flag and resource == 0:
				print(f"生成完了 ({i+1} トークン)")
				break

		# デコードしてJSONを抽出
		generated_text = model._decode(input_ids[start_len:])
		print("\n生成されたテキスト:")
		print(generated_text)

		# JSONをパース
		json_obj: dict
		try:
			start = generated_text.find("{")
			end = generated_text.rfind("}")
			payload = generated_text[start : end + 1] if start != -1 and end != -1 and end > start else "{}"
			json_obj = json.loads(payload)
			print("\n抽出されたJSON:")
			print(json.dumps(json_obj, ensure_ascii=False, indent=2))
		except Exception as e:
			print(f"\nJSONパース失敗: {e}")
			json_obj = {"fn_name": "", "args": {}}

		# args が配列になる場合があるためここで検証
		fn_name = json_obj.get("fn_name", "")
		args = json_obj.get("args", {})
		if not isinstance(args, dict):
			args = {}

		results.append({
			"prompt": item.prompt,
			"fn_name": fn_name,
			"args": args,
		})

	print("\n全プロンプトの処理が完了しました")

	# 出力を保存
	output_path = results_dir / "function_calling_name.json"
	with output_path.open("w", encoding="utf-8") as f:
		json.dump(results, f, ensure_ascii=False, indent=2)
	print(f"\n結果を保存しました: {output_path}")

	return

if __name__ == "__main__":
	main()
