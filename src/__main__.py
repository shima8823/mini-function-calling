#!/usr/bin/env python3

from llm_sdk import Small_LLM_Model
from .inputs import FunctionDefinition, PromptItem, load_function_definitions, load_prompts
from .simple_bpe_tokenizer import SimpleBPETokenizer
import json
from pathlib import Path


def main() -> None:
	# 入力読み込み
	base_path = Path(__file__).parent.parent
	functions_path = base_path / "data" / "exercise_input" / "functions_definition.json"
	prompts_path = base_path / "data" / "exercise_input" / "function_calling_tests.json"
	results_dir = base_path / "results"
	results_dir.mkdir(exist_ok=True)

	function_definitions = load_function_definitions(functions_path)
	prompts = load_prompts(prompts_path)

	# TODO データの整形


	# モデル初期化
	model = Small_LLM_Model(
		model_name="Qwen/Qwen3-0.6B",
		device="cpu",
		dtype=None,
	)

	encoded = model._encode("Hello, how are you?")
	print(encoded)
	decoded = model._decode(encoded.tolist()[0])
	print(decoded)
	return

if __name__ == "__main__":
	main()
