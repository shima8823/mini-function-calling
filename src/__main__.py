#!/usr/bin/env python3

from llm_sdk import Small_LLM_Model

def main() -> None:
	try:
		model = Small_LLM_Model(
			model_name="Qwen/Qwen3-0.6B",
			device="cpu",
			dtype=None,
		)
		print(model.get_path_to_vocabulary_json())
		
	except Exception as e:
		print(f"❌ エラーが発生しました: {e}")
		return

if __name__ == "__main__":
	main()
