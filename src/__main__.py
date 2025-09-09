#!/usr/bin/env python3

# from llm_sdk import Small_LLM_Model
from .inputs import FunctionDefinition, PromptItem, load_function_definitions, load_prompts


def main() -> None:

	function_definitions = load_function_definitions("data/exercise_input/functions_definition.json")
	prompts = load_prompts("data/exercise_input/function_calling_tests.json")

	print(function_definitions)
	print(prompts)

	return

if __name__ == "__main__":
	main()
