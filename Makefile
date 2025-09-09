install:
	uv sync

run:
	uv run python -m src

test:
	uv run python -m src.llm_connection_test

debug:
	uv run python -m src --debug

clean:
	rm -rf .venv
	rm -rf __pycache__
	rm -rf .pytest_cache

lint:
	uv run flake8 src/ tests/