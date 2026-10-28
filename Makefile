.PHONY: install run test lint format typecheck

install:
	python -m pip install -e '.[dev]'

run:
	uvicorn issue_triage.main:app --reload --app-dir src

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy src
