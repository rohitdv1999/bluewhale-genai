.PHONY: install test lint eval api ui up

install:
	python -m venv .venv && .venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check src tests

eval:
	.venv/bin/python -m bluewhale.evals.run --gate

api:
	.venv/bin/uvicorn bluewhale.api.app:create_app --factory --reload

ui:
	streamlit run ui/streamlit_app.py

up:
	docker compose up --build
