.PHONY: install test test-all lint cov simulate experiments

install:
	pip install -e ".[dev]"

test:
	pytest -m "not slow"

test-all:
	pytest

lint:
	ruff check src tests

cov:
	pytest --cov=ashdpf --cov-report=term-missing

simulate:
	ashdpf simulate --events 2000

experiments:
	ashdpf experiments --events 10000 --out results
