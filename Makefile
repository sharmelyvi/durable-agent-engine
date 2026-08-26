VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help setup verify test test-postgres pg-up pg-down lint types fmt demo bench clean

help:
	@echo "setup   install into .venv"
	@echo "verify  lint + types + full test suite + demo (what CI runs)"
	@echo "demo    crash a payment run and watch it recover"
	@echo "bench   measure what delta correction actually saves"
	@echo ""
	@echo "test-postgres  run the same suite against a real PostgreSQL"
	@echo "pg-up/pg-down  start or stop that PostgreSQL (port 55433)"

setup:
	python3 -m venv $(VENV)
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e ".[dev]"

verify: lint types test demo
	@echo ""
	@echo "  verified: lint clean, suite green, demo invariant held"

test:
	PYTHONPATH=src $(PY) -m pytest -q

# The suite is parametrised over both backends. Without a server the Postgres
# half skips; with one, every invariant is asserted twice.
pg-up:
	docker compose up -d --wait

pg-down:
	docker compose down -v

test-postgres: pg-up
	PYTHONPATH=src $(PY) -m pytest -q

lint:
	$(VENV)/bin/ruff check src tests scripts
	$(VENV)/bin/ruff format --check src tests scripts

types:
	PYTHONPATH=src $(VENV)/bin/mypy

fmt:
	$(VENV)/bin/ruff check --fix src tests scripts
	$(VENV)/bin/ruff format src tests scripts

demo:
	@PYTHONPATH=src $(PY) -m durable_agent.demo

bench:
	@PYTHONPATH=src $(PY) scripts/benchmark.py

clean:
	rm -rf .pytest_cache .ruff_cache bench-results.json
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
