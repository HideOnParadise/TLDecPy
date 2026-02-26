PYTHON ?= python3
PIP := $(PYTHON) -m pip

.PHONY: install-dev lint typecheck test qa build

install-dev:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check tldecpy tests

typecheck:
	$(PYTHON) -m mypy tldecpy

test:
	$(PYTHON) -m pytest tests -q

qa: lint typecheck test

build:
	$(PYTHON) -m build
