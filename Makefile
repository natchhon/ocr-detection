PYTHON := .venv/bin/python
PROVIDER ?= chakkhu
HOST ?= 0.0.0.0
PORT ?= 8000

.PHONY: run dev install test

run:
	$(PYTHON) -m uvicorn app.main:app --host $(HOST) --port $(PORT) --workers 1

dev:
	$(PYTHON) -m uvicorn app.main:app --host $(HOST) --port $(PORT) --reload --reload-dir app

install:
	$(PYTHON) -m pip install -r requirements/$(PROVIDER).txt

test:
	$(PYTHON) -m pytest tests/ -v
