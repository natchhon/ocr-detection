PYTHON := .venv/bin/python
PROVIDER ?= chakkhu
HOST ?= localhost
PORT ?= 8000

.PHONY: run dev install test
run:
	$(PYTHON) -m uvicorn app.main:app --host $(HOST) --port $(PORT) --workers 1

dev:
	$(PYTHON) -m uvicorn app.main:app --host $(HOST) --port $(PORT) --reload --reload-dir app

install:
	$(PYTHON) -m pip install -r requirements/$(subst _,-,$(PROVIDER)).txt

test:
	$(PYTHON) -m pytest tests/ -v

.PHONY: install-dev download-model check format benchmark evaluate

install-dev:
	$(PYTHON) -m pip install -r requirements/dev.txt

download-model:
	$(PYTHON) -m scripts.download_model --provider $(PROVIDER)

check:
	$(PYTHON) -m compileall -q app scripts
	$(PYTHON) -c 'from app.main import app; from app.bootstrap import create_service'
	$(PYTHON) -m ruff check app scripts tests
	$(PYTHON) -m ruff format --check app scripts tests

format:
	$(PYTHON) -m ruff check --fix app scripts tests
	$(PYTHON) -m ruff format app scripts tests

benchmark:
	$(PYTHON) -m scripts.benchmark --provider $(PROVIDER) --output artifacts/benchmark.json $(ARGS)

evaluate:
	$(PYTHON) -m scripts.evaluate_accuracy --provider $(PROVIDER) --output artifacts/evaluation.json $(ARGS)

.PHONY: run-photo run-ais recognize-ais

run-photo:
	OCR_PROVIDER=chakkhu IMAGE_PROCESSOR=original OCR_BEAM=4 CARD_DETECTOR=none TEXT_DETECTOR=photo DOCUMENT_TEMPLATE= $(MAKE) run

run-ais:
	$(MAKE) run-photo

recognize-ais:
	$(PYTHON) -m scripts.test_recognition samples/ais_id_card.jpg --output private-data/ais-result.json
