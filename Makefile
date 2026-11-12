IMG ?=
API ?= http://localhost:8000/api/v1/scan
HEALTH ?= http://localhost:8000/api/v1/health

# macOS + Homebrew: pyzbar finds libzbar through this path.
ifeq ($(shell uname -s),Darwin)
export DYLD_FALLBACK_LIBRARY_PATH := /opt/homebrew/lib:/usr/local/lib
endif

.PHONY: setup sync lock lock-check build up down logs health scan test test-slow test-docker lint fmt check-no-pip

setup:
	@[ -f .env ] || cp .env.example .env
	@mkdir -p dataset
	@echo "setup done (.env, ./dataset)"

sync:
	uv sync

lock:
	uv lock

lock-check:
	uv lock --check

build: setup
	docker compose build

up: setup
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

health:
	@curl -fsS $(HEALTH) | uv run --no-project python -m json.tool

scan:
	@test -n "$(IMG)" || (echo "usage: make scan IMG=path/to/image.jpg" && exit 1)
	@curl -sS -F "file=@$(IMG)" $(API) | uv run --no-project python -c 'import json, sys; r = json.load(sys.stdin); keys = ("request_id", "barcodes", "timing_ms", "dataset_queued"); print(json.dumps({k: r.get(k) for k in keys} if "barcodes" in r else r, indent=2))'

test:
	uv run pytest -q

test-slow:
	uv run pytest -q -m slow

test-docker:
	docker compose run --rm --no-deps -v ./tests:/app/tests api sh -c "uv sync --frozen && uv run pytest -q"

lint:
	uv run ruff check . && uv run ruff format --check .

fmt:
	uv run ruff format . && uv run ruff check --fix .

# Zero-pip policy: no requirements.txt and no installer invocations outside legacy/ and docs.
check-no-pip:
	@! test -e requirements.txt || (echo "requirements.txt must not exist" && exit 1)
	@! grep -rnIE '(^|[^[:alnum:]_-])pip3?([[:space:]]+install|[[:space:]]*$$)|python[0-9.]* -m pip|requirements\.txt' \
	   --exclude-dir=.venv --exclude-dir=legacy --exclude-dir=dataset --exclude-dir=.git \
	   --exclude-dir=.pytest_cache --exclude-dir=.ruff_cache \
	   --exclude=uv.lock --exclude=Makefile --exclude='*.md' . \
	   || (echo "pip usage found" && exit 1)
	@echo "zero-pip OK"
