.PHONY: up down build restart logs ps seed test clean init-env ingest split bench-embed build-index pez-baseline cliptag-baseline eval ingest-alpaca split-alpaca weak-label audit-labels

COMPOSE := docker compose

init-env:
	@[ -f .env ] || cp .env.example .env
	@[ -f client/.env ] || cp client/.env.example client/.env
	@[ -f server/.env ] || cp server/.env.example server/.env
	@[ -f ml/.env ] || cp ml/.env.example ml/.env

## Build and start all services (mongo, ml, server, client) in the background
up: init-env
	$(COMPOSE) up --build -d

## Stop and remove all containers
down:
	$(COMPOSE) down

## Rebuild images without starting
build: init-env
	$(COMPOSE) build

## Restart all services
restart: down up

## Tail logs from all services
logs:
	$(COMPOSE) logs -f

## Show status of all services
ps:
	$(COMPOSE) ps

## Task 2.3: populate MongoDB's reference_prompts collection from /data
seed: init-env
	$(COMPOSE) run --rm ml python -m scripts.seed_mongo

## Task 1.1: download + curate the DiffusionDB subset into /data/diffusiondb
ingest: init-env
	$(COMPOSE) run --rm ml python scripts/ingest_diffusiondb.py

## Task 1.2: build prompt-disjoint train/val/test splits into /data/splits
split: init-env
	$(COMPOSE) run --rm ml python scripts/split_dataset.py

## Task 2.1: log embedding throughput on a sample of ingested data
bench-embed: init-env
	$(COMPOSE) run --rm ml python -m scripts.bench_embed

## Task 2.2: embed the train split and build the FAISS retrieval index
build-index: init-env
	$(COMPOSE) run --rm ml python -m scripts.build_faiss_index

## Task 4.1: run the PEZ baseline on 20 test images, logs mean CLIP-score
pez-baseline: init-env
	$(COMPOSE) run --rm ml python -m scripts.run_pez_baseline

## Task 4.2: run the naive CLIP-tag baseline on the same 20 test images
cliptag-baseline: init-env
	$(COMPOSE) run --rm ml python -m scripts.run_cliptag_baseline

## Task 4.3: score both baselines through the shared evaluation harness
eval: init-env
	$(COMPOSE) run --rm ml python -m scripts.run_eval

## Task 1.3: apply weak structured labels; writes pairs_labeled.parquet and
## fills structured_fields on the seeded reference_prompts documents
weak-label: init-env
	$(COMPOSE) run --rm ml python -m scripts.run_weak_label

## Task 1.3: emit the 100-sample label audit worksheet (then fill in verdicts
## and re-run with --score to produce docs/label-quality.md)
audit-labels: init-env
	$(COMPOSE) run --rm ml python -m scripts.audit_weak_labels --emit

## Task 1.4: download + curate the Alpaca text subset into /data/alpaca
ingest-alpaca: init-env
	$(COMPOSE) run --rm ml python scripts/ingest_alpaca.py

## Task 1.4: build prompt-disjoint train/val/test splits for the Alpaca pairs
split-alpaca: init-env
	$(COMPOSE) run --rm ml python -m scripts.split_alpaca

## Run test suites for server and ml
test:
	$(COMPOSE) run --rm server npm test
	$(COMPOSE) run --rm ml pytest

## Remove containers, volumes, and local env files
clean:
	$(COMPOSE) down -v
	rm -f .env client/.env server/.env ml/.env
