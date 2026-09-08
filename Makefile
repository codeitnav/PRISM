.PHONY: up down build restart logs ps seed test clean init-env ingest split bench-embed build-index

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

## Run test suites for server and ml
test:
	$(COMPOSE) run --rm server npm test
	$(COMPOSE) run --rm ml pytest

## Remove containers, volumes, and local env files
clean:
	$(COMPOSE) down -v
	rm -f .env client/.env server/.env ml/.env
