.PHONY: up down build restart logs ps seed test clean init-env

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

## Populate MongoDB / FAISS index from /data (not yet implemented)
seed: init-env
	$(COMPOSE) run --rm ml python -m app.scripts.seed

## Run test suites for server and ml
test:
	$(COMPOSE) run --rm server npm test
	$(COMPOSE) run --rm ml pytest

## Remove containers, volumes, and local env files
clean:
	$(COMPOSE) down -v
	rm -f .env client/.env server/.env ml/.env
