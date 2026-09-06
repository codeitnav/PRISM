# PRISM

**P**rompt **R**econstruction and **I**nference from **S**emantic **M**ultimodal representations

Reverse Prompt Engineering platform: given an AI-generated image (and, at reduced depth, text), reconstructs the most probable structured prompt behind it — subject, style, medium, modifiers, tone, constraints — with a calibrated confidence score and an interactive similarity graph against retrieved candidate prompts. See `docs/` for the full project design and technical approach.

## Architecture

```
client/   React + Vite        — upload UI, structured results, similarity graph
server/   Node + Express      — orchestration: uploads, persistence, routes to ml/
ml/       Python + FastAPI    — all ML logic: embeddings, FAISS retrieval, PEZ baseline,
                                 structured decomposition, confidence scoring
data/     datasets & derived artifacts (not committed — see .gitignore)
docs/     project design, API contract, evaluation reports
```

`server` never contains ML logic — it's a thin orchestration layer that calls `ml` over REST.
MongoDB is the system of record (uploads, results, metadata); FAISS (inside `ml`) handles vector similarity search.

## Prerequisites

- Docker + Docker Compose
- (for local, non-Docker dev only) Node.js 22+, Python 3.11+

## Getting started

```bash
make up      # copies .env.example -> .env for each service, builds images, starts mongo/ml/server/client
make ps      # check container status
make logs    # tail logs from all services
```

Once up, each service exposes a health check:

| Service | URL |
|---|---|
| client | http://localhost:5173 |
| server | http://localhost:4000/health |
| ml | http://localhost:8000/health |
| mongo | localhost:27017 |

```bash
make test    # run server (node --test) and ml (pytest) test suites
make seed    # populate MongoDB + FAISS index from /data (implemented in Task 2.3)
make down    # stop and remove containers
make clean   # also remove volumes and local .env files
```