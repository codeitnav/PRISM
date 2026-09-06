# PRISM

**P**rompt **R**econstruction and **I**nference from **S**emantic **M**ultimodal representations

Reverse Prompt Engineering platform: given an AI-generated image (and, at reduced depth, text), reconstructs the most probable structured prompt behind it — subject, style, medium, modifiers, tone, constraints — with a calibrated confidence score and an interactive similarity graph against retrieved candidate prompts. See [`docs/`](docs/) for the full project design and technical approach, and [`docs/api-contract.md`](docs/api-contract.md) for the frozen API/data contract.

## Architecture

```
client/   React + Vite        — upload UI, structured results, similarity graph
server/   Node + Express      — orchestration: uploads, persistence, routes to ml/
ml/       Python + FastAPI    — all ML logic: embeddings, FAISS retrieval, PEZ baseline,
                                 structured decomposition, confidence scoring
data/     datasets & derived artifacts (not committed — see .gitignore)
docs/     project design, API contract, schema, evaluation reports
```

`server` never contains ML logic — it's a thin orchestration layer that calls `ml` over REST. MongoDB is the system of record (uploads, results, metadata); FAISS (inside `ml`) handles vector similarity search.

---

## 1. Prerequisites

| Tool | Needed for |
|---|---|
| **Docker + Docker Compose** | The recommended way to run everything — required either way |
| **Node.js 22+** | Only if you'll run `client`/`server` outside Docker |
| **Python 3.11** | Only if you'll run `ml` outside Docker. Must be **exactly 3.11**, not whatever `python3` resolves to on your machine — see [Troubleshooting](#troubleshooting) |
| **~6GB free disk** | The `ml` service's dependencies (PyTorch + CLIP + friends) are large |

Check what you have:
```bash
docker --version && docker compose version
node --version      # want v22+
python3.11 --version  # want 3.11.x — if missing, see Troubleshooting
```

---

## 2. First-time setup (Docker — recommended)

This is the path everyone should use day-to-day; it needs nothing installed locally except Docker.

```bash
git clone https://github.com/codeitnav/PRISM.git
cd PRISM
make up
```

`make up` does three things: copies each service's `.env.example` → `.env` (only if missing — see [Configuration](#3-configuration-env-files) below for what's in them), builds all four images, and starts them.

**Verify it worked:**
```bash
make ps
```
You should see `prism-mongo`, `prism-ml`, `prism-server`, `prism-client` all `Up`, with mongo/ml/server showing `(healthy)`.

```bash
curl http://localhost:4000/health   # server
curl http://localhost:8000/health   # ml
curl -I http://localhost:5173       # client
```
Expected: `{"status":"ok","service":"server",...}` and `{"status":"ok","service":"ml",...}` respectively, and a `200` from client. Open **http://localhost:5173** in a browser to see the (currently default/scaffold) UI.

**Run the test suites:**
```bash
make test
```
Expect `pass 6` (server) and `6 passed` (ml) — these include the API contract tests that validate the shared JSON Schema against both the Pydantic and Mongoose models.

**Tear down when done:**
```bash
make down     # stop + remove containers, keep the mongo volume and .env files
make clean    # also wipe the mongo volume and all .env files (fresh start)
```

That's it for day-to-day use — skip to [Configuration](#3-configuration-env-files) or [Common commands](#5-common-commands) unless you specifically need to run a service outside Docker.

---

## 3. Configuration (`.env` files)

Each service has its own `.env`, generated from `.env.example` by `make up` (or `make init-env` alone). You only need to hand-edit these if you're changing a port, connecting to an external Mongo instance, or similar — the defaults work out of the box for local Docker use.

**`server/.env`**
| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `4000` | Port the Express server listens on |
| `NODE_ENV` | `development` | |
| `MONGO_URI` | `mongodb://mongo:27017/prism` | `mongo` here is the Docker service name — only resolves inside the Docker network. **Overridden by `docker-compose.yml`'s `environment:` block when run via `make up`** — edit this file only for standalone (non-Docker) runs |
| `ML_SERVICE_URL` | `http://ml:8000` | Same caveat as above |

**`ml/.env`**
| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `8000` | Port uvicorn listens on |
| `ENV` | `development` | |
| `MONGO_URI` | `mongodb://mongo:27017/prism` | Same Docker-network caveat as `server/.env` |

**`client/.env`**
| Variable | Default | Meaning |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:4000` | Where the frontend sends API requests |

**Root `.env`** just sets `COMPOSE_PROJECT_NAME` — rarely needs editing.

If you're not using Docker's local Mongo (e.g. pointing at MongoDB Atlas instead), update `MONGO_URI` in **both** `server/.env` and `ml/.env`, and remove the `mongo` service + the `MONGO_URI` overrides + the `depends_on: mongo` entries from `docker-compose.yml` — see git history / ask a teammate if this comes up, it's not the default setup.

---

## 4. Running a single service outside Docker

Useful for faster iteration on one service without rebuilding its image. Needs `mongo` reachable — either leave it running via `docker compose up mongo` in another terminal, or point `MONGO_URI` at wherever it actually is.

### client
```bash
cd client
npm install
npm run dev      # http://localhost:5173
npm run build    # production build check
```

### server
```bash
cd server
npm install
npm test         # node --test — expect 6 passing
npm run dev      # http://localhost:4000, auto-restarts on file change
```

### ml
```bash
cd ml
python3.11 -m venv .venv        # must be 3.11 specifically, see Troubleshooting
source .venv/bin/activate
```

Then install dependencies. **If this machine has no NVIDIA GPU** (check with `nvidia-smi` — no output/command-not-found means no GPU), install the CPU build first so pip doesn't silently pull ~5GB of unused CUDA packages:
```bash
pip install torch==2.5.1 torchvision==0.20.1 open_clip_torch==2.29.0 \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple
pip install -r requirements.txt
```
**If this machine has an NVIDIA GPU**, just run:
```bash
pip install -r requirements.txt
```

Then:
```bash
pytest                          # expect 6 passed
python scripts/check_env.py     # confirms CUDA/CPU status and that CLIP ViT-L/14 actually loads and runs
uvicorn app.main:app --reload   # http://localhost:8000
```

Always run `check_env.py` after setting up `ml` on a new machine — it tells you whether GPU-dependent phases of the project (LoRA fine-tuning, CLIP contrastive fine-tuning, Stable Diffusion regeneration) are feasible here, or need to be planned around a different machine.

---

## 5. Common commands

```bash
make up       # build + start all services (copies .env files first if missing)
make down     # stop + remove containers
make restart  # down then up
make build    # rebuild images without starting
make ps       # container status
make logs     # tail logs from all services (Ctrl+C to stop tailing, doesn't stop containers)
make test     # run server (node --test) and ml (pytest) suites, inside containers
make seed     # populate MongoDB + FAISS index from /data (not yet implemented)
make clean    # down -v (wipes mongo volume) + removes all .env files
```

Every target is a thin wrapper — see the `Makefile` itself if you want the equivalent raw `docker compose` command.

---

## Troubleshooting

**Port already in use** (`address already in use` on `make up`): something else is bound to that port — most often a standalone `npm run dev` / `uvicorn` process left running from local dev. Find it and kill it:
```bash
sudo ss -ltnp | grep :4000     # or :8000, :5173, :27017
kill <pid>
```

**`pip install` fails with a PyO3/maturin build error mentioning Python 3.14**: your system's default `python3` is too new for some ML packages (`pydantic-core` and others use PyO3, which lags behind the newest CPython). Install Python 3.11 specifically and use it for the venv:
```bash
sudo dnf install python3.11   # or your distro's equivalent
python3.11 -m venv .venv      # not `python3 -m venv`
```

**`ml` install pulls in gigabytes of `nvidia-*` packages on a machine with no GPU**: this is real, confirmed PyPI behavior — the default `torch` wheel bundles the full CUDA toolkit regardless of hardware, and installing a torch-dependent package (like `open_clip_torch`) in a *separate* `pip install` call from `torch` itself lets the resolver silently swap in a CUDA build. Always install `torch`+`torchvision`+`open_clip_torch` together, from the CPU index — see the exact command in [§4](#4-running-a-single-service-outside-docker) or the comment block at the top of `ml/requirements.txt`.

**Contract tests fail after editing a schema**: `docs/schema/*.schema.json` is the source of truth. If you change one, you must also update the matching Pydantic model (`ml/app/schemas/`) and Mongoose schema (`server/src/models/`) by hand, then re-run `make test` — see [`docs/api-contract.md`](docs/api-contract.md) for the full sync process.

---

## Status

Scaffold + contract stage: all four services boot and expose `/health`; the `Reconstruction`/`Graph`/`Candidate` API contract is frozen and validated by tests on both `server` and `ml`; the `ml` environment has been checked and currently runs CPU-only (see `ml/scripts/check_env.py` output). No retrieval, reconstruction, or ML pipeline is wired up yet — see `docs/` for the phased implementation plan.
