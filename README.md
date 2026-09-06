# PRISM

**P**rompt **R**econstruction and **I**nference from **S**emantic **M**ultimodal representations

PRISM is a reverse prompt engineering platform. Given an AI-generated image (and, at reduced depth, text), it reconstructs the most probable structured prompt behind it: subject, style, medium, modifiers, tone, and constraints, along with a calibrated confidence score and an interactive similarity graph against retrieved candidate prompts.

See [`docs/`](docs/) for the full project design and technical approach, and [`docs/api-contract.md`](docs/api-contract.md) for the frozen API and data contract.

## Architecture

```
client/   React + Vite      upload UI, structured results, similarity graph
server/   Node + Express    orchestration: uploads, persistence, routes to ml/
ml/       Python + FastAPI  all ML logic: embeddings, FAISS retrieval, PEZ baseline,
                             structured decomposition, confidence scoring
data/     datasets and derived artifacts (not committed, see .gitignore)
docs/     project design, API contract, schema, evaluation reports
```

`server` never contains ML logic. It is a thin orchestration layer that calls `ml` over REST. MongoDB is the system of record (uploads, results, metadata), and FAISS (inside `ml`) handles vector similarity search.

---

## Quick start

New to the project? This gets the whole stack running in three commands.

```bash
git clone https://github.com/codeitnav/PRISM.git
cd PRISM
make up
```

Then check everything is healthy:

```bash
make ps
```

You should see four containers, all `Up`, with `mongo`, `ml`, and `server` showing `(healthy)`. Open **http://localhost:5173** in a browser, or hit the health endpoints directly:

```bash
curl http://localhost:4000/health   # server
curl http://localhost:8000/health   # ml
```

That's the whole setup. Everything below is reference material: environment variables, running a service outside Docker, and fixes for the handful of issues a new setup commonly hits.

---

## 1. Prerequisites

| Tool | Required for |
|---|---|
| Docker and Docker Compose | Everything. This is the only hard requirement. |
| Node.js 22+ | Running `client` or `server` outside Docker |
| Python 3.11 | Running `ml` outside Docker. Must be exactly 3.11, not whatever `python3` resolves to on your machine. See [Troubleshooting](#troubleshooting). |
| About 6GB of free disk | The `ml` service's dependencies (PyTorch, CLIP, and related packages) are large |

Check what you already have:

```bash
docker --version && docker compose version
node --version          # want v22 or newer
python3.11 --version    # want 3.11.x, see Troubleshooting if missing
```

---

## 2. Full setup walkthrough (Docker)

This is the path everyone should use day to day. Nothing needs to be installed locally except Docker.

### Step 1: Clone and start

```bash
git clone https://github.com/codeitnav/PRISM.git
cd PRISM
make up
```

`make up` does three things: it copies each service's `.env.example` to `.env` if one doesn't already exist (see [Configuration](#3-configuration-env-files) for what these contain), builds all four images, and starts them.

### Step 2: Verify the containers are healthy

```bash
make ps
```

Expected output: `prism-mongo`, `prism-ml`, `prism-server`, and `prism-client`, all `Up`, with `mongo`, `ml`, and `server` marked `(healthy)`.

### Step 3: Check each service responds

```bash
curl http://localhost:4000/health   # server
curl http://localhost:8000/health   # ml
curl -I http://localhost:5173       # client
```

Expected: `{"status":"ok","service":"server",...}` and `{"status":"ok","service":"ml",...}`, and a `200` from the client. Open **http://localhost:5173** in a browser to see the current (scaffold) UI.

### Step 4: Run the test suites

```bash
make test
```

Expected: `pass 6` for server and `6 passed` for ml. These include the contract tests that validate the shared JSON Schema against both the Pydantic and Mongoose models, so a pass here means the frontend, backend, and ML service all agree on the data shape.

### Step 5: Shut down when you're done

```bash
make down     # stops and removes containers, keeps the mongo volume and .env files
make clean    # also wipes the mongo volume and all .env files, for a fresh start
```

Day to day, that's the entire workflow: `make up`, work, `make down`. Continue reading only if you need to configure something or run a service outside Docker.

---

## 3. Configuration (`.env` files)

Each service has its own `.env`, generated from its `.env.example` by `make up`. You only need to edit these to change a port, point at an external database, or similar. The defaults work as-is for local Docker use.

### `server/.env`

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `4000` | Port the Express server listens on |
| `NODE_ENV` | `development` | |
| `MONGO_URI` | `mongodb://mongo:27017/prism` | `mongo` is the Docker service name and only resolves inside the Docker network. This value is overridden by `docker-compose.yml` when running via `make up`; edit it here only for standalone (non-Docker) runs. |
| `ML_SERVICE_URL` | `http://ml:8000` | Same Docker-network note as above |

### `ml/.env`

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `8000` | Port uvicorn listens on |
| `ENV` | `development` | |
| `MONGO_URI` | `mongodb://mongo:27017/prism` | Same Docker-network note as `server/.env` |

### `client/.env`

| Variable | Default | Meaning |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:4000` | Where the frontend sends API requests |

### Root `.env`

Sets `COMPOSE_PROJECT_NAME` only. Rarely needs editing.

### Using an external database

If you're not using Docker's local Mongo (for example, pointing at MongoDB Atlas instead), update `MONGO_URI` in both `server/.env` and `ml/.env`, and remove the `mongo` service, its `MONGO_URI` overrides, and the `depends_on: mongo` entries from `docker-compose.yml`. This isn't the default setup, so check with a teammate first if you're not sure.

---

## 4. Running a single service outside Docker

Useful for faster iteration on one service without rebuilding its image. Each service needs `mongo` reachable: either leave it running with `docker compose up mongo` in another terminal, or point `MONGO_URI` at wherever it actually lives.

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
npm test         # node --test, expect 6 passing
npm run dev      # http://localhost:4000, restarts automatically on file changes
```

### ml

```bash
cd ml
python3.11 -m venv .venv    # must be 3.11 specifically, see Troubleshooting
source .venv/bin/activate
```

Next, install dependencies. First check whether this machine has an NVIDIA GPU:

```bash
nvidia-smi
```

No output, or `command not found`, means no GPU. In that case, install the CPU build first so pip doesn't silently pull several gigabytes of unused CUDA packages:

```bash
pip install torch==2.5.1 torchvision==0.20.1 open_clip_torch==2.29.0 \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple
pip install -r requirements.txt
```

If this machine does have an NVIDIA GPU, just run:

```bash
pip install -r requirements.txt
```

Then:

```bash
pytest                          # expect 6 passed
python scripts/check_env.py     # confirms CUDA/CPU status and that CLIP ViT-L/14 loads and runs
uvicorn app.main:app --reload   # http://localhost:8000
```

Always run `check_env.py` after setting up `ml` on a new machine. It reports whether GPU-dependent phases of the project (LoRA fine-tuning, CLIP contrastive fine-tuning, Stable Diffusion regeneration) are feasible here, or need to be planned around a different machine.

---

## 5. Common commands

```bash
make up       # build and start all services (copies .env files first if missing)
make down     # stop and remove containers
make restart  # down, then up
make build    # rebuild images without starting them
make ps       # container status
make logs     # tail logs from all services (Ctrl+C stops tailing, doesn't stop containers)
make test     # run server (node --test) and ml (pytest) suites, inside containers
make seed     # populate MongoDB and the FAISS index from /data (not yet implemented)
make clean    # down -v (wipes the mongo volume) plus removes all .env files
```

Every target is a thin wrapper. See the `Makefile` itself for the equivalent raw `docker compose` command behind each one.

---

## Troubleshooting

**Port already in use** (`address already in use` on `make up`). Something else is bound to that port, most often a standalone `npm run dev` or `uvicorn` process left running from local development. Find and stop it:

```bash
sudo ss -ltnp | grep :4000     # or :8000, :5173, :27017
kill <pid>
```

**`pip install` fails with a PyO3/maturin build error mentioning Python 3.14.** Your system's default `python3` is too new for some ML packages (`pydantic-core` and others use PyO3, which lags behind the newest CPython release). Install Python 3.11 specifically and use it for the virtual environment:

```bash
sudo dnf install python3.11   # or your distro's equivalent
python3.11 -m venv .venv      # not python3 -m venv
```

**`ml` install pulls in gigabytes of `nvidia-*` packages on a machine with no GPU.** This is confirmed PyPI behavior: the default `torch` wheel bundles the full CUDA toolkit regardless of hardware, and installing a torch-dependent package (such as `open_clip_torch`) in a separate `pip install` call from `torch` itself lets the resolver silently swap in a CUDA build. Always install `torch`, `torchvision`, and `open_clip_torch` together, from the CPU index. See the exact command in [section 4](#4-running-a-single-service-outside-docker), or the comment block at the top of `ml/requirements.txt`.

**Contract tests fail after editing a schema.** `docs/schema/*.schema.json` is the source of truth. If you change one, also update the matching Pydantic model (`ml/app/schemas/`) and Mongoose schema (`server/src/models/`) by hand, then re-run `make test`. See [`docs/api-contract.md`](docs/api-contract.md) for the full sync process.

---

## Status

Scaffold and contract stage. All four services boot and expose `/health`. The `Reconstruction`, `Graph`, and `Candidate` API contract is frozen and validated by tests on both `server` and `ml`. The `ml` environment has been checked and currently runs CPU only (see `ml/scripts/check_env.py`). No retrieval, reconstruction, or ML pipeline is wired up yet. See `docs/` for the phased implementation plan.
