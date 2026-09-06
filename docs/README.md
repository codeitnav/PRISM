# /docs

Project documentation.

- `api-contract.md` — shared API contract between client/server/ml (Task 0.2)
- `schema/` — frozen JSON Schema (draft 2020-12) for every object in the contract, plus worked examples under `schema/examples/`. Source of truth for `ml/app/schemas/*.py` (Pydantic) and `server/src/models/*.js` (Mongoose) — see `api-contract.md` for how the three stay in sync
- `results/` — benchmark, ablation, and calibration reports (Phase 8)
- `failure-analysis.md` — failure taxonomy (Task 8.3)
- `REPRODUCE.md` — reproducibility pack (Task 8.5)
- `label-quality.md` — weak label audit (Task 1.3)
