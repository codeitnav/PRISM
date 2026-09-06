# /data

Datasets and derived artifacts live here (not committed to git — see `.gitignore`).

Expected layout (populated by later tasks):

```
data/
  diffusiondb/        # Task 1.1 - raw + curated image-prompt pairs
  text/                # Task 1.4 - Alpaca/Dolly (prompt, completion) pairs
  splits/              # Task 1.2 - train/val/test split id lists
```

Run `make seed` to populate MongoDB from these artifacts once Task 2.3 is implemented.
