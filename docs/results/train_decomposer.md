# Decomposer LoRA Training - Task 5.3

- **Base model:** `HuggingFaceTB/SmolLM2-135M-Instruct`
- **LoRA:** r=16, alpha=32, dropout=0.05, targets=['q_proj', 'v_proj']
- **Trainable:** 0.92M / 135.4M (0.68%)
- **Data:** 560 train / 70 val (prompt-disjoint splits from Task 1.2)
- **Schedule:** 5 epochs, batch 4 x grad-accum 4 (effective 16), lr 0.001, linear decay with 6% warmup
- **Wall clock:** 134.8 min on CPU (12 threads)
- **Val loss:** 2.1785 untrained -> **1.3185** best (epoch 5)

## Val loss by epoch

| Epoch | Val loss |
|---|---|
| 0 (untrained) | 2.1785 |
| 1 | 1.4705 |
| 2 | 1.3754 |
| 3 | 1.3374 |
| 4 | 1.3231 |
| 5 | 1.3185 **(best, saved)** |

## Train loss curve

| Step | Train loss | LR |
|---|---|---|
| 10 | 1.8534 | 1.00e-03 |
| 20 | 1.2807 | 9.39e-04 |
| 30 | 1.2123 | 8.79e-04 |
| 40 | 1.0559 | 8.18e-04 |
| 50 | 1.1242 | 7.58e-04 |
| 60 | 1.1669 | 6.97e-04 |
| 70 | 1.0491 | 6.36e-04 |
| 80 | 0.9773 | 5.76e-04 |
| 90 | 0.9875 | 5.15e-04 |
| 100 | 1.0582 | 4.55e-04 |
| 110 | 0.8688 | 3.94e-04 |
| 120 | 0.9978 | 3.33e-04 |
| 130 | 1.0176 | 2.73e-04 |
| 140 | 0.8870 | 2.12e-04 |
| 150 | 0.9040 | 1.52e-04 |
| 160 | 0.9266 | 9.09e-05 |
| 170 | 0.8716 | 3.03e-05 |
