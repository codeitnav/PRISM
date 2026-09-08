# Captioning Results - Task 5.1

- **Model:** `Salesforce/blip-image-captioning-large`
- **Split(s):** test (70 images)
- **Wall clock:** 583.2s total, 8.33s/image (includes cache hits; a fully cached re-run is near-instant)
- **Caption length (words):** min 9, max 19, mean 13.8
- **Empty captions:** 0

## Model choice — why not BLIP-2 or LLaVA

The roadmap asks for BLIP-2, or LLaVA-1.5-7B "if VRAM allows". Neither fits this machine.
The arithmetic, from each checkpoint's own config rather than from memory:

| Model | Params | fp32 | bf16 | Fits in 6.3 GB available? |
|---|---|---|---|---|
| `blip2-opt-2.7b` | ~3.74B (OPT-2.7b 2560×32 + EVA ViT-g 1408×39 + Q-Former) | ~15.0 GB | ~7.5 GB | No |
| `llava-1.5-7b-hf` | ~7B | ~28.0 GB | ~14.0 GB | No |
| **`blip-image-captioning-large`** | **~470M** | **~1.9 GB** | — | **Yes** |

This box is CPU-only (per `ml/scripts/check_env.py`) with ~6.3 GB RAM actually free, so
BLIP-2 does not fit even at bf16 — and torch's CPU bf16 inference would be too slow to use
if it did. So the default is BLIP-1 large: the direct predecessor, same Salesforce BLIP
captioning lineage, trained for exactly this task. The quality cost is real — captions are
shorter and less compositional than BLIP-2's — and is visible in the table below.

**This is a configuration limit, not a code limit.** `CAPTION_MODEL` selects any BLIP or
BLIP-2 checkpoint and `app.caption._model_class()` picks the architecture from its config,
so on a GPU machine the roadmap's intended model runs with no code change:

```bash
CAPTION_MODEL=Salesforce/blip2-opt-2.7b docker compose run --rm ml python -m scripts.run_captioning
```

The cache key includes the model name, so switching checkpoints cannot serve stale captions.

## Caption boilerplate — a measured artifact

BLIP's decoder opens **55 of 70 captions (79%)** with a contentless phrase:

| Opener | Count | Share |
|---|---|---|
| `there is` / `there are` | 48 | 69% |
| `this is` | 4 | 6% |
| `an image of` / `a picture of` | 3 | 4% |

As an input feature for Task 5.2 that prefix is pure noise the decomposer would have to
learn to ignore, so `app.caption.strip_caption_boilerplate()` removes it — verified to
leave **0 of 70** captions with a residual opener.

It deliberately does **not** strip `photo of`, `photograph of`, `painting of`,
`screenshot of`, or `3d rendering of`. Those name the medium, which is real visual evidence
and one of the very fields being reconstructed; dropping them would destroy signal rather
than noise. `"black and white photo of a shark"` and `"painting of two women playing
tennis"` pass through untouched.

The stripper is **not** applied inside `caption_images()`. The cache and
`captions.parquet` hold the raw model output, so the stored artifact stays a faithful
record of what the model actually said; callers building model inputs opt in.

## Caption vs. true prompt

The caption describes what is *in* the image; the true prompt is what generated it. The gap between the two columns is exactly the gap the decomposition model (Task 5.3) has to close - the caption grounds the subject, but carries almost none of the style/medium/quality vocabulary.

| id | Caption | True prompt |
|---|---|---|
| `000051` | there is a poster with a picture of an orange on it | 5 0 s minimalist poster of an orange |
| `000073` | screenshot of a screenshot of a game being played in a video game | protoss cannon rush |
| `000080` | there are two ships that are in the ocean with a lot of waves | highly detailed stunning images of a frozen black pearl pirate ship in a frozen vibrant psychedelic ocean, half surface ranging through a st |
| `000096` | there is a vase with a picture of a giraffe and a horse on it | god, paleolithic cave art |
| `000099` | there is a 3d rendering of a creepy looking mask with sharp teeth | highly detailed render of a man wearing a skull fox mask, vray render, unreal engine, highly detailed faces, thin body, |
| `000107` | there is a painting on the wall of a cave depicting a man and a deer | hunting, paleolithic cave painting |
| `000111` | an image of a game with a bunch of characters on the screen | screenshot from heroes 3 |
| `000117` | there is a painting of a wolf on the side of a rock | zen, wolf, chauvet cave |
| `000120` | this is a picture of a carved wolf head on a black background | wolf, paleolithic figurine |
| `000130` | there is a blue door and a planter on the side of a building | painting of a european blue door surrunded by bougainvillea award winning fuzzy |
| `000140` | black and white photo of a great white shark with its mouth open | shark attack caught on camera, found footage video, analog horror |
| `000152` | there is a statue of a man in armor holding two swords | humanoid armored tiger. |
| `000154` | painting of two women playing tennis with a tennis ball and racket | collage cutout painting of a young adult women playing tennis, white rabbit, light palette, illumination, grass, tennis shoes, blue backgrou |
| `000165` | there is a dolphin that is jumping out of the water | a dolphin in full swat gear, zeiss lens, 5 0 mm |
| `000183` | spider - man in a black suit with a yellow belt on his chest | venom as a good hero, hyper detailed masterpiece, digital art painting, hyper realism aesthetic |
