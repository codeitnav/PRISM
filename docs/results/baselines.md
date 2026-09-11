# Reconstruction Method Evaluation

PEZ, the naive CLIP-tag baseline, and the LoRA decomposition model scored through the same harness (`ml/app/eval.py`) on the same 20 test-split images.

**Decomposer note:** its output is structured JSON (subject/style/medium/lighting/modifiers/tone/negative_constraints), flattened into a single string here so it can be scored on the same footing as the other two methods. Component-wise precision/recall/F1 against weak structured labels is a separate, complementary evaluation - see `docs/results/decomposer_eval.md`.

## Summary

| Reconstructor | Mean CLIP-score | Mean BERTScore F1 | Mean latency (s/image) |
|---|---|---|---|
| pez | 0.2964 | 0.7188 | 67.98 |
| clip_tag | 0.2355 | 0.7404 | 1.64 |
| decomposer | 0.1689 | 0.7199 | 61.58 |

**Latency caveat:** the decomposer's 61.58 s/image includes a one-time base-model download and load on the machine this was run on; that cost is amortized over only 20 images here. Navya's own run (model already cached) measured 3.34 s/row, which is the representative inference-time figure.

## Interpretation

The decomposer currently scores **lowest on CLIP-score** of the three methods, consistent with the low component-level F1 (0.1248) already reported in `docs/results/decomposer_eval.md`: at 560 training rows and 135M parameters, it learned the JSON output format and a plausible-sounding subject, but not a reliable mapping from evidence to the specific style/medium/lighting/modifier values that would make its output match the image more precisely.

Its **BERTScore F1 (0.7199) sits between PEZ and CLIP-tag**, which is a different and informative signal: unlike PEZ's word-salad output, the decomposer's flattened prompts are grammatical, readable sentences (e.g. `"a giant mythical leviathan flying across the ocean, digital art, trending on artstation"`), so they read more like a real prompt even where the specific details are wrong. This mirrors the PEZ vs. CLIP-tag trade-off noted below: image-similarity and text-readability are measuring different things, and no single method wins both here yet.

This is an expected, honestly-reported result for a first fine-tune on a small model/dataset, not a bug in the harness or the eval — see `docs/results/decomposer_eval.md` for the model's own detailed self-assessment and next-step recommendations (loss re-weighting, a bigger base model, higher LoRA rank).

## Per-image detail

| Reconstructor | Image ID | CLIP-score | BERTScore F1 | Generated prompt |
|---|---|---|---|---|
| pez | 000001 | 0.2686 | 0.7178 | amalexoplandrenated creeps creeps horrifying daredevil |
| pez | 000003 | 0.3014 | 0.7281 | viral fossilmeatsneaker monster tabletop slime ': |
| pez | 000005 | 0.2670 | 0.7294 | anirfuture future 🌀 exquisite exhilarating unnoticed zen |
| pez | 000006 | 0.2753 | 0.6549 | wesome homme *-* decal ❥ subliunlea― |
| pez | 000027 | 0.2591 | 0.7254 | forestation snowy stormdarkness coupon stocdelicious |
| pez | 000032 | 0.3607 | 0.7564 | london panorama explored pathways boroughperegrairspace urged |
| pez | 000034 | 0.3255 | 0.7359 | vehicle vehicles topia conceptart vanesssmog rejecgames |
| pez | 000049 | 0.3334 | 0.7122 | palestinians armenian chisabandon stalin poindictatinductees |
| pez | 000070 | 0.2909 | 0.7327 | saberzom💥 backpacks biomesocietal combat machines |
| pez | 000077 | 0.2948 | 0.6980 | tro fury porto calvert starwars roadsafety folklurus |
| pez | 000084 | 0.2836 | 0.7347 | gotham examines cinematic cinematic reboot inyourcontexcinemato |
| pez | 000088 | 0.3059 | 0.6996 | colonel aasnavis unveils aromatherapy psychedelic snuggle lsd |
| pez | 000089 | 0.3250 | 0.7188 | 【 assassinscreed inmate ryn ryu 😙 remastered clones |
| pez | 000103 | 0.2768 | 0.7447 | suspense explores powerful unleacards intelligcreepshabbat |
| pez | 000115 | 0.2627 | 0.6988 | mapp important toothpaste shops business woman adoptdont:/ |
| pez | 000128 | 0.3110 | 0.6850 | possible warped terror lancers bandits nya karate raccoon |
| pez | 000131 | 0.2688 | 0.6700 | how parade intilebron affectiondemocratic aloha happiness |
| pez | 000139 | 0.2883 | 0.7592 | rebellion indicted incarceration collaborators browsing fairly yale eliza |
| pez | 000144 | 0.2864 | 0.7352 | streetphotography portrait sidewalk cafes todayshow smartphones challengers slive |
| pez | 000158 | 0.3422 | 0.7399 | ว waterdescription dog threaten aquatic husky receive |
| clip_tag | 000001 | 0.2510 | 0.7546 | concept art, digital painting, cinematic lighting |
| clip_tag | 000003 | 0.2439 | 0.7552 | concept art, digital painting, volumetric lighting |
| clip_tag | 000005 | 0.3100 | 0.7777 | psychedelic, digital painting, volumetric lighting |
| clip_tag | 000006 | 0.2125 | 0.6679 | photorealistic, 3d render, high contrast lighting |
| clip_tag | 000027 | 0.1995 | 0.7044 | concept art, 3d render, cinematic lighting |
| clip_tag | 000032 | 0.2620 | 0.8383 | photorealistic, matte painting, volumetric lighting |
| clip_tag | 000034 | 0.2613 | 0.8034 | cyberpunk, digital painting, cinematic lighting |
| clip_tag | 000049 | 0.1564 | 0.7329 | vaporwave, photograph, harsh lighting |
| clip_tag | 000070 | 0.2402 | 0.7620 | concept art, matte painting, cinematic lighting |
| clip_tag | 000077 | 0.2477 | 0.7022 | surrealism, matte painting, cinematic lighting |
| clip_tag | 000084 | 0.3192 | 0.7991 | cyberpunk, matte painting, cinematic lighting |
| clip_tag | 000088 | 0.2374 | 0.7466 | psychedelic, matte painting, volumetric lighting |
| clip_tag | 000089 | 0.2263 | 0.7646 | photorealistic, 3d render, cinematic lighting |
| clip_tag | 000103 | 0.2499 | 0.7827 | surrealism, 3d render, cinematic lighting |
| clip_tag | 000115 | 0.1989 | 0.6959 | photorealistic, 3d render, ambient lighting |
| clip_tag | 000128 | 0.2054 | 0.7147 | concept art, 3d render, cinematic lighting |
| clip_tag | 000131 | 0.1547 | 0.6633 | psychedelic, photograph, harsh lighting |
| clip_tag | 000139 | 0.2310 | 0.7176 | surrealism, photograph, studio lighting |
| clip_tag | 000144 | 0.2863 | 0.7361 | cyberpunk, photograph, neon lighting |
| clip_tag | 000158 | 0.2172 | 0.6882 | photorealistic, photograph, golden hour lighting |
| decomposer | 000001 | 0.1644 | 0.7780 | retrofuturistic portrait of a man in a space suit, artstation, cinematic lighting, smooth transparent visor, close up, trending on artstation, intricate, 8 k |
| decomposer | 000003 | 0.2318 | 0.7640 | a giant mythical leviathan flying across the ocean, digital art, trending on artstation |
| decomposer | 000005 | 0.1650 | 0.7831 | retrofuturistic portrait of a 2 1 savage in astronaut helmet, artstation, smooth transparent visor, detailed space graphics in background, close up, 5 k, artgerm |
| decomposer | 000006 | 0.1313 | 0.7187 | a futuristic fine lasers tracing, art, oil on canvas, dark, art of a women in modern fashion, color ink painting, candy chip color, octane hyperrealism photorealistic airbrush collage painting, dark monochrome, trending on artnet |
| decomposer | 000027 | 0.0831 | 0.6838 | an oil painting of dwayne johnson dressed as all might from my hero academia by artgerm |
| decomposer | 000032 | 0.1694 | 0.7186 | batman the dark knight portrait insanely defined intricate low angle grim powerful mist dark environment city background on top of a gotham building moon dark clouds red highlights |
| decomposer | 000034 | 0.1872 | 0.6369 | a woman sitting next to a man |
| decomposer | 000049 | 0.0633 | 0.6951 | an abstract painting of a red frog with blue and green circles on it, abstract, oil on canvas, dark, realistic, photorealistic, photorealistic, photorealistic acrylic, octane hyperrealism photorealistic airbrush collage painting, dark monochrome, cinematic, 8ies eros |
| decomposer | 000070 | 0.1630 | 0.7117 | portrait of one! white anthropomorphic lion with helmet cyborg blue and gold |
| decomposer | 000077 | 0.0920 | 0.6984 | joe biden standing ominously deep in the foggy woods with a demonic smile in his face, iphone photo, creepy, low visibility creepy |
| decomposer | 000084 | 0.2609 | 0.8041 | futuristic futuristic city in the middle of a lake, 3 d render, cinematic lighting, cgsociety, pixiv |
| decomposer | 000088 | 0.2201 | 0.7789 | a cat with human teeth |
| decomposer | 000089 | 0.1843 | 0.7200 | zen futuristic city ink |
| decomposer | 000103 | 0.1836 | 0.7601 | a complex thick bifurcated robotic cnc surgical arm cybernetic symbiosis hybrid mri 3 d printer machine printing some highly detailed large ferrari motors in inspection laboratory control panel room, octane render, natural color scheme, f 1 8, octane render |
| decomposer | 000115 | 0.1916 | 0.7353 | sitting on a rock |
| decomposer | 000128 | 0.2325 | 0.5873 | wolf |
| decomposer | 000131 | 0.1356 | 0.6794 | a man taking a selfie from the top of the tallest tower in the world, studio lighting, beautiful city, wide lens |
| decomposer | 000139 | 0.1849 | 0.7465 | a paint of dan mumford |
| decomposer | 000144 | 0.1586 | 0.7148 | a bearded man in bulky mech armor that looks too big for his head |
| decomposer | 000158 | 0.1751 | 0.6837 | lt. liama from fortnite game |
