# Baseline Evaluation (Task 4.3)

Both baselines (PEZ, Task 4.1; naive CLIP-tag, Task 4.2) scored through the same harness (`ml/app/eval.py`) on the same 20 test-split images.

**Not included:** component-wise precision/recall/F1 against weak structured labels - Task 1.3 (weak labeling) hasn't landed yet. Extend this harness once it does.

## Summary

| Reconstructor | Mean CLIP-score | Mean BERTScore F1 | Mean latency (s/image) |
|---|---|---|---|
| pez | 0.2964 | 0.7188 | 67.98 |
| clip_tag | 0.2355 | 0.7404 | 1.64 |

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
