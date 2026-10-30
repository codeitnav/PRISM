# Captioning Results - Task 5.1

- **Model:** `Salesforce/blip-image-captioning-large`
- **Split(s):** train, val, test (3220 images)
- **Wall clock:** 263.3s total, 0.08s/image (includes cache hits; a fully cached re-run is near-instant)
- **Caption length (words):** min 6, max 40, mean 13.7
- **Empty captions:** 0

## Caption vs. true prompt

The caption describes what is *in* the image; the true prompt is what generated it. The gap between the two columns is exactly the gap the decomposition model (Task 5.3) has to close - the caption grounds the subject, but carries almost none of the style/medium/quality vocabulary.

| id | Caption | True prompt |
|---|---|---|
| `000000` | painting of people walking down a cobblestone street with a starry sky in the background | walking in paris, orsay museum, midgnight, starry nights, van gogh, hyperrealism, astrophotography, nature, architecture |
| `000003` | there is a painting of a man in a brown coat and a yellow cape | jean reno painted by jan vermeer, aristocratic pose, high contrast |
| `000004` | there is a painting of a man walking in the rain | rainy street, homeless girl portrait photo, messy bleached hair, torn clothes, richard avedon, martin schoeller, victor ngai, id magazine, e |
| `000007` | there is a woman walking in the rain with an umbrella over her head | rainy street, homeless girl portrait photo, messy bleached hair, torn clothes, richard avedon, martin schoeller, victor ngai, id magazine, e |
| `000008` | there is a man that is standing in front of a sign that says don ' t who win is run | don't run with sissors |
| `000010` | there are many cubes that are stacked together on a red surface | random sculpture made of rectangle tetris, octane render, trending on artstation |
| `000011` | there is a drawing of a witch and a girl sitting on a stool | greg and wirt, greg's pet frog, and a happy dog, over the garden wall by patrick mchale, |
| `000012` | there are two robots that are standing next to each other | tin toys wooden cyborg |
| `000013` | there is a multicolored sculpture made of blocks on a gray surface | contemporary sculpture made of rectangle tetris, octane render, trending on artstation |
| `000016` | there is a painting of a group of minions dancing in the street | 1 7 0 0 s photo of minions rioting during french revolution 4 k hd vivid colorized rgb |
| `000017` | there is a painting of a girl standing next to a bear | neverending story, fantasy art painting |
| `000018` | this is a picture of a circular picture of a bunch of small pieces of glass | a happy bowl of ramen |
| `000019` | illustration of a group of people sitting around a dinner table with a full moon in the background | mid - autumn festival, moon, family gathering, wine bottle, mooncake, surreal, dramatic light, by victo ngai, by james jean, by rossdraws, f |
| `000020` | there is a painting of two lemons and a half of a lemon on a table | concept art still life of closeup lemon slice by 司 机, 橡 树 山 雨, 黄 凤 年 啊, ruo xin, rongzhen luo, fenghua zhong, ruan jia. fine details, trendi |
| `000022` | there is a digital painting of a clock with a sky in the background | a detailed portrait of made in abyss star compass, by victo ngai and justin gerard, digital art, realistic painting, very detailed, fantasy, |
