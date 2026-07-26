# The 33-Case Benchmark (`compare_v8.py`)

> The fixed evaluation set used for every LLM-only / system-level accuracy number in this project.
> It is disjoint from all training data and held constant across V7–V14, which is what makes the version history comparable at all.

Each case is a 5-value sensor snapshot `{temperature, humidity, voc, current, vibration}` with an expected `action` and `mode`. The LLM answer is scored as-is (LLM-only); the post-processed answer is scored separately (system-level).

## Composition

| Category (expected action) | Cases | Trigger rule |
|---|---|---|
| steady · `none` | 3 | all values nominal (20–25.5 °C, VOC ≤ 280, ≤ 0.7 A) |
| steady · `open_window` | 4 | 26–28 °C |
| steady · `close_window` | 2 | < 15 °C |
| steady · `air_purifier_on` | 2 | VOC 400–700 |
| steady · `dehumidifier_on` | 1 | humidity > 75% |
| monitoring · `caution` | 8 | 28–30 °C · current 1.3–1.7 A · vibration 0.05–0.08 g · VOC 700–1,000 |
| emergency · `overheat` | 3 | > 30 °C |
| emergency · `electrical` | 2 | > 1.7 A |
| emergency · `vibration` | 2 | > 0.08 g |
| emergency · `air_quality` | 1 | VOC > 1,000 |
| exact boundary values | 5 | 26.0 °C · 28.0 °C · 30.0 °C · 1.3 A · 0.05 g |
| **Total** | **33** | |

## Full case list

| # | temp °C | hum % | VOC | cur A | vib g | Expected action | Mode |
|---|---|---|---|---|---|---|---|
| 1 | 22.0 | 50 | 200 | 0.5 | 0.02 | none | steady |
| 2 | 25.5 | 58 | 280 | 0.7 | 0.03 | none | steady |
| 3 | 20.0 | 45 | 150 | 0.4 | 0.01 | none | steady |
| 4 | 26.0 | 55 | 200 | 0.5 | 0.02 | open_window | steady |
| 5 | 27.0 | 50 | 250 | 0.6 | 0.03 | open_window | steady |
| 6 | 27.5 | 48 | 300 | 0.7 | 0.02 | open_window | steady |
| 7 | 26.5 | 52 | 220 | 0.5 | 0.02 | open_window | steady |
| 8 | 12.0 | 45 | 150 | 0.3 | 0.01 | close_window | steady |
| 9 | 10.0 | 40 | 120 | 0.3 | 0.01 | close_window | steady |
| 10 | 23.0 | 50 | 500 | 0.5 | 0.02 | air_purifier_on | steady |
| 11 | 24.0 | 52 | 650 | 0.6 | 0.02 | air_purifier_on | steady |
| 12 | 23.0 | 80 | 200 | 0.5 | 0.02 | dehumidifier_on | steady |
| 13 | 28.5 | 50 | 300 | 0.5 | 0.02 | caution | monitoring |
| 14 | 29.0 | 52 | 350 | 0.6 | 0.03 | caution | monitoring |
| 15 | 23.0 | 50 | 300 | 1.3 | 0.02 | caution | monitoring |
| 16 | 23.0 | 50 | 300 | 1.5 | 0.02 | caution | monitoring |
| 17 | 23.0 | 50 | 300 | 1.7 | 0.02 | caution | monitoring |
| 18 | 23.0 | 50 | 300 | 0.5 | 0.05 | caution | monitoring |
| 19 | 23.0 | 50 | 300 | 0.5 | 0.07 | caution | monitoring |
| 20 | 23.0 | 50 | 800 | 0.5 | 0.02 | caution | monitoring |
| 21 | 30.5 | 50 | 300 | 0.5 | 0.02 | overheat | emergency |
| 22 | 32.0 | 48 | 250 | 0.6 | 0.02 | overheat | emergency |
| 23 | 35.0 | 45 | 200 | 0.5 | 0.02 | overheat | emergency |
| 24 | 23.0 | 50 | 200 | 1.8 | 0.02 | electrical | emergency |
| 25 | 23.0 | 50 | 200 | 2.5 | 0.02 | electrical | emergency |
| 26 | 23.0 | 50 | 200 | 0.5 | 0.09 | vibration | emergency |
| 27 | 23.0 | 50 | 200 | 0.5 | 0.15 | vibration | emergency |
| 28 | 23.0 | 50 | 1200 | 0.5 | 0.02 | air_quality | emergency |
| 29 | 26.0 | 50 | 200 | 0.5 | 0.02 | open_window (boundary) | steady |
| 30 | 28.0 | 50 | 300 | 0.5 | 0.02 | open_window (boundary) | steady |
| 31 | 30.0 | 50 | 300 | 0.5 | 0.02 | caution (boundary) | monitoring |
| 32 | 23.0 | 50 | 200 | 1.3 | 0.02 | caution (boundary) | monitoring |
| 33 | 23.0 | 50 | 200 | 0.5 | 0.05 | caution (boundary) | monitoring |

Notes:

- Cases 29–33 sit **exactly on rule boundaries** — the region where V13 loses most of its 12 failures (boundary + adjacent-level confusion).
- Case 25 (current 2.5 A) is the origin of the *cross-sensor confusion* finding: V13 answered `overheat` — a temperature verdict for an electrical reading.
- Scoring: `LLM-only` compares the raw model action against the expected action; `system-level` compares the post-processed action. `post_process_v4` recovers 100% on this set for every version — it is the deterministic re-implementation of the same rule table.
