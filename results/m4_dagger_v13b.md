# Approach→Strike success matrix — v13b **round 0 (BC baseline)** (TRACKER §35.8)

> This matrix is `dagger_best.pt` = **round 0**, i.e. the **BC baseline** (the
> noisy stratified in-loop eval mis-picked it as "best"). It scores **115/435
> (26.4%)** on all-87 — the pure-BC-on-all-keys baseline. The actual best
> checkpoint is **round 6 (full DAgger) = 200/435 (46.0%)**
> (`dagger_round_06.pt`); see TRACKER §35.8.
>
> `approach` column only — Strike is a meaningless stand-in (real Strike lost in
> §35.0). This file is kept as the **BC-vs-DAgger ablation** lower end.

Approach: `checkpoints/approach_v13b_dagger_strat/dagger_best.pt` (= round 0, BC baseline)  
Strike: `checkpoints/approach_v13b_dagger_strat/dagger_best.pt` (stand-in — ignore)  
Trials per key: 5

| key | approach | strike | full |
|---|---|---|---|
| esc | 0/5 | 0/5 | 0/5 |
| f1 | 1/5 | 0/5 | 0/5 |
| f2 | 2/5 | 0/5 | 0/5 |
| f3 | 0/5 | 0/5 | 0/5 |
| f4 | 1/5 | 0/5 | 0/5 |
| f5 | 1/5 | 0/5 | 0/5 |
| f6 | 5/5 | 0/5 | 0/5 |
| f7 | 4/5 | 0/5 | 0/5 |
| f8 | 3/5 | 0/5 | 0/5 |
| f9 | 3/5 | 0/5 | 0/5 |
| f10 | 3/5 | 0/5 | 0/5 |
| f11 | 3/5 | 0/5 | 0/5 |
| f12 | 2/5 | 0/5 | 0/5 |
| prtsc | 2/5 | 0/5 | 0/5 |
| scrlk | 2/5 | 0/5 | 0/5 |
| pause | 1/5 | 0/5 | 0/5 |
| grave | 1/5 | 0/5 | 0/5 |
| 1 | 0/5 | 0/5 | 0/5 |
| 2 | 0/5 | 0/5 | 0/5 |
| 3 | 1/5 | 0/5 | 0/5 |
| 4 | 2/5 | 0/5 | 0/5 |
| 5 | 0/5 | 0/5 | 0/5 |
| 6 | 2/5 | 0/5 | 0/5 |
| 7 | 1/5 | 0/5 | 0/5 |
| 8 | 4/5 | 0/5 | 0/5 |
| 9 | 2/5 | 0/5 | 0/5 |
| 0 | 1/5 | 0/5 | 0/5 |
| minus | 1/5 | 0/5 | 0/5 |
| equal | 3/5 | 0/5 | 0/5 |
| backspace | 2/5 | 0/5 | 0/5 |
| ins | 3/5 | 0/5 | 0/5 |
| home | 3/5 | 0/5 | 0/5 |
| pgup | 2/5 | 0/5 | 0/5 |
| tab | 1/5 | 0/5 | 0/5 |
| q | 0/5 | 0/5 | 0/5 |
| w | 0/5 | 0/5 | 0/5 |
| e | 1/5 | 0/5 | 0/5 |
| r | 0/5 | 0/5 | 0/5 |
| t | 2/5 | 0/5 | 0/5 |
| y | 0/5 | 0/5 | 0/5 |
| u | 2/5 | 0/5 | 0/5 |
| i | 3/5 | 0/5 | 0/5 |
| o | 1/5 | 0/5 | 0/5 |
| p | 4/5 | 0/5 | 0/5 |
| lbracket | 1/5 | 0/5 | 0/5 |
| rbracket | 2/5 | 0/5 | 0/5 |
| backslash | 0/5 | 0/5 | 0/5 |
| del | 3/5 | 0/5 | 0/5 |
| end | 2/5 | 0/5 | 0/5 |
| pgdn | 1/5 | 0/5 | 0/5 |
| caps | 0/5 | 0/5 | 0/5 |
| a | 0/5 | 0/5 | 0/5 |
| s | 0/5 | 0/5 | 0/5 |
| d | 0/5 | 0/5 | 0/5 |
| f | 0/5 | 0/5 | 0/5 |
| g | 1/5 | 0/5 | 0/5 |
| h | 0/5 | 0/5 | 0/5 |
| j | 1/5 | 0/5 | 0/5 |
| k | 3/5 | 0/5 | 0/5 |
| l | 4/5 | 0/5 | 0/5 |
| semicolon | 0/5 | 0/5 | 0/5 |
| quote | 2/5 | 0/5 | 0/5 |
| enter | 3/5 | 0/5 | 0/5 |
| lshift | 0/5 | 0/5 | 0/5 |
| z | 0/5 | 0/5 | 0/5 |
| x | 0/5 | 0/5 | 0/5 |
| c | 0/5 | 0/5 | 0/5 |
| v | 0/5 | 0/5 | 0/5 |
| b | 0/5 | 0/5 | 0/5 |
| n | 0/5 | 0/5 | 0/5 |
| m | 2/5 | 0/5 | 0/5 |
| comma | 3/5 | 0/5 | 0/5 |
| period | 2/5 | 0/5 | 0/5 |
| slash | 2/5 | 0/5 | 0/5 |
| rshift | 0/5 | 0/5 | 0/5 |
| up | 0/5 | 0/5 | 0/5 |
| lctrl | 0/5 | 0/5 | 0/5 |
| win | 0/5 | 0/5 | 0/5 |
| lalt | 0/5 | 0/5 | 0/5 |
| space | 0/5 | 0/5 | 0/5 |
| ralt | 0/5 | 0/5 | 0/5 |
| fn | 1/5 | 0/5 | 0/5 |
| menu | 3/5 | 0/5 | 0/5 |
| rctrl | 1/5 | 0/5 | 0/5 |
| left | 2/5 | 0/5 | 0/5 |
| down | 0/5 | 0/5 | 0/5 |
| right | 1/5 | 0/5 | 0/5 |

**Full chain: 0/435 = 0.0%**
