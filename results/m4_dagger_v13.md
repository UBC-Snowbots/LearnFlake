# Approach→Strike success matrix — v13 DAgger all-keys (TRACKER §35.6)

> **Only the `approach` column is valid.** Strike checkpoint was lost in the
> §35.0 env wipe; `--strike` is a meaningless stand-in (Approach ckpt reused),
> so `strike`/`full` are 0/5 by construction.
>
> **Approach total: 196/435 (45.1%)** vs v12 (DAgger central) 129/435 and v8
> (pure BC) 10/435. 29 keys at ≥80%; 28 keys at 0/5 are expert-unreachable.
> Note: `dagger_best` was selected by the 12-key central eval, a weak proxy for
> all-87 — so 196 is a lower bound for v13's rounds (TRACKER §35.7).

Approach: `checkpoints/approach_v13_dagger_all/dagger_best.pt`  
Strike: `checkpoints/approach_v13_dagger_all/dagger_best.pt` (stand-in — ignore)  
Trials per key: 5

| key | approach | strike | full |
|---|---|---|---|
| esc | 0/5 | 0/5 | 0/5 |
| f1 | 0/5 | 0/5 | 0/5 |
| f2 | 2/5 | 0/5 | 0/5 |
| f3 | 0/5 | 0/5 | 0/5 |
| f4 | 2/5 | 0/5 | 0/5 |
| f5 | 4/5 | 0/5 | 0/5 |
| f6 | 5/5 | 0/5 | 0/5 |
| f7 | 3/5 | 0/5 | 0/5 |
| f8 | 5/5 | 0/5 | 0/5 |
| f9 | 4/5 | 0/5 | 0/5 |
| f10 | 4/5 | 0/5 | 0/5 |
| f11 | 2/5 | 0/5 | 0/5 |
| f12 | 3/5 | 0/5 | 0/5 |
| prtsc | 4/5 | 0/5 | 0/5 |
| scrlk | 5/5 | 0/5 | 0/5 |
| pause | 4/5 | 0/5 | 0/5 |
| grave | 0/5 | 0/5 | 0/5 |
| 1 | 0/5 | 0/5 | 0/5 |
| 2 | 1/5 | 0/5 | 0/5 |
| 3 | 0/5 | 0/5 | 0/5 |
| 4 | 0/5 | 0/5 | 0/5 |
| 5 | 5/5 | 0/5 | 0/5 |
| 6 | 2/5 | 0/5 | 0/5 |
| 7 | 4/5 | 0/5 | 0/5 |
| 8 | 5/5 | 0/5 | 0/5 |
| 9 | 5/5 | 0/5 | 0/5 |
| 0 | 4/5 | 0/5 | 0/5 |
| minus | 5/5 | 0/5 | 0/5 |
| equal | 2/5 | 0/5 | 0/5 |
| backspace | 4/5 | 0/5 | 0/5 |
| ins | 3/5 | 0/5 | 0/5 |
| home | 3/5 | 0/5 | 0/5 |
| pgup | 4/5 | 0/5 | 0/5 |
| tab | 0/5 | 0/5 | 0/5 |
| q | 0/5 | 0/5 | 0/5 |
| w | 1/5 | 0/5 | 0/5 |
| e | 1/5 | 0/5 | 0/5 |
| r | 1/5 | 0/5 | 0/5 |
| t | 2/5 | 0/5 | 0/5 |
| y | 4/5 | 0/5 | 0/5 |
| u | 5/5 | 0/5 | 0/5 |
| i | 5/5 | 0/5 | 0/5 |
| o | 5/5 | 0/5 | 0/5 |
| p | 4/5 | 0/5 | 0/5 |
| lbracket | 3/5 | 0/5 | 0/5 |
| rbracket | 4/5 | 0/5 | 0/5 |
| backslash | 4/5 | 0/5 | 0/5 |
| del | 3/5 | 0/5 | 0/5 |
| end | 4/5 | 0/5 | 0/5 |
| pgdn | 4/5 | 0/5 | 0/5 |
| caps | 0/5 | 0/5 | 0/5 |
| a | 0/5 | 0/5 | 0/5 |
| s | 0/5 | 0/5 | 0/5 |
| d | 0/5 | 0/5 | 0/5 |
| f | 0/5 | 0/5 | 0/5 |
| g | 2/5 | 0/5 | 0/5 |
| h | 2/5 | 0/5 | 0/5 |
| j | 3/5 | 0/5 | 0/5 |
| k | 5/5 | 0/5 | 0/5 |
| l | 5/5 | 0/5 | 0/5 |
| semicolon | 5/5 | 0/5 | 0/5 |
| quote | 2/5 | 0/5 | 0/5 |
| enter | 3/5 | 0/5 | 0/5 |
| lshift | 0/5 | 0/5 | 0/5 |
| z | 0/5 | 0/5 | 0/5 |
| x | 0/5 | 0/5 | 0/5 |
| c | 0/5 | 0/5 | 0/5 |
| v | 0/5 | 0/5 | 0/5 |
| b | 0/5 | 0/5 | 0/5 |
| n | 1/5 | 0/5 | 0/5 |
| m | 0/5 | 0/5 | 0/5 |
| comma | 5/5 | 0/5 | 0/5 |
| period | 3/5 | 0/5 | 0/5 |
| slash | 0/5 | 0/5 | 0/5 |
| rshift | 2/5 | 0/5 | 0/5 |
| up | 2/5 | 0/5 | 0/5 |
| lctrl | 0/5 | 0/5 | 0/5 |
| win | 0/5 | 0/5 | 0/5 |
| lalt | 0/5 | 0/5 | 0/5 |
| space | 0/5 | 0/5 | 0/5 |
| ralt | 2/5 | 0/5 | 0/5 |
| fn | 3/5 | 0/5 | 0/5 |
| menu | 3/5 | 0/5 | 0/5 |
| rctrl | 2/5 | 0/5 | 0/5 |
| left | 2/5 | 0/5 | 0/5 |
| down | 0/5 | 0/5 | 0/5 |
| right | 0/5 | 0/5 | 0/5 |

**Full chain: 0/435 = 0.0%**
