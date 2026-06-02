# Approach→Strike success matrix — v12 DAgger (TRACKER §35.4)

> **Only the `approach` column is valid.** The real Strike checkpoint was lost
> in the §35.0 env wipe, so `--strike` here is a meaningless stand-in (the
> Approach checkpoint reused). `strike`/`full` columns are therefore all 0/5 by
> construction and say nothing about Strike.
>
> **Approach total: 129/435 (29.7%)** vs the prior best (v8 pure BC) 10/435.
> DAgger was trained on the 12 `central` keys; it generalizes to 45 keys here.

Approach: `checkpoints/approach_v12_dagger/dagger_best.pt`  
Strike: `checkpoints/approach_v12_dagger/dagger_best.pt` (stand-in — ignore)  
Trials per key: 5

| key | approach | strike | full |
|---|---|---|---|
| esc | 0/5 | 0/5 | 0/5 |
| f1 | 0/5 | 0/5 | 0/5 |
| f2 | 0/5 | 0/5 | 0/5 |
| f3 | 2/5 | 0/5 | 0/5 |
| f4 | 4/5 | 0/5 | 0/5 |
| f5 | 4/5 | 0/5 | 0/5 |
| f6 | 2/5 | 0/5 | 0/5 |
| f7 | 1/5 | 0/5 | 0/5 |
| f8 | 2/5 | 0/5 | 0/5 |
| f9 | 0/5 | 0/5 | 0/5 |
| f10 | 0/5 | 0/5 | 0/5 |
| f11 | 1/5 | 0/5 | 0/5 |
| f12 | 0/5 | 0/5 | 0/5 |
| prtsc | 0/5 | 0/5 | 0/5 |
| scrlk | 0/5 | 0/5 | 0/5 |
| pause | 0/5 | 0/5 | 0/5 |
| grave | 0/5 | 0/5 | 0/5 |
| 1 | 1/5 | 0/5 | 0/5 |
| 2 | 2/5 | 0/5 | 0/5 |
| 3 | 3/5 | 0/5 | 0/5 |
| 4 | 1/5 | 0/5 | 0/5 |
| 5 | 2/5 | 0/5 | 0/5 |
| 6 | 4/5 | 0/5 | 0/5 |
| 7 | 3/5 | 0/5 | 0/5 |
| 8 | 4/5 | 0/5 | 0/5 |
| 9 | 3/5 | 0/5 | 0/5 |
| 0 | 5/5 | 0/5 | 0/5 |
| minus | 3/5 | 0/5 | 0/5 |
| equal | 3/5 | 0/5 | 0/5 |
| backspace | 0/5 | 0/5 | 0/5 |
| ins | 0/5 | 0/5 | 0/5 |
| home | 0/5 | 0/5 | 0/5 |
| pgup | 0/5 | 0/5 | 0/5 |
| tab | 0/5 | 0/5 | 0/5 |
| q | 2/5 | 0/5 | 0/5 |
| w | 0/5 | 0/5 | 0/5 |
| e | 1/5 | 0/5 | 0/5 |
| r | 0/5 | 0/5 | 0/5 |
| t | 1/5 | 0/5 | 0/5 |
| y | 3/5 | 0/5 | 0/5 |
| u | 4/5 | 0/5 | 0/5 |
| i | 4/5 | 0/5 | 0/5 |
| o | 5/5 | 0/5 | 0/5 |
| p | 5/5 | 0/5 | 0/5 |
| lbracket | 5/5 | 0/5 | 0/5 |
| rbracket | 3/5 | 0/5 | 0/5 |
| backslash | 2/5 | 0/5 | 0/5 |
| del | 0/5 | 0/5 | 0/5 |
| end | 0/5 | 0/5 | 0/5 |
| pgdn | 0/5 | 0/5 | 0/5 |
| caps | 0/5 | 0/5 | 0/5 |
| a | 0/5 | 0/5 | 0/5 |
| s | 0/5 | 0/5 | 0/5 |
| d | 0/5 | 0/5 | 0/5 |
| f | 1/5 | 0/5 | 0/5 |
| g | 2/5 | 0/5 | 0/5 |
| h | 1/5 | 0/5 | 0/5 |
| j | 5/5 | 0/5 | 0/5 |
| k | 5/5 | 0/5 | 0/5 |
| l | 5/5 | 0/5 | 0/5 |
| semicolon | 5/5 | 0/5 | 0/5 |
| quote | 4/5 | 0/5 | 0/5 |
| enter | 2/5 | 0/5 | 0/5 |
| lshift | 0/5 | 0/5 | 0/5 |
| z | 0/5 | 0/5 | 0/5 |
| x | 0/5 | 0/5 | 0/5 |
| c | 0/5 | 0/5 | 0/5 |
| v | 0/5 | 0/5 | 0/5 |
| b | 0/5 | 0/5 | 0/5 |
| n | 0/5 | 0/5 | 0/5 |
| m | 1/5 | 0/5 | 0/5 |
| comma | 1/5 | 0/5 | 0/5 |
| period | 4/5 | 0/5 | 0/5 |
| slash | 5/5 | 0/5 | 0/5 |
| rshift | 1/5 | 0/5 | 0/5 |
| up | 0/5 | 0/5 | 0/5 |
| lctrl | 0/5 | 0/5 | 0/5 |
| win | 0/5 | 0/5 | 0/5 |
| lalt | 0/5 | 0/5 | 0/5 |
| space | 0/5 | 0/5 | 0/5 |
| ralt | 2/5 | 0/5 | 0/5 |
| fn | 0/5 | 0/5 | 0/5 |
| menu | 0/5 | 0/5 | 0/5 |
| rctrl | 0/5 | 0/5 | 0/5 |
| left | 0/5 | 0/5 | 0/5 |
| down | 0/5 | 0/5 | 0/5 |
| right | 0/5 | 0/5 | 0/5 |

**Full chain: 0/435 = 0.0%**
