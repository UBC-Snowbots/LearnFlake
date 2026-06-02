# Approach→Strike success matrix — v15 DAgger @ tuned keyboard pos (TRACKER §37)

> **Best result to date. `approach` column only** (Strike lost in §35.0; `--strike`
> is a stand-in → strike/full all 0). Evaluated with `--keyboard-offset=-0.10,-0.10`
> (must match training).
>
> **Approach: 268/435 (61.6%)** — vs v13b (default pos) 200/435 and v8 (BC) 10/435.
> 84/87 keys reachable, 36 at ≥80%, only **3** dead (`scrlk del right`). The whole
> left side (`a s d f g z x c …`) is now ≥80%. Remaining lever: residual RL to push
> the 48 mid keys (1–3/5) past 80% (TRACKER §37.1).

Approach: `checkpoints/approach_v15_dagger_kbpos/dagger_round_06.pt`  
Strike: `checkpoints/approach_v15_dagger_kbpos/dagger_round_06.pt` (stand-in — ignore)  
Trials per key: 5

| key | approach | strike | full |
|---|---|---|---|
| esc | 3/5 | 0/5 | 0/5 |
| f1 | 5/5 | 0/5 | 0/5 |
| f2 | 4/5 | 0/5 | 0/5 |
| f3 | 2/5 | 0/5 | 0/5 |
| f4 | 4/5 | 0/5 | 0/5 |
| f5 | 4/5 | 0/5 | 0/5 |
| f6 | 5/5 | 0/5 | 0/5 |
| f7 | 1/5 | 0/5 | 0/5 |
| f8 | 4/5 | 0/5 | 0/5 |
| f9 | 3/5 | 0/5 | 0/5 |
| f10 | 4/5 | 0/5 | 0/5 |
| f11 | 1/5 | 0/5 | 0/5 |
| f12 | 1/5 | 0/5 | 0/5 |
| prtsc | 1/5 | 0/5 | 0/5 |
| scrlk | 0/5 | 0/5 | 0/5 |
| pause | 2/5 | 0/5 | 0/5 |
| grave | 2/5 | 0/5 | 0/5 |
| 1 | 1/5 | 0/5 | 0/5 |
| 2 | 2/5 | 0/5 | 0/5 |
| 3 | 3/5 | 0/5 | 0/5 |
| 4 | 3/5 | 0/5 | 0/5 |
| 5 | 5/5 | 0/5 | 0/5 |
| 6 | 5/5 | 0/5 | 0/5 |
| 7 | 4/5 | 0/5 | 0/5 |
| 8 | 4/5 | 0/5 | 0/5 |
| 9 | 3/5 | 0/5 | 0/5 |
| 0 | 3/5 | 0/5 | 0/5 |
| minus | 4/5 | 0/5 | 0/5 |
| equal | 2/5 | 0/5 | 0/5 |
| backspace | 2/5 | 0/5 | 0/5 |
| ins | 2/5 | 0/5 | 0/5 |
| home | 3/5 | 0/5 | 0/5 |
| pgup | 1/5 | 0/5 | 0/5 |
| tab | 2/5 | 0/5 | 0/5 |
| q | 4/5 | 0/5 | 0/5 |
| w | 5/5 | 0/5 | 0/5 |
| e | 5/5 | 0/5 | 0/5 |
| r | 5/5 | 0/5 | 0/5 |
| t | 4/5 | 0/5 | 0/5 |
| y | 3/5 | 0/5 | 0/5 |
| u | 5/5 | 0/5 | 0/5 |
| i | 2/5 | 0/5 | 0/5 |
| o | 3/5 | 0/5 | 0/5 |
| p | 5/5 | 0/5 | 0/5 |
| lbracket | 3/5 | 0/5 | 0/5 |
| rbracket | 1/5 | 0/5 | 0/5 |
| backslash | 3/5 | 0/5 | 0/5 |
| del | 0/5 | 0/5 | 0/5 |
| end | 3/5 | 0/5 | 0/5 |
| pgdn | 1/5 | 0/5 | 0/5 |
| caps | 3/5 | 0/5 | 0/5 |
| a | 4/5 | 0/5 | 0/5 |
| s | 4/5 | 0/5 | 0/5 |
| d | 4/5 | 0/5 | 0/5 |
| f | 5/5 | 0/5 | 0/5 |
| g | 5/5 | 0/5 | 0/5 |
| h | 4/5 | 0/5 | 0/5 |
| j | 2/5 | 0/5 | 0/5 |
| k | 5/5 | 0/5 | 0/5 |
| l | 4/5 | 0/5 | 0/5 |
| semicolon | 5/5 | 0/5 | 0/5 |
| quote | 3/5 | 0/5 | 0/5 |
| enter | 2/5 | 0/5 | 0/5 |
| lshift | 4/5 | 0/5 | 0/5 |
| z | 5/5 | 0/5 | 0/5 |
| x | 5/5 | 0/5 | 0/5 |
| c | 5/5 | 0/5 | 0/5 |
| v | 3/5 | 0/5 | 0/5 |
| b | 4/5 | 0/5 | 0/5 |
| n | 5/5 | 0/5 | 0/5 |
| m | 2/5 | 0/5 | 0/5 |
| comma | 3/5 | 0/5 | 0/5 |
| period | 2/5 | 0/5 | 0/5 |
| slash | 2/5 | 0/5 | 0/5 |
| rshift | 1/5 | 0/5 | 0/5 |
| up | 3/5 | 0/5 | 0/5 |
| lctrl | 3/5 | 0/5 | 0/5 |
| win | 4/5 | 0/5 | 0/5 |
| lalt | 5/5 | 0/5 | 0/5 |
| space | 3/5 | 0/5 | 0/5 |
| ralt | 1/5 | 0/5 | 0/5 |
| fn | 2/5 | 0/5 | 0/5 |
| menu | 2/5 | 0/5 | 0/5 |
| rctrl | 2/5 | 0/5 | 0/5 |
| left | 3/5 | 0/5 | 0/5 |
| down | 2/5 | 0/5 | 0/5 |
| right | 0/5 | 0/5 | 0/5 |

**Full chain: 0/435 = 0.0%**
