# Approach→Strike success matrix — v16b residual RL (TRACKER §38) — APPROACH SOLVED

> **Best result: Approach 428/435 (98.4%), 86/87 keys at ≥80%** (residual-on-IK,
> pbrs_only reward, tube 0.15, keyboard (-0.10,-0.10)). 83 keys 5/5, 3 at 4/5,
> only `lctrl` (1/5) short; zero dead. 100k checkpoint = 431/435 (99.1%).
> Eval with `--residual --residual-tube 0.15 --keyboard-offset=-0.10,-0.10`.
>
> **`approach` column only.** Strike lost in §35.0 → `--strike` is a stand-in,
> so strike/full are 0/5. Full-chain M4 needs Strike retrained (§38.4).
> Ledger: v8 BC 2.3% → v15 DAgger 61.6% → **v16b residual 98.4%**.

Approach: `checkpoints/approach_v16b_residual_sparse/residual_final.pt`  
Strike: `checkpoints/approach_v16b_residual_sparse/residual_final.pt` (stand-in — ignore)  
Trials per key: 5

| key | approach | strike | full |
|---|---|---|---|
| esc | 5/5 | 0/5 | 0/5 |
| f1 | 5/5 | 0/5 | 0/5 |
| f2 | 5/5 | 0/5 | 0/5 |
| f3 | 5/5 | 0/5 | 0/5 |
| f4 | 5/5 | 0/5 | 0/5 |
| f5 | 5/5 | 0/5 | 0/5 |
| f6 | 5/5 | 0/5 | 0/5 |
| f7 | 5/5 | 0/5 | 0/5 |
| f8 | 5/5 | 0/5 | 0/5 |
| f9 | 5/5 | 0/5 | 0/5 |
| f10 | 5/5 | 0/5 | 0/5 |
| f11 | 5/5 | 0/5 | 0/5 |
| f12 | 5/5 | 0/5 | 0/5 |
| prtsc | 5/5 | 0/5 | 0/5 |
| scrlk | 5/5 | 0/5 | 0/5 |
| pause | 5/5 | 0/5 | 0/5 |
| grave | 5/5 | 0/5 | 0/5 |
| 1 | 5/5 | 0/5 | 0/5 |
| 2 | 5/5 | 0/5 | 0/5 |
| 3 | 5/5 | 0/5 | 0/5 |
| 4 | 5/5 | 0/5 | 0/5 |
| 5 | 5/5 | 0/5 | 0/5 |
| 6 | 5/5 | 0/5 | 0/5 |
| 7 | 5/5 | 0/5 | 0/5 |
| 8 | 5/5 | 0/5 | 0/5 |
| 9 | 5/5 | 0/5 | 0/5 |
| 0 | 5/5 | 0/5 | 0/5 |
| minus | 5/5 | 0/5 | 0/5 |
| equal | 5/5 | 0/5 | 0/5 |
| backspace | 5/5 | 0/5 | 0/5 |
| ins | 5/5 | 0/5 | 0/5 |
| home | 5/5 | 0/5 | 0/5 |
| pgup | 5/5 | 0/5 | 0/5 |
| tab | 5/5 | 0/5 | 0/5 |
| q | 5/5 | 0/5 | 0/5 |
| w | 5/5 | 0/5 | 0/5 |
| e | 5/5 | 0/5 | 0/5 |
| r | 5/5 | 0/5 | 0/5 |
| t | 5/5 | 0/5 | 0/5 |
| y | 5/5 | 0/5 | 0/5 |
| u | 5/5 | 0/5 | 0/5 |
| i | 5/5 | 0/5 | 0/5 |
| o | 5/5 | 0/5 | 0/5 |
| p | 5/5 | 0/5 | 0/5 |
| lbracket | 5/5 | 0/5 | 0/5 |
| rbracket | 5/5 | 0/5 | 0/5 |
| backslash | 5/5 | 0/5 | 0/5 |
| del | 5/5 | 0/5 | 0/5 |
| end | 5/5 | 0/5 | 0/5 |
| pgdn | 5/5 | 0/5 | 0/5 |
| caps | 5/5 | 0/5 | 0/5 |
| a | 5/5 | 0/5 | 0/5 |
| s | 5/5 | 0/5 | 0/5 |
| d | 5/5 | 0/5 | 0/5 |
| f | 5/5 | 0/5 | 0/5 |
| g | 5/5 | 0/5 | 0/5 |
| h | 5/5 | 0/5 | 0/5 |
| j | 5/5 | 0/5 | 0/5 |
| k | 5/5 | 0/5 | 0/5 |
| l | 5/5 | 0/5 | 0/5 |
| semicolon | 5/5 | 0/5 | 0/5 |
| quote | 5/5 | 0/5 | 0/5 |
| enter | 5/5 | 0/5 | 0/5 |
| lshift | 5/5 | 0/5 | 0/5 |
| z | 5/5 | 0/5 | 0/5 |
| x | 5/5 | 0/5 | 0/5 |
| c | 5/5 | 0/5 | 0/5 |
| v | 5/5 | 0/5 | 0/5 |
| b | 5/5 | 0/5 | 0/5 |
| n | 5/5 | 0/5 | 0/5 |
| m | 5/5 | 0/5 | 0/5 |
| comma | 5/5 | 0/5 | 0/5 |
| period | 5/5 | 0/5 | 0/5 |
| slash | 5/5 | 0/5 | 0/5 |
| rshift | 5/5 | 0/5 | 0/5 |
| up | 5/5 | 0/5 | 0/5 |
| lctrl | 1/5 | 0/5 | 0/5 |
| win | 4/5 | 0/5 | 0/5 |
| lalt | 5/5 | 0/5 | 0/5 |
| space | 4/5 | 0/5 | 0/5 |
| ralt | 5/5 | 0/5 | 0/5 |
| fn | 5/5 | 0/5 | 0/5 |
| menu | 5/5 | 0/5 | 0/5 |
| rctrl | 5/5 | 0/5 | 0/5 |
| left | 5/5 | 0/5 | 0/5 |
| down | 5/5 | 0/5 | 0/5 |
| right | 4/5 | 0/5 | 0/5 |

**Full chain: 0/435 = 0.0%**
