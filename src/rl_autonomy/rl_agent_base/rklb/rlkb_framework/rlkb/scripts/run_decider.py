from __future__ import annotations

import argparse
import sys
from typing import Dict, List

from rlkb.decider.backends import MockKeyPressBackend
from rlkb.decider.engine import DeciderEngine
from rlkb.decider.layouts import DEFAULT_KEY_LAYOUT
from rlkb.decider.strategy import build_default_registry


def parse_verification_plan(raw: str) -> Dict[str, List[bool]]:
    """
    Format: "2:false,true;5:false"
    """
    plan: Dict[str, List[bool]] = {}
    if not raw:
        return plan

    for group in raw.split(";"):
        key_id, _, outcomes = group.partition(":")
        if not key_id or not outcomes:
            raise ValueError(f"Invalid verification plan segment: {group}")
        parsed = []
        for token in outcomes.split(","):
            normalized = token.strip().lower()
            if normalized not in {"true", "false"}:
                raise ValueError(f"Unsupported verification token: {token}")
            parsed.append(normalized == "true")
        plan[key_id.strip()] = parsed
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Omega key-presser decider.")
    parser.add_argument("--code", required=True, help="Startup code as a digit string, interpreted by the selected decider.")
    parser.add_argument("--strategy", default="sequence", help="Decider strategy name.")
    parser.add_argument("--max-retries", type=int, default=1, help="Retries per key after a failed verification.")
    parser.add_argument(
        "--verification-plan",
        default="",
        help='Mock verification outcomes, e.g. "2:false,true;5:false".',
    )
    parser.add_argument("--show-layout", action="store_true", help="Print the default layout and exit.")
    parser.add_argument("--list-strategies", action="store_true", help="List available decider strategies and exit.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    registry = build_default_registry()

    if args.list_strategies:
        for name in registry.names():
            print(name)
        return

    if args.show_layout:
        for key_id, target in sorted(DEFAULT_KEY_LAYOUT.items()):
            print(f"{key_id}: position={target.position} description={target.description}")
        return

    strategy = registry.get(args.strategy)
    verification_plan = parse_verification_plan(args.verification_plan)
    backend = MockKeyPressBackend(verification_plan=verification_plan)

    engine = DeciderEngine(
        code=args.code,
        strategy=strategy,
        backend=backend,
        layout=DEFAULT_KEY_LAYOUT,
        max_retries=max(0, args.max_retries),
    )
    try:
        result = engine.run()
    except ValueError as exc:
        print(f"tool=Omega error={exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    print(f"tool=Omega code={result.code} strategy={result.strategy_name} success={result.success} final_state={result.final_state}")
    for step in result.completed_steps:
        print(
            f"step key={step.target.key_id} position={step.target.position} "
            f"attempts={step.attempts} verified={step.verified}"
        )
    if result.failed_step is not None:
        step = result.failed_step
        print(
            f"failed key={step.target.key_id} position={step.target.position} "
            f"attempts={step.attempts} verified={step.verified} message={step.message}"
        )

    print("transitions:")
    for transition in result.transitions:
        print(f"  {transition.previous_state} -> {transition.next_state}: {transition.reason}")


if __name__ == "__main__":
    main()
