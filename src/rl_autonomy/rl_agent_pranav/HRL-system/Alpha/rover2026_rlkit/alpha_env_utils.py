from __future__ import annotations

import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Tuple, Type, TypeVar


T = TypeVar("T")


def resolve_alpha_paths() -> Tuple[Path, Path]:
    """
    Returns:
      - RoboSuite source path
      - rlkb framework path
    """
    here = Path(__file__).resolve().parent
    repo_root = here.parents[5]
    robosuite_path = repo_root / "src" / "external_pkgs" / "RoboSuite"
    rlkb_path = repo_root / "src" / "rl_autonomy" / "rl_agent_base" / "rklb" / "rlkb_framework"
    return robosuite_path, rlkb_path


def ensure_alpha_import_paths() -> None:
    robosuite_path, rlkb_path = resolve_alpha_paths()
    for path in (robosuite_path, rlkb_path):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def migrate_legacy_env_config(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backward compatibility for older policy checkpoints which stored:
    - success_threshold_m -> standoff_tolerance_m
    - success_bonus -> standoff_bonus
    """
    migrated = dict(raw_config or {})
    if "standoff_tolerance_m" not in migrated and "success_threshold_m" in migrated:
        migrated["standoff_tolerance_m"] = migrated["success_threshold_m"]
    if "standoff_bonus" not in migrated and "success_bonus" in migrated:
        migrated["standoff_bonus"] = migrated["success_bonus"]
    migrated.pop("success_threshold_m", None)
    migrated.pop("success_bonus", None)
    return migrated


def config_from_payload(config_cls: Type[T], raw_config: Dict[str, Any]) -> T:
    migrated = migrate_legacy_env_config(raw_config)
    valid = {field.name for field in fields(config_cls)}
    filtered = {key: value for key, value in migrated.items() if key in valid}
    return config_cls(**filtered)
