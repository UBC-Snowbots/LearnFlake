from .accuracy_report import (
    build_controlled_selection_suite,
    build_engine_reliability_suite,
    build_noisy_selection_suite,
    generate_accuracy_report,
)
from .backends import KeyPressBackend, MockKeyPressBackend
from .layout import DEFAULT_KEY_LAYOUT
from .metrics import (
    EngineAccuracyMetrics,
    EngineScenario,
    SelectionCase,
    SelectionMetrics,
    evaluate_engine_scenarios,
    evaluate_target_selection,
)
from .omega import EngineState, KeyboardDeciderEngine, SequentialCodeDecider
from .perception import KeyDetection, KeyDetector, MockKeyDetector, PixelBBox, TargetKeySelector, TargetSelection
from .sim_perception_eval import (
    DetectionMetrics,
    OracleProjectionDetector,
    SimEvalConfig,
    bbox_iou,
    build_env,
    evaluate_detector_on_sim,
    format_detection_report,
    match_detections,
)
from .task_types import KeyLayout, KeyTarget, RunResult, StateTransition, StepResult

__all__ = [
    "DEFAULT_KEY_LAYOUT",
    "DetectionMetrics",
    "EngineState",
    "EngineAccuracyMetrics",
    "EngineScenario",
    "build_controlled_selection_suite",
    "build_engine_reliability_suite",
    "build_noisy_selection_suite",
    "generate_accuracy_report",
    "KeyDetection",
    "KeyDetector",
    "KeyLayout",
    "KeyPressBackend",
    "KeyboardDeciderEngine",
    "KeyTarget",
    "MockKeyPressBackend",
    "MockKeyDetector",
    "PixelBBox",
    "OracleProjectionDetector",
    "RunResult",
    "SimEvalConfig",
    "SelectionCase",
    "SelectionMetrics",
    "SequentialCodeDecider",
    "StateTransition",
    "TargetKeySelector",
    "TargetSelection",
    "StepResult",
    "bbox_iou",
    "build_env",
    "evaluate_engine_scenarios",
    "evaluate_detector_on_sim",
    "evaluate_target_selection",
    "format_detection_report",
    "match_detections",
]
