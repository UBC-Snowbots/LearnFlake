#!/usr/bin/env python3
"""
Hierarchical SAC Training for Rover2026 Arm - Version 3
Shim script pointing to the new refactored package in src/rl_autonomy/
"""
import sys
import os

# Add src to path
ROOT = os.path.dirname(os.path.abspath(__file__))
# src/ is at ../.. relative to src/rl_autonomy/testing
SRC_PATH = os.path.abspath(os.path.join(ROOT, "..", ".."))

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from rl_autonomy.testing.main import main

if __name__ == "__main__":
    main()
