RL Agent Schematic:

1. Feed agent data (concatenate and implement testing onto a ros2 topic for easy viewing)
- Teleoperated sequences
- Arm initial joint states
- Environmental parameters
- Knowledge of the URDF

2. Arm will train using existing training pipeline
- DAgger to aggregate teleop with existing loop

3. Evaluation involves the arm sending over joint velocities onto ROS2 topic

4. Visualize results in RViz2

---

Domain Randomization:

1. Full arm orientation differs
2. Arm initial position should always be homed

