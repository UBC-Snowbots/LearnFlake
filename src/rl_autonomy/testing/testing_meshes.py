import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from keyboard_env import KeyboardEnv


def main():
    """Build the environment and launch the interactive viewer."""
    env = KeyboardEnv(
        render=False,
        randomize_keyboard_pos=False,
        log_contacts=True,
        horizon=100000,
    )
    env.reset()

    import mujoco.viewer
    model = env.sim.model._model
    data = env.sim.data._data

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = 0
        viewer.cam.lookat[:] = [-0.15, 0.0, 0.15]
        viewer.cam.distance = 0.8
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -30

        print("Viewer open — drag to rotate, scroll to zoom.")
        print("Keys are pressable — push them down with the actuator.")
        print("Close the window to exit.")

        while viewer.is_running():
            env.step([0] * 7)
            viewer.sync()

    env.close()


if __name__ == "__main__":
    main()
