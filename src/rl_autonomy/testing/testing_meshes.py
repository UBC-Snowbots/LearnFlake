"""
Visualize the Gaming_Keyboard OBJ mesh in a MuJoCo viewer.

Loads the OBJ from meshes/, scales it to real-world dimensions,
and displays it on a table surface using the MuJoCo interactive viewer.

Usage
-----
    python testing_meshes.py
"""

import os
import mujoco
import mujoco.viewer

MESH_DIR = os.path.join(os.path.dirname(__file__), "..", "meshes")
OBJ_PATH = os.path.join(MESH_DIR, "Gaming_Keyboard.obj")

# The OBJ is in mm-scale coords (~654 x 2004 x 160 units).
# A real Redragon K552 TKL is roughly 0.355 x 0.130 x 0.035 m.
# The mesh's longest axis (Z in OBJ) maps to keyboard width (~355mm).
# Scale factor: 0.355 / 2004 ≈ 0.000177
MESH_SCALE = 0.000177

MJCF_XML = f"""
<mujoco model="keyboard_viewer">
  <compiler meshdir="{MESH_DIR}"/>

  <asset>
    <mesh name="keyboard_mesh"
          file="Gaming_Keyboard.stl"
          scale="{MESH_SCALE} {MESH_SCALE} {MESH_SCALE}"
          inertia="shell"/>
    <texture name="grid" type="2d" builtin="checker"
             width="512" height="512"
             rgb1="0.9 0.9 0.9" rgb2="0.7 0.7 0.7"/>
    <material name="floor_mat" texture="grid"
              texrepeat="4 4" reflectance="0.1"/>
    <material name="keyboard_mat"
              rgba="0.85 0.85 0.85 1"
              specular="0.5" shininess="0.8"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="0.5 0.5 1.0" dir="-0.5 -0.5 -1" diffuse="0.4 0.4 0.4"/>

    <!-- Floor -->
    <geom type="plane" size="1 1 0.01" material="floor_mat"/>

    <!-- Table surface -->
    <geom name="table" type="box" pos="0 0 0.4" size="0.4 0.3 0.02"
          rgba="0.45 0.3 0.2 1"/>

    <!-- Keyboard mesh (static geom in worldbody — no inertia needed) -->
    <geom name="keyboard" type="mesh" mesh="keyboard_mesh"
          pos="0 0 0.425" material="keyboard_mat"
          contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>
"""


def main():
    """Load the keyboard mesh in MuJoCo and launch the interactive viewer."""
    if not os.path.exists(OBJ_PATH):
        raise FileNotFoundError(f"Keyboard mesh not found: {OBJ_PATH}")

    model = mujoco.MjModel.from_xml_string(MJCF_XML)
    data = mujoco.MjData(model)

    print("Launching MuJoCo viewer — close the window to exit.")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
