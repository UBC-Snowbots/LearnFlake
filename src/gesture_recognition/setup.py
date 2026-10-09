from glob import glob

from setuptools import find_packages, setup

package_name = "gesture_recognition"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "numpy", "opencv-python-headless"],
    extras_require={"pose": ["ultralytics>=8.3"]},
    zip_safe=True,
    maintainer="Pranav",
    maintainer_email="pranavrajeshkrishnan06@gmail.com",
    description="Beckoning / gesture recognition from the rover camera.",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "gesture_node = gesture_recognition.ros_node:main",
            "gesture-detect = gesture_recognition.cli:main",
        ],
    },
)
