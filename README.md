# Frankx Installation Guide 
```bash
Cath2 not found: clone https://github.com/catchorg/Catch2.git and switch to v2.x branch
eigen3 not found: upgrade eigen to version 3.4 and install
pybind11 cmake not found: pip install pybind11 and add the install path to frankx/affx, frankx/ruckig, frankx CMakeLists.txt ex: set(pybind11_DIR your_pybind11_install_path)   
```

# Build up Guide

## Run as ROS2 node
```bash
cd ~/franka_robotic_arm
colcon build
source install/setup.bash
ros2 run frankx_py motion_test_node 
```

## Run as python script (Recommend)
```bash
cd ~/franka_robotic_arm
colcon build
source install/setup.bash
cd src/frankx/frankx
python3 motion_test.py
```



