FROM osrf/ros:humble-desktop

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV ROS_WS=/ros2_ws

# System tools, ROS build tools, and Python dependencies used by the MPC package.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions \
    python3-pip \
    python3-rosdep \
    python3-vcstool \
    build-essential \
    git \
    ros-${ROS_DISTRO}-rosidl-default-generators \
    ros-${ROS_DISTRO}-rosidl-default-runtime \
    ros-${ROS_DISTRO}-geometry-msgs \
    ros-${ROS_DISTRO}-nav-msgs \
    ros-${ROS_DISTRO}-novatel-oem7-msgs \
    ros-${ROS_DISTRO}-sensor-msgs \
    ros-${ROS_DISTRO}-tf2-ros-py \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    osqp \
    matplotlib

# Initialize rosdep if the base image does not already contain a rosdep database.
RUN rosdep init 2>/dev/null || true && rosdep update

WORKDIR ${ROS_WS}

# Copy this repository as ROS 2 workspace sources.
COPY mpc_controller ${ROS_WS}/src/mpc_controller
COPY pretil-zoe-ze50-can-msgs ${ROS_WS}/src/pretil-zoe-ze50-can-msgs

# Install dependencies that are available from apt/rosdep.
# Some dependencies in package.xml may be project-specific or unavailable publicly,
# so they are skipped here and can be added later if you provide those packages.
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
    && rosdep install --from-paths src --ignore-src -r -y \
        --skip-keys="av_path_speed_analyzer_interface"

# Build message package first, then the MPC package.
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
    && colcon build --symlink-install --packages-up-to can_zoe_msgs \
    && source install/setup.bash \
    && colcon build --symlink-install --packages-select mpc_package

# Source ROS and this workspace automatically for interactive shells.
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc \
    && echo "if [ -f ${ROS_WS}/install/setup.bash ]; then source ${ROS_WS}/install/setup.bash; fi" >> /root/.bashrc

CMD ["bash"]
