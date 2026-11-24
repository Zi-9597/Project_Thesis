# Project_Thesis

## ROS 2 Humble: which branch to test

Use **`dev_software`** for the experimental LPV MPC, tube constraint tightening,
and set-library examples described here. `main` is not the branch for this
experiment. Run these commands **on the host**, from the repository root:

```bash
git status --short
git switch dev_software
git branch --show-current
git log -1 --oneline
```

If Git refuses to switch because of local edits, preserve those edits before
continuing. A Docker build from this checkout sees local edits in the source
folders too. A separate clone sees only the commits it has fetched, so the
branch name alone does not guarantee that two checkouts contain the same code.

## Start Docker and rebuild after a branch change

For the first start, or if the Dockerfile or image dependencies changed:

```bash
docker compose up --build -d mpc_ros
docker compose logs -f mpc_ros
```

Wait for the startup builds of `can_zoe_msgs` and `mpc_package` to finish
successfully. Press `Ctrl+C` to stop following the logs, not the container.

**After every branch switch, rebuild and re-source the ROS 2 workspace before
testing again.** The default Compose setup mounts the host package folders into
`/ros2_ws/src`, but does not mount the repository's `.git` directory. Switch
branches on the **host**, not under `/ros2_ws/src`. The `build` and `install`
directories are persistent Docker volumes; switching branches does not refresh
the installed packages by itself.

For a source-only branch change, stop the old test node, switch branches on the
host, and restart the service to run the configured builds again:

```bash
docker compose restart mpc_ros
docker compose logs -f mpc_ros
```

If you instead use a full Git checkout inside a custom container, switch branches
in **that checkout** and rebuild the workspace that actually uses those sources.
To rebuild manually in an already-running Compose container, stop the test node
and make sure the startup build is not running, then use a fresh shell:

```bash
# On the host:
docker compose exec mpc_ros bash --noprofile --norc

# Inside the container:
cd /ros2_ws
source /opt/ros/humble/setup.bash && \
  colcon build --symlink-install --packages-up-to can_zoe_msgs && \
  source install/setup.bash && \
  colcon build --symlink-install --packages-select mpc_package && \
  source install/setup.bash
```

If a build fails, fix the error before launching the test. In every new test
shell, source the current workspace and check that ROS 2 sees the executable:

```bash
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
ros2 pkg executables mpc_package
```

### Experimental test without vehicle actuation

Use simulation, rosbag replay, or a bench with actuation disabled. Compose
uses host networking, so the container is not automatically isolated from
vehicle ROS 2 nodes. Choose an unused test domain (here `42` is an example),
set the same `ROS_DOMAIN_ID` in the rosbag or simulator shell, and remap both
outputs to test-only topics with no CAN bridge attached:

```bash
export ROS_DOMAIN_ID=42
ros2 run mpc_package mpc_dynamics --ros-args \
  -r /cmd_steering:=/mpc_test/cmd_steering \
  -r /cmd_acceleration:=/mpc_test/cmd_acceleration
```

The experimental configuration uses `tube_enabled = True` and
`LTI_sys = False` in `dynamics_command.py`. This enables tube tightening and
stage-by-stage LPV prediction once a previous solution is available. It is not
a verified real-time or full robust tube controller: the published command does
not include the ancillary feedback assumed in the tube prediction.

## Set library

See the [set library README](mpc_controller/mpc_package/sets_library/README.md)
for standalone Python examples covering intervals, zonotopes, set operations,
and plotting. These examples do not start ROS 2 or require a workspace build.
