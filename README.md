# Project_Thesis

Experimental ROS 2 Humble LPV MPC for the Renault Zoe, with tube-based
constraint tightening. Use the **`dev_software`** branch for this version.

## Build and run

From the repository root on your host, build and start Docker once:

```bash
git switch dev_software
docker compose up --build -d mpc_ros
```

**After switching branches, rebuild the ROS 2 packages before testing again.**
Stop the test node, switch branches on the **host** (Docker mounts the host
source folders), then run:

```bash
docker compose restart mpc_ros
```

Compose rebuilds `can_zoe_msgs` and `mpc_package` on startup. Check that both
builds finished with `docker compose logs mpc_ros`, then open a new shell with
`docker compose exec mpc_ros bash` to load the rebuilt workspace. Only rebuild
the Docker image (`up --build`) if the Dockerfile or image dependencies changed.

Test in simulation, with rosbag replay, or on a bench with vehicle actuation disabled.

## Project folders

| Folder | Contents |
| --- | --- |
| [`mpc_controller/`](mpc_controller/) | The `mpc_package` ROS 2 package, including its build configuration and tests. |
| [`mpc_controller/mpc_package/`](mpc_controller/mpc_package/) | The controller node (`dynamics_command.py`), the OSQP solver interface (`class_mpc.py`), and the vehicle model (`utils_system.py`). |
| [`mpc_controller/mpc_package/sets_library/`](mpc_controller/mpc_package/sets_library/) | Intervals and zonotopes used for set calculations. The [set library README](mpc_controller/mpc_package/sets_library/README.md) shows how to test operations and plot zonotopes without ROS 2. |
| [`pretil-zoe-ze50-can-msgs/`](pretil-zoe-ze50-can-msgs/) | The `can_zoe_msgs` ROS 2 message definitions and bridge mapping for the Zoe. |
| [`data/`](data/) | Recorded ROS 2 bag data, a measurement CSV, and a Python script. Mounted at `/ros2_data` in Docker. |
| [`.vscode/`](.vscode/) | Workspace settings for VS Code. |
