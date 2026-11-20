from pathlib import Path
from math import atan2
import subprocess
import sys

import polars as pl
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

# Run: python3 main_measurements.py /path/to/your/rosbag
bag = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/ros2_data/rosbag2_2026_11_18-17_47_03')
output = Path('/ros2_data/measured_data.csv')

# The paths below are actual fields observed in the decoded ROS 2 messages.
MEASUREMENTS = {
    'nav_msgs/msg/Odometry': {
        'x_m': 'pose.pose.position.x',
        'y_m': 'pose.pose.position.y',
    },
    'can_zoe_msgs/msg/Smotion': {
        'vx_kmh': 'vel_x', 'vy_kmh': 'vel_y',
        'speed_kmh': 'vel', 'sideslip_deg': 'angle',
    },
    'sensor_msgs/msg/Imu': {
        'yaw_rate_rad_s': 'angular_velocity.z',
        'ax_m_s2': 'linear_acceleration.x',
        'ay_m_s2': 'linear_acceleration.y',
    },
    'novatel_oem7_msgs/msg/INSPVA': {
        'latitude_deg': 'latitude', 'longitude_deg': 'longitude',
        'azimuth_deg': 'azimuth',
        'north_velocity_m_s': 'north_velocity',
        'east_velocity_m_s': 'east_velocity',
        'roll_deg': 'roll', 'pitch_deg': 'pitch',
    },
    'can_zoe_msgs/msg/SteeringInfo': {
        'steering_wheel_raw': 'position_dae',
        'steering_torque_raw': 'couple_volant_mesure_dae',
    },
    'can_zoe_msgs/msg/AccelInfo': {
        'accelerator_1_raw': 'position_accel_piste_1_mesure',
        'accelerator_2_raw': 'position_accel_piste_2_mesure',
    },
    'can_zoe_msgs/msg/BrakeInfo': {
        'brake_pressure_raw': 'pression_frein_mesure',
        'brake_pedal_1_raw': 'position_frein_piste_1_mesure',
    },
    'can_zoe_msgs/msg/FrontWheelsSteering': {
        'wheel_fl_speed_raw': 'vit_roue_avg',
        'wheel_fr_speed_raw': 'vit_roue_avd',
    },
    'can_zoe_msgs/msg/RearWheelsSteering': {
        'wheel_rl_speed_raw': 'vit_roue_arg',
        'wheel_rr_speed_raw': 'vit_roue_ard',
    },
    'novatel_oem7_msgs/msg/CORRIMU': {
        'corrimu_yaw_rate_raw': 'yaw_rate',
        'corrimu_long_acc_raw': 'longitudinal_acc',
        'corrimu_lat_acc_raw': 'lateral_acc',
    },
}

# Load all custom message definitions directly from ROS 2, with no .msg files.
typestore = get_typestore(Stores.ROS2_HUMBLE)
types = {}
for name in subprocess.check_output(['ros2', 'interface', 'list'], text=True).splitlines():
    name = name.strip()
    if name.startswith(('can_zoe_msgs/msg/', 'novatel_oem7_msgs/msg/')):
        definition = subprocess.check_output(['ros2', 'interface', 'show', name], text=True)
        definition = '\n'.join(
            line for line in definition.splitlines()
            if line and not line[0].isspace()
        )
        types.update(get_types_from_msg(definition, name))
typestore.register(types)


def value_at(message, path):
    for field in path.split('.'):
        message = getattr(message, field)
    return message


rows = {name: [] for name in MEASUREMENTS}

with AnyReader([bag], default_typestore=typestore) as reader:
    for connection, timestamp, rawdata in reader.messages():
        name = connection.msgtype
        if name not in MEASUREMENTS:
            continue

        msg = reader.deserialize(rawdata, name)
        row = {'time_ns': timestamp}
        for column, path in MEASUREMENTS[name].items():
            row[column] = value_at(msg, path)

        if name == 'nav_msgs/msg/Odometry':
            q = msg.pose.pose.orientation
            row['yaw_rad'] = atan2(
                2 * (q.w * q.z + q.x * q.y),
                1 - 2 * (q.y * q.y + q.z * q.z)
            )

        rows[name].append(row)

# Use position measurements as the reference time series.
reference = 'nav_msgs/msg/Odometry'
if not rows[reference]:
    raise RuntimeError('The bag has no Odometry messages.')

data = pl.DataFrame(rows[reference]).sort('time_ns')

# Match the nearest sensor measurement within 0.1 second.
for name, measurements in rows.items():
    if name != reference and measurements:
        sensor_data = pl.DataFrame(measurements).sort('time_ns')
        data = data.join_asof(
            sensor_data, on='time_ns', strategy='nearest', tolerance=100_000_000
        )

# Velocities from Smotion are specified in km/h.
if 'vx_kmh' in data.columns:
    data = data.with_columns(
        (pl.col('vx_kmh') / 3.6).alias('vx_m_s'),
        (pl.col('vy_kmh') / 3.6).alias('vy_m_s'),
    )

data = data.with_columns(
    ((pl.col('time_ns') - pl.col('time_ns').first()) / 1e9).alias('time_s')
)
data = data.select(['time_s', 'time_ns'] + [c for c in data.columns if c not in ('time_s', 'time_ns')])

data.write_csv(output)
print(f'Saved {data.height} rows and {data.width} columns to {output}')